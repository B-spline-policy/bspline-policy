"""Utilities for B-spline action chunks.

The policy-facing representation is a dense parameter matrix:

    (chunk_size + 2 * degree, 1 + action_dim)

Column 0 stores the knot vector. The remaining columns store B-spline
control points for the regular action dimensions.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import torch
from filelock import FileLock
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline

from bspline_policy.common.knots import decode_relative_knots, encode_relative_knots
from diffusion_policy.common.replay_buffer import ReplayBuffer


class ScipyBSplineCompression:
    """Fit a multi-dimensional trajectory with a reduced-knot B-spline."""

    def __init__(self, degree: int = 3):
        self.degree = int(degree)
        self.spline = None
        self.knots = None

    def compress(
        self,
        data: np.ndarray,
        max_error: float = 0.01,
        verbose: bool = False,
        s: float = 1e-12,
    ) -> np.ndarray:
        t = np.arange(len(data))
        last_knots = None
        last_error = None
        for knots in generate_knots(t, data, s=s):
            spl = make_lsq_spline(t, data, knots)
            pred_data = spl(t)
            error = np.abs(pred_data - data).max()
            last_knots = knots
            last_error = error
            if error < max_error:
                self.knots = knots
                self.spline = spl
                break

        if self.knots is None:
            print(
                "Failing to compress trajectory with max error "
                f"{max_error}, use min error we can find. Error is {last_error}. "
                "You can try to increase the s value."
            )
            self.knots = last_knots
            self.spline = make_lsq_spline(t, data, self.knots)

        if verbose:
            print(f"compression ratio: {len(self.knots) / len(t)}")

        return self.knots


def extract_unique_knots(t_full: np.ndarray, degree: int) -> np.ndarray:
    """Extract the unique knot span from FITPACK's repeated-boundary format."""
    return t_full[degree:-degree]


def chunk_bspline_trajectory(
    compressor: ScipyBSplineCompression,
    chunk_size: int = 8,
    stride: Optional[int] = None,
    episode_length: Optional[int] = None,
    verbose: bool = False,
) -> list[dict]:
    """Split a fitted B-spline into fixed-size parameter chunks."""
    del episode_length
    if compressor.spline is None:
        raise ValueError("Please call compress() before chunking")

    if stride is None:
        stride = chunk_size - 1

    degree = compressor.degree
    t_full, c_full, _ = compressor.spline.tck
    unique_t = extract_unique_knots(t_full, degree)
    n_unique = len(unique_t)
    chunks = []

    if verbose:
        print(
            f"B-spline chunking: len(t)={len(t_full)}, len(c)={len(c_full)}, "
            f"degree={degree}, unique_knots={n_unique}, chunk_size={chunk_size}, "
            f"stride={stride}"
        )

    for start_idx in range(0, n_unique - 1, stride):
        first_pos = start_idx + degree
        last_pos = start_idx + chunk_size + degree

        t_start = max(0, first_pos - degree)
        t_end = min(len(t_full), last_pos + degree)

        chunk_t = t_full[t_start:t_end]
        chunk_c = c_full[t_start:t_end]
        expected_len = chunk_size + 2 * degree

        if len(chunk_t) < expected_len:
            chunk_t = np.concatenate(
                [chunk_t, np.full(expected_len - len(chunk_t), chunk_t[-1])]
            )
        if len(chunk_c) < expected_len:
            pad = np.repeat(chunk_c[-1:], expected_len - len(chunk_c), axis=0)
            chunk_c = np.concatenate([chunk_c, pad], axis=0)

        if len(chunk_t) != expected_len:
            raise AssertionError("chunk_t length should equal chunk_size + 2 * degree")
        if len(chunk_c) != expected_len:
            raise AssertionError("chunk_c length should equal chunk_size + 2 * degree")

        chunks.append({"t": chunk_t, "c": chunk_c, "k": degree})

    return chunks


def make_bspline_sampler_cache_path(
    base_path: str,
    episode_mask: np.ndarray,
    key_first_k: Optional[dict],
    chunk_size: int,
    degree: int,
    max_error: float,
    stride: int,
    n_action_steps: int,
    n_action_channels: int,
    relative_knots: bool = False,
) -> str:
    hasher = hashlib.sha1()
    hasher.update(np.ascontiguousarray(episode_mask.astype(np.uint8)).tobytes())
    if key_first_k:
        for key, value in sorted(key_first_k.items()):
            hasher.update(f"{key}:{value}".encode("utf-8"))
    hasher.update(
        (
            f"chunk={chunk_size}|degree={degree}|err={max_error}|stride={stride}|"
            f"steps={n_action_steps}|channels={n_action_channels}|"
            f"relative_knots={int(relative_knots)}"
        ).encode("utf-8")
    )
    digest = hasher.hexdigest()[:16]
    return f"{base_path}.bspline_sampler_{digest}.npz"


class BSplineChunkSampler:
    """Preprocess replay-buffer actions into fixed B-spline action chunks."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        chunk_size: int = 8,
        degree: int = 3,
        max_error: float = 0.01,
        stride: int = 1,
        keys: Optional[list] = None,
        key_first_k: Optional[dict] = None,
        episode_mask: Optional[np.ndarray] = None,
        action_key: str = "action",
        n_action_steps: int = 14,
        n_action_channels: int = 7,
        relative_knots: bool = False,
        cache_path: Optional[str] = None,
    ):
        if keys is None:
            keys = list(replay_buffer.keys())
        if key_first_k is None:
            key_first_k = {}
        max_first_k = max(key_first_k.values(), default=1)
        if max_first_k < 1:
            max_first_k = 1

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        self.replay_buffer = replay_buffer
        self.keys = list(keys)
        self.key_first_k = key_first_k
        self.chunk_size = int(chunk_size)
        self.degree = int(degree)
        self.max_error = float(max_error)
        self.stride = int(stride)
        self.action_key = action_key
        self.n_action_steps = int(n_action_steps)
        self.n_action_channels = int(n_action_channels)
        self.relative_knots = bool(relative_knots)
        self.cache_path = cache_path

        expected_knot_length = self.chunk_size + 2 * self.degree
        if self.n_action_steps != expected_knot_length:
            print(
                f"Warning: n_action_steps ({self.n_action_steps}) != "
                f"chunk_size + 2*degree ({expected_knot_length})"
            )

        if cache_path is not None:
            cache_lock_path = cache_path + ".lock"
            with FileLock(cache_lock_path):
                if os.path.exists(cache_path):
                    self._load_cache(cache_path)
                    return
                self._preprocess_chunks(
                    replay_buffer, episode_ends, episode_mask, max_first_k
                )
                self._save_cache(cache_path)
                return

        self._preprocess_chunks(replay_buffer, episode_ends, episode_mask, max_first_k)

    def _load_cache(self, cache_path: str) -> None:
        print(f"Loading cached B-spline chunks from Disk: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            self.all_actions = data["all_actions"]
            self.timestep_to_chunk = data["timestep_to_chunk"]
            self.valid_timesteps = data["valid_timesteps"]
            self.episode_ends = data["episode_ends"]
            self.episode_mask = data["episode_mask"].astype(bool)
            self.chunks_per_episode = data["chunks_per_episode"].tolist()
            self.episode_starts = data["episode_starts"].tolist()
        print(
            f"Loaded cached B-spline chunks: {len(self.all_actions)} chunks, "
            f"{len(self.valid_timesteps)} valid timesteps"
        )

    def _save_cache(self, cache_path: str) -> None:
        tmp_path = cache_path + ".tmp.npz"
        np.savez(
            tmp_path,
            all_actions=self.all_actions,
            timestep_to_chunk=self.timestep_to_chunk,
            valid_timesteps=self.valid_timesteps,
            episode_ends=self.episode_ends,
            episode_mask=self.episode_mask.astype(np.uint8),
            chunks_per_episode=np.asarray(self.chunks_per_episode, dtype=np.int64),
            episode_starts=np.asarray(self.episode_starts, dtype=np.int64),
        )
        os.replace(tmp_path, cache_path)
        print(f"Saved cached B-spline chunks to Disk: {cache_path}")

    def _preprocess_chunks(
        self,
        replay_buffer: ReplayBuffer,
        episode_ends: np.ndarray,
        episode_mask: np.ndarray,
        max_first_k: int,
    ) -> None:
        print(f"Preprocessing {np.sum(episode_mask)} episodes into B-spline chunks...")
        print(
            f"  chunk_size={self.chunk_size}, degree={self.degree}, "
            f"max_error={self.max_error}, stride={self.stride}"
        )

        all_chunks = []
        chunks_per_episode = []
        episode_starts = []
        self.timestep_to_chunk = np.full(episode_ends[-1], -1, dtype=np.int64)

        for ep_idx in range(len(episode_ends)):
            if not episode_mask[ep_idx]:
                continue

            ep_start = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            ep_end = episode_ends[ep_idx]
            episode_actions = replay_buffer[self.action_key][ep_start:ep_end]
            ep_length = len(episode_actions)

            n_dims = self.n_action_channels - 1
            episode_actions_to_fit = episode_actions[:, :n_dims]

            compressor = ScipyBSplineCompression(degree=self.degree)
            compressor.compress(
                episode_actions_to_fit, max_error=self.max_error, verbose=False
            )
            episode_starts.append(ep_start)

            chunks = chunk_bspline_trajectory(
                compressor,
                chunk_size=self.chunk_size,
                stride=self.stride,
                episode_length=ep_length,
                verbose=False,
            )
            chunks_per_episode.append(len(chunks))

            local_idx_in_episode = 0
            for chunk in chunks:
                chunk_data = np.zeros(
                    (self.n_action_steps, self.n_action_channels), dtype=np.float32
                )
                t_timesteps = chunk["t"]
                chunk_data[:, 0] = t_timesteps.copy()
                chunk_data[:, 1:] = chunk["c"]

                while local_idx_in_episode <= t_timesteps[self.degree]:
                    local_chunk_data = chunk_data.copy()
                    local_chunk_data[:, 0] -= local_idx_in_episode
                    if self.relative_knots:
                        local_chunk_data = encode_relative_knots(
                            local_chunk_data, degree=self.degree
                        )
                    all_chunks.append(local_chunk_data)
                    self.timestep_to_chunk[local_idx_in_episode + ep_start] = (
                        len(all_chunks) - 1
                    )
                    local_idx_in_episode += 1

            while local_idx_in_episode < ep_length - max_first_k + 1:
                local_chunk_data = chunk_data.copy()
                local_chunk_data[:, 0] -= local_idx_in_episode
                if self.relative_knots:
                    local_chunk_data = encode_relative_knots(
                        local_chunk_data, degree=self.degree
                    )
                all_chunks.append(local_chunk_data)
                self.timestep_to_chunk[local_idx_in_episode + ep_start] = (
                    len(all_chunks) - 1
                )
                local_idx_in_episode += 1

        if len(all_chunks) > 0:
            self.all_actions = np.asarray(all_chunks, dtype=np.float32)
        else:
            self.all_actions = np.zeros(
                (0, self.n_action_steps, self.n_action_channels), dtype=np.float32
            )

        self.episode_ends = episode_ends
        self.episode_mask = episode_mask
        self.chunks_per_episode = chunks_per_episode
        self.episode_starts = episode_starts
        self.valid_timesteps = np.flatnonzero(self.timestep_to_chunk >= 0)

        print("Preprocessing complete:")
        print(f"  Total chunks: {len(self.all_actions)}")
        print(f"  Action shape: {self.all_actions.shape}")
        print(
            f"  Valid mappings: {len(self.valid_timesteps)} / "
            f"{len(self.timestep_to_chunk)}"
        )

    def __len__(self) -> int:
        return len(self.valid_timesteps)

    def sample_sequence(self, idx: int) -> dict:
        if idx >= len(self.valid_timesteps):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        timestep = self.valid_timesteps[idx]
        chunk_idx = self.timestep_to_chunk[timestep]

        result = {}
        for key in self.keys:
            if key == self.action_key:
                continue

            input_arr = self.replay_buffer[key]
            if key not in self.key_first_k:
                sample = input_arr[timestep : timestep + 1]
            else:
                k_data = self.key_first_k[key]
                end_idx = min(timestep + k_data, len(input_arr))
                n_data = end_idx - timestep
                fill_value = np.nan if np.issubdtype(input_arr.dtype, np.floating) else 0
                sample = np.full(
                    (k_data,) + input_arr.shape[1:],
                    fill_value=fill_value,
                    dtype=input_arr.dtype,
                )
                sample[:n_data] = input_arr[timestep:end_idx]
                if np.issubdtype(sample.dtype, np.floating) and np.isnan(sample).any():
                    raise AssertionError("Sample contains nan")

            result[key] = sample

        result[self.action_key] = self.all_actions[chunk_idx].copy()
        return result

    def get_action_stats(self) -> dict:
        if len(self.all_actions) == 0:
            shape = (1, self.n_action_steps, self.n_action_channels)
            return {
                "min": np.zeros(shape, dtype=np.float32),
                "max": np.ones(shape, dtype=np.float32),
                "mean": np.zeros(shape, dtype=np.float32),
                "std": np.ones(shape, dtype=np.float32),
            }

        return {
            "min": np.min(self.all_actions, axis=0, keepdims=True),
            "max": np.max(self.all_actions, axis=0, keepdims=True),
            "mean": np.mean(self.all_actions, axis=0, keepdims=True),
            "std": np.std(self.all_actions, axis=0, keepdims=True),
        }


def decode_bspline_action(
    action_params,
    degree: int = 3,
    num_actions: int = 8,
    relative_knots: bool = False,
) -> np.ndarray:
    """Decode one B-spline parameter matrix into regular action vectors."""
    if torch.is_tensor(action_params):
        action_params = action_params.detach().cpu().numpy()
    action_params = np.asarray(action_params, dtype=np.float64)
    if relative_knots:
        action_params = decode_relative_knots(action_params, degree=degree)

    knots = action_params[:, 0].copy()
    control_points = action_params[: -(degree + 1), 1:].copy()
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    if t_max <= t_min:
        raise ValueError(f"Invalid B-spline range: [{t_min}, {t_max}]")

    if num_actions <= 1:
        t_eval = np.asarray([t_min], dtype=np.float64)
    else:
        t_eval = np.linspace(t_min, t_max, int(num_actions), dtype=np.float64)
    return BSpline(knots, control_points, degree, extrapolate=False)(t_eval).astype(
        np.float32
    )
