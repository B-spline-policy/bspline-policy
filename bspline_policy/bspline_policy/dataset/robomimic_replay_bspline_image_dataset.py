from typing import Dict
import copy
import os
import shutil

import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits

from bspline_policy.common.bspline_action import (
    BSplineChunkSampler,
    make_bspline_sampler_cache_path,
)
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.preprocessed_sample_cache import (
    build_preprocessed_sample_cache,
    get_preprocessed_cached_item,
    make_empty_preprocessed_sample_cache,
    normalize_rgb_cache_dtype,
)
from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    _convert_robomimic_to_replay,
    _replay_cache_path,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def _cache_base_path(dataset_path: str, cache_suffix: str = None) -> str:
    if not cache_suffix:
        return dataset_path
    if not cache_suffix.startswith("."):
        cache_suffix = "." + cache_suffix
    return dataset_path + cache_suffix


class RobomimicReplayBSplineImageDataset(BaseImageDataset):
    """Robomimic image dataset that represents actions as B-spline chunks.

    This keeps the target repo's existing real bimanual conversion logic
    unchanged, then replaces the regular sequence sampler with
    BSplineChunkSampler.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        chunk_size=10,
        bspline_degree=3,
        max_error=0.002,
        stride=1,
        abs_action=False,
        rotation_rep="rotation_6d",
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        relative_knots=False,
        cache_suffix=None,
        cache_decoded_replay=False,
        cache_preprocessed_samples=False,
        cache_preprocessed_device="cpu",
        cache_preprocessed_share_memory=False,
        cache_preprocessed_rgb_dtype="float32",
    ):
        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )
        cache_preprocessed_rgb_dtype = normalize_rgb_cache_dtype(
            cache_preprocessed_rgb_dtype
        )

        replay_buffer = None
        if use_cache:
            cache_zarr_path = _replay_cache_path(dataset_path, cache_suffix)
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("Cache does not exist. Creating!")
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                        if cache_decoded_replay:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=replay_buffer.root.store, store=None
                            )
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        if cache_decoded_replay:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=None
                            )
                        else:
                            replay_buffer = ReplayBuffer.copy_from_store(
                                src_store=zip_store, store=zarr.MemoryStore()
                            )
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
            )
            if cache_decoded_replay:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=replay_buffer.root.store, store=None
                )

        rgb_keys = []
        lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        n_action_steps = int(chunk_size) + 2 * int(bspline_degree)
        n_control_dims = shape_meta["action"]["shape"][0]
        n_action_channels = 1 + n_control_dims

        sampler_cache_path = None
        if use_cache:
            sampler_cache_path = make_bspline_sampler_cache_path(
                base_path=_cache_base_path(dataset_path, cache_suffix),
                episode_mask=train_mask,
                key_first_k=key_first_k,
                chunk_size=chunk_size,
                degree=bspline_degree,
                max_error=max_error,
                stride=stride,
                n_action_steps=n_action_steps,
                n_action_channels=n_action_channels,
                relative_knots=relative_knots,
            )

        sampler = BSplineChunkSampler(
            replay_buffer=replay_buffer,
            chunk_size=chunk_size,
            degree=bspline_degree,
            max_error=max_error,
            stride=stride,
            episode_mask=train_mask,
            key_first_k=key_first_k,
            action_key="action",
            n_action_steps=n_action_steps,
            n_action_channels=n_action_channels,
            relative_knots=relative_knots,
            cache_path=sampler_cache_path,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_path = dataset_path
        self.cache_suffix = cache_suffix
        self.use_cache = use_cache
        self.chunk_size = chunk_size
        self.bspline_degree = bspline_degree
        self.max_error = max_error
        self.stride = stride
        self.relative_knots = bool(relative_knots)
        self.n_action_steps = n_action_steps
        self.n_action_channels = n_action_channels
        self.cache_decoded_replay = cache_decoded_replay
        self.cache_preprocessed_samples = cache_preprocessed_samples
        self.cache_preprocessed_device = cache_preprocessed_device
        self.cache_preprocessed_share_memory = cache_preprocessed_share_memory
        self.cache_preprocessed_rgb_dtype = cache_preprocessed_rgb_dtype
        self._preprocessed_cache = None
        self._length = len(sampler)

        if len(sampler) > 0:
            action_shape = sampler.sample_sequence(0)["action"].shape
            expected_shape = (n_action_steps, n_action_channels)
            print(f"Action shape from B-spline sampler: {action_shape}")
            print(f"Expected shape: {expected_shape}")
            if action_shape != expected_shape:
                raise AssertionError(
                    f"Action shape mismatch: expected {expected_shape}, got {action_shape}"
                )

        if cache_preprocessed_samples:
            self._build_preprocessed_cache(
                device=cache_preprocessed_device,
                share_memory=cache_preprocessed_share_memory,
            )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        key_first_k = {}
        if self.n_obs_steps is not None:
            for key in self.rgb_keys + self.lowdim_keys:
                key_first_k[key] = self.n_obs_steps

        sampler_cache_path = None
        if self.use_cache:
            sampler_cache_path = make_bspline_sampler_cache_path(
                base_path=_cache_base_path(self.dataset_path, self.cache_suffix),
                episode_mask=~self.train_mask,
                key_first_k=key_first_k,
                chunk_size=self.chunk_size,
                degree=self.bspline_degree,
                max_error=self.max_error,
                stride=self.stride,
                n_action_steps=self.n_action_steps,
                n_action_channels=self.n_action_channels,
                relative_knots=self.relative_knots,
            )

        val_set.sampler = BSplineChunkSampler(
            replay_buffer=self.replay_buffer,
            chunk_size=self.chunk_size,
            degree=self.bspline_degree,
            max_error=self.max_error,
            stride=self.stride,
            episode_mask=~self.train_mask,
            key_first_k=key_first_k,
            action_key="action",
            n_action_steps=self.n_action_steps,
            n_action_channels=self.n_action_channels,
            relative_knots=self.relative_knots,
            cache_path=sampler_cache_path,
        )
        val_set.train_mask = ~self.train_mask
        val_set._length = len(val_set.sampler)
        if self._preprocessed_cache is not None:
            val_set._preprocessed_cache = None
            val_set._build_preprocessed_cache(
                device=self.cache_preprocessed_device,
                share_memory=self.cache_preprocessed_share_memory,
            )
        return val_set

    def __copy__(self):
        result = self.__class__.__new__(self.__class__)
        result.__dict__.update(self.__dict__)
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("_preprocessed_cache") is not None:
            state["replay_buffer"] = None
            state["sampler"] = None
        return state

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        action_stats = self.sampler.get_action_stats()
        channel_stats = {
            "min": np.min(action_stats["min"], axis=1, keepdims=True),
            "max": np.max(action_stats["max"], axis=1, keepdims=True),
            "mean": np.mean(action_stats["mean"], axis=1, keepdims=True),
            "std": np.mean(action_stats["std"], axis=1, keepdims=True),
        }
        n_action_steps = action_stats["min"].shape[1]
        n_channels = action_stats["min"].shape[2]
        stat = {}
        for key in ["min", "max", "mean", "std"]:
            stat[key] = np.broadcast_to(
                channel_stats[key], (1, n_action_steps, n_channels)
            ).reshape(-1)
        normalizer["action"] = get_range_normalizer_from_stat(stat)

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if "pos" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif "quat" in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif "qpos" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == "base_pose":
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"unsupported lowdim key: {key}")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.sampler.all_actions)

    def __len__(self):
        return self._length

    def _sample_uncached_item(
        self,
        idx: int,
        rgb_dtype: str = "float32",
    ) -> Dict[str, torch.Tensor]:
        if self.sampler is None:
            raise RuntimeError(
                "RobomimicReplayBSplineImageDataset was serialized without "
                "replay/sampler because cache_preprocessed_samples is enabled, "
                "but no preprocessed cache is available for this item."
            )
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            image = np.moveaxis(data[key][t_slice], -1, 1)
            if rgb_dtype == "uint8":
                obs_dict[key] = np.ascontiguousarray(image, dtype=np.uint8)
            else:
                obs_dict[key] = np.ascontiguousarray(image, dtype=np.float32)
                obs_dict[key] *= 1.0 / 255.0
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][t_slice].astype(np.float32)
            del data[key]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }

    def _build_preprocessed_cache(self, device: str = "cpu", share_memory: bool = False):
        print(
            f"Building B-spline preprocessed sample cache: samples={len(self)}, "
            f"device={device}, share_memory={share_memory}, "
            f"rgb_dtype={self.cache_preprocessed_rgb_dtype}"
        )
        obs_steps = self.n_obs_steps if self.n_obs_steps is not None else self.horizon
        empty_cache = make_empty_preprocessed_sample_cache(
            shape_meta=self.shape_meta,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.lowdim_keys,
            action_shape=(self.n_action_steps, self.n_action_channels),
            obs_steps=obs_steps,
            device=torch.device(device),
            share_memory=share_memory,
            rgb_dtype=self.cache_preprocessed_rgb_dtype,
        )
        self._preprocessed_cache = build_preprocessed_sample_cache(
            length=len(self),
            sample_fn=lambda idx: self._sample_uncached_item(
                idx,
                rgb_dtype=self.cache_preprocessed_rgb_dtype,
            ),
            device=device,
            share_memory=share_memory,
            empty_cache=empty_cache,
            desc="Precomputing B-spline samples",
        )
        print("B-spline preprocessed sample cache ready.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._preprocessed_cache is not None:
            return get_preprocessed_cached_item(
                self._preprocessed_cache,
                idx,
                rgb_keys=self.rgb_keys,
            )
        return self._sample_uncached_item(idx)
