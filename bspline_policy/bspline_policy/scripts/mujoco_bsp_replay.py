import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path


TIDYBOT2_DIR = Path(__file__).resolve().parents[3] / "simple_mobile" / "tidybot2"
if str(TIDYBOT2_DIR) not in sys.path:
    sys.path.insert(0, str(TIDYBOT2_DIR))

import cv2 as cv
import numpy as np
from scipy.interpolate import generate_knots, make_lsq_spline

from constants import POLICY_CONTROL_FREQ
from episode_storage import EpisodeReader
from mujoco_env import MujocoEnv


@dataclass
class ActionLeaf:
    path: tuple
    shape: tuple
    dtype: np.dtype
    size: int
    is_scalar: bool


def iter_numeric_leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_numeric_leaves(child, path + (key,))
        return

    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Non-numeric action leaf at {'.'.join(path)}: {type(value)}")

    yield ActionLeaf(
        path=path,
        shape=array.shape,
        dtype=array.dtype,
        size=int(array.size),
        is_scalar=np.isscalar(value) or array.shape == (),
    )


def get_by_path(value, path):
    for key in path:
        value = value[key]
    return value


def set_by_path(value, path, leaf):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = leaf


def build_action_schema(action):
    leaves = list(iter_numeric_leaves(action))
    if not leaves:
        raise ValueError("Action has no numeric leaves")
    return copy.deepcopy(action), leaves


def flatten_action(action, leaves):
    parts = []
    for leaf in leaves:
        array = np.asarray(get_by_path(action, leaf.path), dtype=np.float64)
        if array.shape != leaf.shape:
            raise ValueError(
                f"Action leaf {'.'.join(leaf.path)} shape changed: "
                f"{array.shape} != {leaf.shape}"
            )
        parts.append(array.reshape(-1))
    return np.concatenate(parts, axis=0)


def unflatten_action(vector, template, leaves):
    action = copy.deepcopy(template)
    offset = 0
    for leaf in leaves:
        chunk = np.asarray(vector[offset : offset + leaf.size])
        offset += leaf.size
        if leaf.is_scalar:
            value = float(chunk[0])
        else:
            value = chunk.reshape(leaf.shape)
            if np.issubdtype(leaf.dtype, np.integer):
                value = np.rint(value).astype(leaf.dtype)
            else:
                value = value.astype(leaf.dtype, copy=False)
        set_by_path(action, leaf.path, value)
    if offset != len(vector):
        raise ValueError(f"Unused action vector values: {len(vector) - offset}")
    return action


def actions_to_matrix(actions):
    template, leaves = build_action_schema(actions[0])
    matrix = np.stack([flatten_action(action, leaves) for action in actions], axis=0)
    return matrix, template, leaves


class ScipyBSplineCompression:
    def __init__(self, degree=3):
        self.degree = degree
        self.spline = None
        self.knots = None

    def compress(self, data, max_error=0.01, verbose=False, s=1e-12):
        t = np.arange(len(data))
        last_knots = None
        for idx, knots in enumerate(generate_knots(t, data, s=s)):
            spl = make_lsq_spline(t, data, knots)
            pred_data = spl(t)
            error = np.abs(pred_data - data).max()
            last_knots = knots
            if error < max_error:
                self.knots = knots
                self.spline = spl
                break

        if self.knots is None:
            print(
                f"Failing to compress trajectory with max error {max_error}, "
                f"use min error we can find. Error is {error}. "
                "You can try to increase the s value."
            )
            self.knots = last_knots
            self.spline = make_lsq_spline(t, data, self.knots)

        if verbose:
            print(f"compression ratio: {len(self.knots) / len(t)}")

        return self.knots


def fit_bspline(actions, degree=3, max_error=0.01, smoothing=1e-12, verbose=False):
    if len(actions) < 2:
        raise ValueError("Need at least two actions to fit a B-spline")

    action_matrix, template, leaves = actions_to_matrix(actions)
    n_steps = len(action_matrix)
    degree = min(int(degree), n_steps - 1)
    compressor = ScipyBSplineCompression(degree=degree)
    compressor.compress(
        action_matrix,
        max_error=max_error,
        verbose=verbose,
        s=smoothing,
    )
    spline = compressor.spline

    if verbose:
        t = np.arange(n_steps)
        recon = spline(t)
        error = float(np.max(np.abs(recon - action_matrix)))
        print(
            "Fitted B-spline: "
            f"steps={n_steps}, dims={action_matrix.shape[1]}, degree={degree}, "
            f"knots={len(spline.t)}, control_points={len(spline.c)}, "
            f"max_error={error:.6g}"
        )

    return spline, template, leaves, n_steps


def sample_bspline_actions(
    spline,
    template,
    leaves,
    source_steps,
    source_freq,
    sample_freq,
    speed_up_times=1.0,
):
    source_freq = float(source_freq)
    sample_freq = float(sample_freq)
    speed_up_times = float(speed_up_times)
    if source_freq <= 0 or sample_freq <= 0 or speed_up_times <= 0:
        raise ValueError("source_freq, sample_freq, and speed_up_times must be positive")

    sample_period = 1.0 / sample_freq
    source_duration = (source_steps - 1) / source_freq
    replay_duration = source_duration / speed_up_times
    n_samples = int(np.floor(replay_duration * sample_freq + 1e-9)) + 1

    for sample_idx in range(n_samples):
        elapsed = sample_idx * sample_period
        source_t = min(elapsed * speed_up_times * source_freq, source_steps - 1)
        vector = np.asarray(spline(source_t), dtype=np.float64).reshape(-1)
        yield sample_idx, unflatten_action(vector, template, leaves)


def show_observation_images(obs):
    window_idx = 0
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            cv.imshow(key, cv.cvtColor(value, cv.COLOR_RGB2BGR))
            cv.moveWindow(key, 640 * window_idx, 0)
            window_idx += 1
    cv.waitKey(1)


def replay_episode(
    env,
    episode_dir,
    show_images=False,
    execute_obs=False,
    degree=3,
    max_error=0.01,
    smoothing=1e-12,
    sample_freq=POLICY_CONTROL_FREQ,
    source_freq=POLICY_CONTROL_FREQ,
    speed_up_times=1.0,
    verbose=False,
):
    # Reset env
    env.reset()

    # Load episode data
    reader = EpisodeReader(episode_dir)
    print(f"Loaded episode from {episode_dir}")

    spline, template, leaves, source_steps = fit_bspline(
        reader.actions,
        degree=degree,
        max_error=max_error,
        smoothing=smoothing,
        verbose=verbose,
    )

    sample_period = 1.0 / float(sample_freq)
    sampled_actions = sample_bspline_actions(
        spline,
        template,
        leaves,
        source_steps=source_steps,
        source_freq=source_freq,
        sample_freq=sample_freq,
        speed_up_times=speed_up_times,
    )

    start_time = time.time()
    for step_idx, action in sampled_actions:
        # Enforce desired sample freq. At 1x and default sample/source freq this
        # matches replay_episodes.py's POLICY_CONTROL_PERIOD time axis.
        step_end_time = start_time + step_idx * sample_period
        while time.time() < step_end_time:
            time.sleep(0.0001)

        nearest_obs_idx = min(
            int(round(step_idx * float(source_freq) / float(sample_freq))), source_steps - 1
        )
        obs = reader.observations[nearest_obs_idx]

        # Show image observations
        if show_images:
            show_observation_images(obs)

        # Execute in action in env
        if execute_obs:
            env.step(obs)
        else:
            env.step(action)

        print(f"Actual frequency: {step_idx / (time.time() - start_time)}")


def main(args):
    # Create env
    print("Creating environment...")
    if args.sim:
        env = MujocoEnv(render_images=False)
    else:
        from real_yam_bimanual_hex_env import RealYamBimanualHexEnv

        env = RealYamBimanualHexEnv(use_cameras=False)  # Cameras add startup overhead; skip for replay
        # Scale the per-tick J-PARSE IK dt on each YAM server so the arm can slew
        # fast enough to track sped-up TCP targets. Without this, the arm lags
        # behind the spline at speed_up_times > 1 and the executed path looks
        # like a low-pass-filtered, corner-cut version of the demo.
        env.arm_left.set_ik_dt_scale(args.speed_up_times)
        env.arm_right.set_ik_dt_scale(args.speed_up_times)
    print("Environment created")
    try:
        episode_dirs = sorted([child for child in Path(args.input_dir).iterdir() if child.is_dir()])
        for episode_dir in episode_dirs:
            replay_episode(
                env,
                episode_dir,
                show_images=args.show_images,
                execute_obs=args.execute_obs,
                degree=args.degree,
                max_error=args.max_error,
                smoothing=args.smoothing,
                sample_freq=args.sample_freq,
                source_freq=args.source_freq,
                speed_up_times=args.speed_up_times,
                verbose=args.verbose,
            )
            input("Press <Enter> to continue...")
    finally:
        if not args.sim:
            try:
                env.arm_left.set_ik_dt_scale(1.0)
                env.arm_right.set_ik_dt_scale(1.0)
            except Exception:
                pass
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/demos")
    parser.add_argument("--sim", action="store_true")

    parser.add_argument("--show-images", action="store_true")
    parser.add_argument("--execute-obs", action="store_true")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.01,
        help="Adaptive B-spline fitting threshold, matching ScipyBSplineCompression.",
    )
    parser.add_argument("--smoothing", type=float, default=1e-12)
    parser.add_argument("--sample-freq", type=float, default=100)
    parser.add_argument("--source-freq", type=float, default=POLICY_CONTROL_FREQ)
    parser.add_argument("--speed-up-times", type=float, default=4.0)
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
