"""Mock single-YAM rollout that mirrors the real rollout_local_policy.py loop.

The control loop below is intentionally identical to
simple_mobile/tidybot2/rollout_local_policy.py::rollout_episode (same pacing,
same obs -> policy.step -> env.step order, action=None steps are skipped, no
clock pausing), so the measured callback timings transfer to the real robot.
The only difference from a real rollout is that env I/O is synthetic; use
--obs-latency-ms to emulate the camera/RPC cost of RealEnv.get_obs().
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import count
from pathlib import Path

import numpy as np

YAM_TELEOP_DIR = Path(__file__).resolve().parent
SIMPLE_MOBILE_DIR = YAM_TELEOP_DIR.parent
REPO_ROOT = SIMPLE_MOBILE_DIR.parent
BSPLINE_POLICY_DIR = REPO_ROOT / "bspline_policy"
TIDYBOT2_DIR = SIMPLE_MOBILE_DIR / "tidybot2"
for path in (BSPLINE_POLICY_DIR, TIDYBOT2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constants import POLICY_CONTROL_FREQ
from constants import RAW_IMAGE_HEIGHT
from constants import RAW_IMAGE_WIDTH


class MockYamEnv:
    """Env-shaped object with the same interface/obs keys as real_env.RealEnv."""

    def __init__(self, image_width=RAW_IMAGE_WIDTH, image_height=RAW_IMAGE_HEIGHT,
                 obs_latency=0.0):
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.obs_latency = float(obs_latency)
        self.reset()

    def reset(self):
        self.obs_idx = 0
        self.arm_pos = np.array([0.35, 0.0, 0.25], dtype=np.float64)
        self.arm_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.gripper_pos = np.array([0.0], dtype=np.float64)

    def get_obs(self):
        if self.obs_latency > 0.0:
            time.sleep(self.obs_latency)
        self.obs_idx += 1
        return {
            "arm_pos": self.arm_pos.copy(),
            "arm_quat": self.arm_quat.copy(),
            "gripper_pos": self.gripper_pos.copy(),
            "wrist_image": self._make_wrist_image(),
        }

    def step(self, action):
        # Mirrors RealEnv.step: non-blocking apply, no obs returned.
        self.arm_pos = np.asarray(action["arm_pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(action["arm_quat"], dtype=np.float64).reshape(4)
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm
        if quat[3] < 0:
            quat = -quat
        self.arm_quat = quat
        self.gripper_pos = np.asarray(action["gripper_pos"], dtype=np.float64).reshape(1)

    def close(self):
        pass

    def _make_wrist_image(self):
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        x = np.linspace(0, 255, self.image_width, dtype=np.uint8)
        y = np.linspace(0, 255, self.image_height, dtype=np.uint8)
        image[:, :, 0] = x[None, :]
        image[:, :, 1] = y[:, None]
        image[:, :, 2] = (self.obs_idx * 17) % 255
        return image


class LoopStats:
    """Per-tick callback timing: obs read, policy step, env step, total."""

    PHASES = ("get_obs", "policy_step", "env_step", "callback")

    def __init__(self, control_period):
        self.control_period = float(control_period)
        self.samples = {phase: [] for phase in self.PHASES}
        self.overruns = 0

    def add(self, get_obs, policy_step, env_step):
        callback = get_obs + policy_step + env_step
        for phase, value in zip(self.PHASES, (get_obs, policy_step, env_step, callback)):
            self.samples[phase].append(value)
        if callback > self.control_period:
            self.overruns += 1

    def print_summary(self):
        n = len(self.samples["callback"])
        if n == 0:
            print("Callback loop summary: no steps recorded")
            return
        print(
            f"Callback loop summary: n={n} target_period={self.control_period*1e3:.1f}ms "
            f"overruns={self.overruns} ({100.0*self.overruns/n:.1f}%)"
        )
        for phase in self.PHASES:
            values = np.asarray(self.samples[phase], dtype=np.float64) * 1e3
            print(
                f"  {phase:<11} mean={values.mean():7.3f}ms p50={np.percentile(values, 50):7.3f}ms "
                f"p95={np.percentile(values, 95):7.3f}ms max={values.max():7.3f}ms"
            )


def make_policy(args):
    from bspline_policy.scripts.policy_local_bspline import PolicyLocalBSpline

    obs_stride = max(1, int(round(args.control_freq / args.data_freq)))
    return PolicyLocalBSpline(
        ckpt_path=args.ckpt_path,
        diffusion_policy_dir=args.diffusion_policy_dir,
        device=args.device,
        use_ema=args.use_ema,
        rotation_output=args.rotation_output,
        num_inference_steps=args.num_inference_steps,
        use_cuda_graph=args.cuda_graph,
        n_obs_steps=args.n_obs_steps,
        obs_stride=obs_stride,
        degree=args.degree,
        speed_up_times=args.speed_up_times,
        predict_before_end=args.predict_before_end,
        origin_time_scale=args.origin_time_scale,
        use_action_derivatives=args.use_action_derivatives,
        disable_time_align=args.disable_time_align,
        time_align_error_threshold=args.time_align_error_threshold,
        time_align_larger_t=args.time_align_larger_t,
        restart_on_time_align_error=args.restart_on_time_align_error,
        consider_gripper_during_align=args.consider_gripper_during_align,
        gripper_slowdown_enabled=args.gripper_slowdown_enabled,
        gripper_slowdown_threshold=args.gripper_slowdown_threshold,
        gripper_slowdown_steps=args.gripper_slowdown_steps,
    )


def format_action(action):
    if action is None:
        return "None"
    return " ".join(
        f"{key}={np.array2string(np.asarray(value), precision=4, suppress_small=True)}"
        for key, value in action.items()
    )


def main(args):
    env = MockYamEnv(
        image_width=args.raw_image_width,
        image_height=args.raw_image_height,
        obs_latency=args.obs_latency_ms / 1e3,
    )
    policy = make_policy(args)

    control_period = 1.0 / float(args.control_freq)
    stats = LoopStats(control_period)
    env.reset()
    policy.reset()

    first_action_step = None
    try:
        # Same pacing/callback structure as rollout_local_policy.rollout_episode.
        start_time = time.perf_counter()
        for step_idx in count():
            if args.steps > 0 and step_idx >= args.steps:
                break
            step_end_time = start_time + step_idx * control_period
            while time.perf_counter() < step_end_time:
                time.sleep(0.0001)

            tick = time.perf_counter()
            obs = env.get_obs()
            t_obs = time.perf_counter()
            action = policy.step(obs)
            t_policy = time.perf_counter()
            if action is not None:
                env.step(action)
            t_env = time.perf_counter()
            stats.add(t_obs - tick, t_policy - t_obs, t_env - t_policy)

            if first_action_step is None and action is not None:
                first_action_step = step_idx
                print(
                    f"First action at step {step_idx} "
                    f"({step_idx * control_period:.3f}s after loop start)"
                )
            if args.verbose:
                print(f"step={step_idx:03d} action={format_action(action)}")
            elif args.log_every > 0 and step_idx % args.log_every == 0:
                print(f"step={step_idx:04d} action={'ok' if action is not None else 'None'}")
    finally:
        # Let any in-flight spline request finish before interpreter teardown,
        # otherwise the daemon thread dies mid-CUDA-call and the process aborts.
        policy.wait_for_pending_inference()
        stats.print_summary()
        policy.print_inference_summary()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mock single-YAM env mirroring the real rollout loop, with callback timing stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--diffusion-policy-dir", default=str(REPO_ROOT / "diffusion_policy"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=None)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--rotation-output", choices=("axis_angle", "euler_xyz"), default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override DDIM inference steps (default: use checkpoint value)")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Capture the DDIM denoising loop as a CUDA graph at load time")
    parser.add_argument("--control-freq", type=float, default=200)
    parser.add_argument("--data-freq", type=float, default=POLICY_CONTROL_FREQ)
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--speed-up-times", type=float, default=1.0)
    parser.add_argument("--predict-before-end", type=float, default=0.06)
    parser.add_argument("--origin-time-scale", type=float, default=10.0)
    parser.add_argument("--use-action-derivatives", action="store_true")
    parser.add_argument("--disable-time-align", action="store_true")
    parser.add_argument("--time-align-error-threshold", type=float, default=0.1)
    parser.add_argument("--time-align-larger-t", type=float, default=0.2)
    parser.add_argument("--restart-on-time-align-error", action="store_true")
    parser.add_argument("--consider-gripper-during-align", action="store_true")
    parser.add_argument("--gripper-slowdown-enabled", action="store_true")
    parser.add_argument("--gripper-slowdown-threshold", type=float, default=0.08)
    parser.add_argument("--gripper-slowdown-steps", type=int, default=7)
    parser.add_argument("--raw-image-width", type=int, default=RAW_IMAGE_WIDTH)
    parser.add_argument("--raw-image-height", type=int, default=RAW_IMAGE_HEIGHT)
    parser.add_argument("--obs-latency-ms", type=float, default=0.0,
                        help="Injected env.get_obs() latency to emulate camera/RPC cost")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--verbose", action="store_true",
                        help="Print the decoded action every control step")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Progress print interval when not verbose (0 disables)")
    main(parser.parse_args())
