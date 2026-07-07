import argparse
import sys
import threading
import time
from itertools import count
from pathlib import Path


BSPLINE_POLICY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BSPLINE_POLICY_DIR.parent
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"
TIDYBOT2_DIR = REPO_ROOT / "simple_mobile" / "tidybot2"
for path in (BSPLINE_POLICY_DIR, DIFFUSION_POLICY_DIR, TIDYBOT2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def make_camera_cfg(args):
    if args.no_cameras:
        return None
    return {
        "frame_width": args.frame_width,
        "frame_height": args.frame_height,
        "fps": args.camera_fps,
        "auto_exposure": args.auto_exposure,
        "exposure": args.exposure,
        "gain": args.gain,
        "auto_white_balance": args.auto_white_balance,
        "white_balance": args.white_balance,
    }


def make_env(args):
    from x5_env import RealEnvLocalX5Dual

    return RealEnvLocalX5Dual(
        model=args.model,
        left_interface=args.left_interface,
        right_interface=args.right_interface,
        enable_camera=not args.no_cameras,
        camera_cfg=make_camera_cfg(args),
        left_camera_serial=args.left_camera_serial,
        right_camera_serial=args.right_camera_serial,
        head_camera_serial=args.head_camera_serial,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        fps=args.camera_fps,
        wait_for_start=not args.no_wait_for_start,
        gripper_soft_contact_hold_torque=args.gripper_soft_contact_hold_torque,
        gripper_soft_contact_torque_threshold=args.gripper_soft_contact_torque_threshold,
        gripper_soft_contact_kd=args.gripper_soft_contact_kd,
        stiffness_kp_scale=args.stiffness_kp_scale,
    )


def make_policy(args):
    from bspline_policy.scripts.policy_local_bspline import PolicyLocalBSpline

    return PolicyLocalBSpline(
        ckpt_path=args.ckpt_path,
        diffusion_policy_dir=args.diffusion_policy_dir,
        device=args.device,
        use_ema=args.use_ema,
        n_obs_steps=args.n_obs_steps,
        obs_stride=max(1, int(round(args.control_freq / args.data_freq))),
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


def serializable_obs(obs):
    clean = {}
    for key, value in obs.items():
        if isinstance(value, dict) or value is None or not hasattr(value, "ndim"):
            continue
        clean[key] = value
    return clean


def start_end_listener(key_char="n"):
    import pynput

    event = threading.Event()

    def on_press(key):
        if getattr(key, "char", None) == key_char:
            event.set()

    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()
    return event, listener


def rollout_episode(env, policy, args, writer=None):
    control_period = 1.0 / float(args.control_freq)
    record_stride = max(1, int(round(args.control_freq / args.record_freq)))

    env.reset()
    policy.reset()
    print("Starting X5 B-spline rollout. Press 'n' to end the episode.")
    end_event, listener = start_end_listener()

    try:
        start_time = time.perf_counter()
        last_action = None
        for step_idx in count():
            if args.max_steps > 0 and step_idx >= args.max_steps:
                print("Max steps reached")
                break
            if end_event.is_set():
                print("Episode end requested")
                break

            step_end_time = start_time + step_idx * control_period
            while time.perf_counter() < step_end_time:
                time.sleep(0.0001)

            obs = env.get_obs()
            action = policy.step(obs)
            if action is not None:
                env.step(action)
                last_action = action

            if (
                writer is not None
                and last_action is not None
                and step_idx % record_stride == 0
            ):
                writer.step(serializable_obs(obs), last_action)
    finally:
        listener.stop()

    if writer is not None and len(writer) > 0:
        writer.flush_async()
        writer.wait_for_flush()


def main(args):
    from episode_storage import EpisodeWriter

    if args.record_freq <= 0:
        args.record_freq = args.data_freq
    env = make_env(args)
    policy = make_policy(args)
    try:
        for episode_idx in range(args.num_episodes):
            print(f"Episode {episode_idx + 1}/{args.num_episodes}")
            writer = EpisodeWriter(args.output_dir) if args.save else None
            rollout_episode(env, policy, args, writer=writer)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local X5 dual-arm B-spline policy rollout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument(
        "--diffusion-policy-dir",
        default=str(DIFFUSION_POLICY_DIR),
        help="Repository root that contains the diffusion_policy package used by the checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-ema", action="store_true")

    parser.add_argument("--model", default="X5")
    parser.add_argument("--left-interface", default="can1")
    parser.add_argument("--right-interface", default="can3")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--no-wait-for-start", action="store_true")
    parser.add_argument("--left-camera-serial", default="230422273182")
    parser.add_argument("--right-camera-serial", default="218622274652")
    parser.add_argument("--head-camera-serial", default="230322272285")
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--exposure", type=float, default=51000)
    parser.add_argument("--gain", type=float, default=32)
    parser.add_argument("--auto-white-balance", action="store_true")
    parser.add_argument("--white-balance", type=float, default=3000)
    parser.add_argument("--gripper-soft-contact-hold-torque", type=float, default=0.3)
    parser.add_argument("--gripper-soft-contact-torque-threshold", type=float, default=None)
    parser.add_argument("--gripper-soft-contact-kd", type=float, default=None)
    parser.add_argument(
        "--stiffness-kp-scale",
        type=float,
        default=2.0,
        help="Scale factor on the Cartesian position gain (kp); 1.0 = arx5 default, >1.0 = stiffer",
    )

    parser.add_argument("--control-freq", type=float, default=200)
    parser.add_argument("--data-freq", type=float, default=10)
    parser.add_argument("--record-freq", type=float, default=0)
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

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output-dir", default="data/x5_bspline_rollouts")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    main(parser.parse_args())
