import argparse
import sys
import threading
import time
from itertools import count
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BSPLINE_POLICY_DIR = REPO_ROOT / "bspline_policy"
YAM_TELEOP_DIR = REPO_ROOT / "simple_mobile" / "yam_teleop"
if str(BSPLINE_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(BSPLINE_POLICY_DIR))


def make_camera_cfg(args):
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
    if args.env == "x5":
        from x5_env import RealEnvLocalX5Dual

        return RealEnvLocalX5Dual(
            model=args.model,
            left_interface=args.left_interface,
            right_interface=args.right_interface,
            enable_camera=True,
            camera_cfg=make_camera_cfg(args),
            left_camera_serial=args.left_camera_serial,
            right_camera_serial=args.right_camera_serial,
            head_camera_serial=args.head_camera_serial,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            fps=args.camera_fps,
            wait_for_start=True,
            gripper_soft_contact_hold_torque=args.gripper_soft_contact_hold_torque,
            gripper_soft_contact_torque_threshold=args.gripper_soft_contact_torque_threshold,
            gripper_soft_contact_kd=args.gripper_soft_contact_kd,
            stiffness_kp_scale=args.stiffness_kp_scale,
        )

    if args.env == "tidybot2":
        from real_yam_bimanual_hex_env import RealYamBimanualHexEnv

        return RealYamBimanualHexEnv(use_cameras=True)

    if args.env == "yam":
        yam_teleop_dir = str(YAM_TELEOP_DIR)
        if yam_teleop_dir in sys.path:
            sys.path.remove(yam_teleop_dir)
        sys.path.insert(0, yam_teleop_dir)
        from real_env import RealEnv

        return RealEnv(
            use_cameras=not args.no_cameras,
            stiffness_kp_scale=args.stiffness_kp_scale,
        )

    raise ValueError(f"Unknown env: {args.env}")


def make_episode_writer(output_dir):
    from episode_storage import EpisodeWriter

    return EpisodeWriter(output_dir)


def set_env_speed_scale(env, scale):
    arm = getattr(env, "arm", None)
    if arm is not None and hasattr(arm, "set_ik_dt_scale"):
        arm.set_ik_dt_scale(float(scale))


def make_policy(args):
    obs_stride = max(1, int(round(args.control_freq / args.data_freq)))
    if args.policy == "bspline":
        from bspline_policy.scripts.policy_local_bspline import PolicyLocalBSpline

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

    if args.policy == "dp":
        from policy_local_dp import PolicyLocalDP

        return PolicyLocalDP(
            ckpt_path=args.ckpt_path,
            diffusion_policy_dir=args.diffusion_policy_dir,
            device=args.device,
            use_ema=args.use_ema,
            rotation_output=args.rotation_output,
            n_obs_steps=args.n_obs_steps,
            obs_stride=obs_stride,
            action_repeat=obs_stride,
        )

    raise ValueError(f"Unknown policy: {args.policy}")


def serializable_obs(obs):
    clean = {}
    for key, value in obs.items():
        if isinstance(value, dict) or value is None or not hasattr(value, "ndim"):
            continue
        clean[key] = value
    return clean


def start_end_listener(key_char="n"):
    try:
        import pynput
    except ImportError:
        event = threading.Event()

        class NoopListener:
            def stop(self):
                pass

        print("pynput not installed; use --max-steps or Ctrl-C to end the episode.")
        return event, NoopListener()

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
    set_env_speed_scale(env, args.speed_up_times if args.policy == "bspline" else 1.0)
    print("Starting local policy rollout. Press 'n' to end the episode.")
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
        set_env_speed_scale(env, 1.0)
        listener.stop()

    if writer is not None and len(writer) > 0:
        writer.flush_async()
        writer.wait_for_flush()


def main(args):
    if args.record_freq <= 0:
        args.record_freq = args.data_freq
    env = make_env(args)
    policy = make_policy(args)
    try:
        for episode_idx in range(args.num_episodes):
            print(f"Episode {episode_idx + 1}/{args.num_episodes}")
            writer = make_episode_writer(args.output_dir) if args.save else None
            rollout_episode(env, policy, args, writer=writer)
    finally:
        # Avoid killing the predictor thread mid-CUDA-call at interpreter
        # teardown (aborts the process with SIGABRT).
        if hasattr(policy, "wait_for_pending_inference"):
            policy.wait_for_pending_inference()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local rollout for X5, tidybot2, or single-arm YAM DP/B-spline checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", choices=("x5", "tidybot2", "yam"), required=True)
    parser.add_argument("--policy", choices=("bspline", "dp"), required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument(
        "--diffusion-policy-dir",
        default="~/simple_mobile_bsp/diffusion_policy",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=None)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--rotation-output", choices=("axis_angle", "euler_xyz"), default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override DDIM inference steps (default: use checkpoint value)")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Capture the B-spline DDIM denoising loop as a CUDA graph at load time")

    parser.add_argument("--model", default="X5")
    parser.add_argument("--left-interface", default="can1")
    parser.add_argument("--right-interface", default="can3")
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
        default=1.0,
        help="Scale on the arm-joint position gain (kp); 1.0 = default, >1.0 = stiffer. "
        "For YAM this scales kp[:6] = [80,80,80,40,10,10]; for X5 it scales the Cartesian kp.",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Disable the single-arm YAM wrist camera and use black image observations.",
    )

    parser.add_argument("--control-freq", type=float, default=10)
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
    parser.add_argument("--output-dir", default="data/local_policy_rollouts")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    main(parser.parse_args())
