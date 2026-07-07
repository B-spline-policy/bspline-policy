import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def _default_diffusion_policy_dir():
    return Path(__file__).resolve().parents[2] / "diffusion_policy"


def _add_diffusion_policy_to_path(diffusion_policy_dir=None):
    root = Path(diffusion_policy_dir) if diffusion_policy_dir else _default_diffusion_policy_dir()
    root = root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def rotation_6d_to_matrix(rot6d):
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1 = rot6d[:3]
    a2 = rot6d[3:]
    b1 = a1 / max(np.linalg.norm(a1), 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)


def rot6d_to_pose_rotation(rot6d, rotation_output):
    rot = R.from_matrix(rotation_6d_to_matrix(rot6d))
    if rotation_output == "axis_angle":
        return rot.as_rotvec()
    if rotation_output == "euler_xyz":
        return rot.as_euler("xyz")
    raise ValueError(f"Unsupported B-spline rotation_output: {rotation_output}")


def rot6d_to_quat_xyzw(rot6d):
    quat = R.from_matrix(rotation_6d_to_matrix(rot6d)).as_quat()
    if quat[3] < 0.0:
        np.negative(quat, out=quat)
    return quat


def _hold_tidybot_arm(obs, prefix):
    return {
        "arm_pos": np.asarray(obs[f"arm_pos_{prefix}"], dtype=np.float64).copy(),
        "arm_quat": np.asarray(obs[f"arm_quat_{prefix}"], dtype=np.float64).copy(),
        "gripper_pos": np.asarray(obs[f"gripper_pos_{prefix}"], dtype=np.float64).copy(),
    }


def decode_action_vector(action_raw, action_meta, latest_obs=None):
    action_raw = np.asarray(action_raw, dtype=np.float64).reshape(-1)
    action_format = (action_meta or {}).get("action_format")

    if action_format == "single_yam_rot6d":
        if action_raw.size != 10:
            raise ValueError(f"Expected 10D single-YAM rot6d action, got {action_raw.size}")
        return {
            "arm_pos": action_raw[:3].copy(),
            "arm_quat": rot6d_to_quat_xyzw(action_raw[3:9]),
            "gripper_pos": np.array([action_raw[9]], dtype=np.float64),
        }

    if action_format in ("single_left_arm_rot6d", "single_right_arm_rot6d"):
        if action_raw.size != 10:
            raise ValueError(f"Expected 10D single-arm rot6d action, got {action_raw.size}")
        side = "left" if action_format == "single_left_arm_rot6d" else "right"
        arm_action = {
            "arm_pos": action_raw[:3].copy(),
            "arm_quat": rot6d_to_quat_xyzw(action_raw[3:9]),
            "gripper_pos": np.array([action_raw[9]], dtype=np.float64),
        }
        if latest_obs is None:
            raise ValueError("10D single-arm action needs latest_obs to hold the other arm")
        if side == "left":
            return {
                "base_velocity": np.zeros(3, dtype=np.float64),
                "arm_left": arm_action,
                "arm_right": _hold_tidybot_arm(latest_obs, "r"),
            }
        return {
            "base_velocity": np.zeros(3, dtype=np.float64),
            "arm_left": _hold_tidybot_arm(latest_obs, "l"),
            "arm_right": arm_action,
        }

    if action_format == "real_bimanual_base_rot6d":
        if action_raw.size != 23:
            raise ValueError(f"Expected 23D bimanual base rot6d action, got {action_raw.size}")
        return {
            "base_velocity": action_raw[:3].copy(),
            "arm_left": {
                "arm_pos": action_raw[3:6].copy(),
                "arm_quat": rot6d_to_quat_xyzw(action_raw[6:12]),
                "gripper_pos": np.array([action_raw[12]], dtype=np.float64),
            },
            "arm_right": {
                "arm_pos": action_raw[13:16].copy(),
                "arm_quat": rot6d_to_quat_xyzw(action_raw[16:22]),
                "gripper_pos": np.array([action_raw[22]], dtype=np.float64),
            },
        }

    if action_format in ("dual_arm_ee_rot6d", "dual_arm_ee_rot6d_next"):
        if action_raw.size != 20:
            raise ValueError(f"Expected 20D rot6d action, got {action_raw.size}")
        rotation_output = (action_meta or {}).get("rotation_output") or "euler_xyz"
        return {
            "left_arm": {
                "pose_6d": np.concatenate(
                    [
                        action_raw[:3],
                        rot6d_to_pose_rotation(action_raw[3:9], rotation_output),
                    ]
                ),
                "gripper_pos": float(action_raw[18]),
            },
            "right_arm": {
                "pose_6d": np.concatenate(
                    [
                        action_raw[9:12],
                        rot6d_to_pose_rotation(action_raw[12:18], rotation_output),
                    ]
                ),
                "gripper_pos": float(action_raw[19]),
            },
        }

    if action_format == "dual_arm_ee_pose6d" or action_raw.size == 14:
        if action_raw.size != 14:
            raise ValueError(f"Expected 14D dual-arm EE action, got {action_raw.size}")
        return {
            "left_arm": {
                "pose_6d": action_raw[:6].copy(),
                "gripper_pos": float(action_raw[12]),
            },
            "right_arm": {
                "pose_6d": action_raw[6:12].copy(),
                "gripper_pos": float(action_raw[13]),
            },
        }

    raise ValueError(
        f"Unsupported action_format={action_format!r}, action_dim={action_raw.size}"
    )


def infer_action_meta(cfg, degree, rotation_output_override=None):
    action_shape = cfg.shape_meta["action"]["shape"]
    action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[-1]
    task_cfg = _cfg_get(cfg, "task", {})
    dataset_cfg = _cfg_get(task_cfg, "dataset", {}) if task_cfg is not None else {}
    dataset_target = str(_cfg_get(dataset_cfg, "_target_", "")).lower()
    is_bspline = "bspline" in dataset_target

    action_format = "bspline"
    action_layout = None
    rotation_output = None
    obs_keys = set(cfg.shape_meta["obs"].keys())
    rotation_rep = _cfg_get(dataset_cfg, "rotation_rep", None)
    if action_dim == 10 and rotation_rep == "rotation_6d":
        if {"arm_pos", "arm_quat", "gripper_pos"} <= obs_keys:
            action_format = "single_yam_rot6d"
        elif {"arm_pos_l", "arm_quat_l", "gripper_pos_l"} <= obs_keys:
            action_format = "single_left_arm_rot6d"
        elif {"arm_pos_r", "arm_quat_r", "gripper_pos_r"} <= obs_keys:
            action_format = "single_right_arm_rot6d"
        elif any(key.startswith("left_") for key in obs_keys):
            action_format = "single_left_arm_rot6d"
        else:
            action_format = "single_right_arm_rot6d"
        action_layout = "arm_xyz,arm_rot6d,gripper"
    elif action_dim == 23 and rotation_rep == "rotation_6d":
        action_format = "real_bimanual_base_rot6d"
        action_layout = (
            "base_velocity,left_pos,left_rot6d,left_gripper,"
            "right_pos,right_rot6d,right_gripper"
        )
    elif action_dim == 20 and rotation_rep == "rotation_6d":
        action_format = "dual_arm_ee_rot6d"
        if rotation_output_override is not None:
            rotation_output = rotation_output_override
        elif _cfg_get(dataset_cfg, "action_from_next_obs", False):
            rotation_output = "euler_xyz"
        else:
            from_rep = _cfg_get(dataset_cfg, "from_rotation_rep", "axis_angle")
            if from_rep == "axis_angle":
                rotation_output = "axis_angle"
            elif from_rep in ("euler_angles", "quaternion", "matrix"):
                rotation_output = "euler_xyz"
            else:
                raise ValueError(f"Unsupported from_rotation_rep: {from_rep}")
        action_layout = (
            "left_xyz,left_rot6d,right_xyz,right_rot6d,"
            "left_gripper,right_gripper"
        )
    elif action_dim == 14:
        action_format = "dual_arm_ee_pose6d"
        action_layout = "left_pose6d,right_pose6d,left_gripper,right_gripper"

    return {
        "action_format": action_format,
        "action_dim": int(action_dim),
        "raw_bspline_dim": int(action_dim + 1 if is_bspline else action_dim),
        "action_layout": action_layout,
        "rotation_output": rotation_output,
        "degree": int(degree),
        "relative_knots": bool(_cfg_get(dataset_cfg, "relative_knots", False)),
    }
