# Author: Haoyu Xiong
# Date: June 8, 2026
#
# RPC server for a single YAM arm (I2RT MotorChainRobot) with PyRoKi + J-PARSE IK.
#
# This replaces the Kinova arm_server.py: it exposes the same RPC surface
# (reset / execute_action / get_state / close) on the same arm RPC host/port
# (ARM_RPC_HOST / ARM_RPC_PORT), so real_env.py / main.py drive it unchanged.
#
# Dependencies (in addition to the base requirements):
#   pip install jax jaxlib jaxlie yourdfpy pyroki
#   pip install -e ../i2rt        # or rely on the sibling-dir sys.path shims below
#
# Phone / policy teleop (main.py): start this server, then run main.py as usual.
# The arm servos at YAM_CONTROL_HZ toward the latest Cartesian goal (i2rt-style
# streaming, not per-RPC IK bursts).
#
# Use multiprocessing "spawn" (set below) so JAX is not initialized in a forked process.

from __future__ import annotations

import argparse
import sys
import threading
import time
from multiprocessing.managers import BaseManager as MPBaseManager
from pathlib import Path

# -----------------------------------------------------------------------------
# Repo paths: simple_mobile/yam_teleop -> simple_mobile/i2rt, simple_mobile/pyroki
# Add the sibling i2rt and pyroki source dirs to sys.path BEFORE importing pyroki /
# i2rt so they resolve without a separate `pip install -e`.
# -----------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parent.parent
_I2RT_ROOT = _SRC_ROOT / "i2rt"
if _I2RT_ROOT.is_dir() and str(_I2RT_ROOT) not in sys.path:
    sys.path.insert(0, str(_I2RT_ROOT))
_PYROKI_SRC = _SRC_ROOT / "pyroki" / "src"
if _PYROKI_SRC.is_dir() and str(_PYROKI_SRC) not in sys.path:
    sys.path.insert(0, str(_PYROKI_SRC))

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation as R

# The arm now lives on the canonical arm RPC port (ARM_RPC_HOST / ARM_RPC_PORT),
# so the rest of the stack (real_env.py, main.py) is unchanged.
from constants import ARM_RPC_HOST
from constants import ARM_RPC_PORT
from constants import RPC_AUTHKEY
from constants import YAM_ARM_TYPE_STR
from constants import YAM_CAN_CHANNEL
from constants import YAM_CONTROL_PERIOD
from constants import YAM_GRIPPER_TYPE_STR
from constants import YAM_GRIPPER_INVERT
from constants import YAM_GRIPPER_MAX_SPEED
from constants import YAM_JOINT_CMD_ALPHA
from constants import YAM_INITIAL_JOINTS
from constants import YAM_RESET_DURATION_S
from constants import YAM_USE_SIM
from yam_jparse import jparse_step

from i2rt.robots.get_robot import get_yam_robot  # noqa: E402
from i2rt.robots.utils import ArmType, GripperType  # noqa: E402
from i2rt.robots.utils import ARM_YAM_XML_PATH  # noqa: E402

# Fixed transform link_6 -> TCP (same frame convention as TRI Raiden yam stack).
_T_LINK6_TO_TCP: np.ndarray = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_T_TCP_TO_LINK6: np.ndarray = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _grip_downstream_to_yam(g: float) -> float:
    """Convert a downstream gripper command (0=open, 1=closed) to YAM units (0=closed, 1=open)."""
    g = float(g)
    return (1.0 - g) if YAM_GRIPPER_INVERT else g


def _grip_yam_to_downstream(g: float) -> float:
    """Convert a YAM gripper reading (0=closed, 1=open) to the downstream convention (0=open, 1=closed)."""
    g = float(g)
    return (1.0 - g) if YAM_GRIPPER_INVERT else g


def _smooth_move_to_joint_q(
    robot,
    q_goal: np.ndarray,
    duration_s: float,
    dt: float,
) -> None:
    """Ramp joint commands from current pose to ``q_goal`` (MuJoCo 7-vector) at rate ``dt``."""
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(7)
    q_start = np.asarray(robot.get_joint_pos(), dtype=np.float64).copy()
    steps = max(1, int(duration_s / dt))
    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        q_cmd = (1.0 - alpha) * q_start + alpha * q_goal
        robot.command_joint_pos(q_cmd)
        time.sleep(dt)


def _mujoco_to_pyroki(q_mujoco6: np.ndarray) -> np.ndarray:
    return np.asarray(q_mujoco6, dtype=np.float64)[::-1].copy()


def _pyroki_to_mujoco(q_pk6: np.ndarray) -> np.ndarray:
    return np.asarray(q_pk6, dtype=np.float64)[::-1].copy()


def _load_yam_urdf() -> yourdfpy.URDF:
    urdf_path = Path(ARM_YAM_XML_PATH).with_suffix(".urdf")
    assets_dir = urdf_path.parent / "assets"

    def _pkg_handler(fname, dir=None):  # noqa: A002
        if isinstance(fname, str) and fname.startswith("package://assets/"):
            return str(assets_dir / fname.replace("package://assets/", ""))
        return fname

    return yourdfpy.URDF.load(
        str(urdf_path),
        filename_handler=_pkg_handler,
        load_meshes=False,
        load_collision_meshes=False,
    )


def _setup_pyroki_ik(dt: float):
    urdf = _load_yam_urdf()
    pk_robot = pk.Robot.from_urdf(urdf)
    link6_idx = list(pk_robot.links.names).index("link_6")
    step_jit = jax.jit(jparse_step, static_argnames=("method",))

    dummy = np.zeros(6, dtype=np.float64)
    result, _ = step_jit(
        robot=pk_robot,
        cfg=dummy,
        target_link_index=link6_idx,
        target_position=np.zeros(3, dtype=np.float64),
        target_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        method="jparse",
        dt=dt,
        home_cfg=dummy,
    )
    jax.block_until_ready(result)
    # Warm up forward_kinematics too — its first call JIT-compiles and can
    # block the command loop long enough to trip DM motor comm watchdogs.
    fk_out = pk_robot.forward_kinematics(jnp.asarray(dummy, dtype=jnp.float64))
    jax.block_until_ready(fk_out)
    return pk_robot, link6_idx, step_jit


def _pose_to_T_tcp(pos_xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    return T


def _T_tcp_to_ik_targets(T_tcp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_link6 = T_tcp @ _T_TCP_TO_LINK6
    target_pos = T_link6[:3, 3].astype(np.float64)
    xyzw = R.from_matrix(T_link6[:3, :3]).as_quat()
    target_wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return target_pos, target_wxyz


def _fk_T_tcp(pk_robot, link6_idx: int, q_pyroki: np.ndarray) -> np.ndarray:
    poses = pk_robot.forward_kinematics(jnp.asarray(q_pyroki, dtype=jnp.float64))
    T_link6 = np.array(jaxlie.SE3(poses[link6_idx]).as_matrix(), dtype=np.float64)
    return T_link6 @ _T_LINK6_TO_TCP


def _fk_tcp(pk_robot, link6_idx: int, q_pyroki: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T_tcp = _fk_T_tcp(pk_robot, link6_idx, q_pyroki)
    pos = T_tcp[:3, 3].copy()
    quat_xyzw = R.from_matrix(T_tcp[:3, :3]).as_quat()
    if quat_xyzw[3] < 0.0:
        quat_xyzw = -quat_xyzw
    return pos, quat_xyzw


def _ik_step_once(
    pk_robot,
    link6_idx: int,
    step_jit,
    q_pyroki: np.ndarray,
    T_target_tcp: np.ndarray,
    home_cfg: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One J-PARSE velocity-IK integration step (matches one tick of Raiden-style EE teleop)."""
    target_pos, target_wxyz = _T_tcp_to_ik_targets(T_target_tcp)
    q_new, _ = step_jit(
        robot=pk_robot,
        cfg=np.asarray(q_pyroki, dtype=np.float64),
        target_link_index=link6_idx,
        target_position=target_pos,
        target_wxyz=target_wxyz,
        method="jparse",
        dt=dt,
        home_cfg=home_cfg,
    )
    return np.asarray(q_new, dtype=np.float64)


class YamArm:
    """Single YAM arm exposing the same RPC surface as the Kinova `Arm` (execute_action / get_state / reset / close).

    The CAN interface is ``YAM_CAN_CHANNEL`` by default; override per process with ``--channel`` (this sets
    ``YamArm.server_channel`` before ``serve_forever``; see ``__main__``).

    Control follows i2rt-style joint streaming (~100 Hz): a fixed-period loop commands
    ``MotorChainRobot.command_joint_pos``. Cartesian targets from ``execute_action`` update the latest
    goal (no queue drops). IK is one J-PARSE step per tick with ``dt`` = control period, integrating a
    virtual joint state—not a multi-step solve per RPC call.
    """

    #: CAN channel for this server process (set from ``--channel`` in RPC mode).
    server_channel: str | None = None

    def __init__(self):
        self._robot = None
        self._pk_robot = None
        self._link6_idx = None
        self._step_jit = None
        self._home_cfg = np.zeros(6, dtype=np.float64)
        self._target_lock = threading.Lock()
        self._target_T = np.eye(4, dtype=np.float64)
        self._target_gripper = 1.0
        self._q_pyroki: np.ndarray | None = None
        self._q_cmd: np.ndarray | None = None
        self._control_stop = threading.Event()
        self._control_pause = threading.Event()
        self._loop_tick_lock = threading.Lock()
        self._control_thread: threading.Thread | None = None
        self._ik_dt_scale = 1.0
        # Default arm gains captured once from get_yam_robot; stiffness is applied
        # relative to these so repeated set_stiffness_scale calls don't compound.
        self._base_kp: np.ndarray | None = None
        self._base_kd: np.ndarray | None = None
        self._stiffness_kp_scale = 1.0

    def _ensure_robot(self):
        if self._robot is not None:
            return
        # JIT-compile pyroki BEFORE bringing up the motor chain. get_yam_robot()
        # spawns a CAN command thread that must tick at ~YAM_CONTROL_HZ; a JAX
        # compile running after it starves that thread and trips DM motor
        # comm-loss watchdogs.
        if self._pk_robot is None:
            self._pk_robot, self._link6_idx, self._step_jit = _setup_pyroki_ik(YAM_CONTROL_PERIOD)
        arm_type = ArmType.from_string_name(YAM_ARM_TYPE_STR)
        gripper_type = GripperType.from_string_name(YAM_GRIPPER_TYPE_STR)
        channel = self.server_channel or YAM_CAN_CHANNEL
        self._robot = get_yam_robot(
            channel=channel,
            arm_type=arm_type,
            gripper_type=gripper_type,
            zero_gravity_mode=False,
            sim=YAM_USE_SIM,
        )
        info = self._robot.get_robot_info()
        self._base_kp = np.asarray(info["kp"], dtype=np.float64).copy()
        self._base_kd = np.asarray(info["kd"], dtype=np.float64).copy()
        if self._stiffness_kp_scale != 1.0:
            self._apply_stiffness_scale()
        self._control_stop.clear()
        self._control_pause.clear()
        self._control_thread = threading.Thread(target=self._control_loop, name="yam-control", daemon=True)
        self._control_thread.start()

    def _control_loop(self):
        period = YAM_CONTROL_PERIOD
        dt = period
        while not self._control_stop.is_set():
            if self._control_pause.is_set():
                time.sleep(0.002)
                continue
            with self._loop_tick_lock:
                t0 = time.perf_counter()
                try:
                    if self._q_pyroki is None:
                        q_full = self._robot.get_joint_pos()
                        self._q_pyroki = _mujoco_to_pyroki(q_full[:6])
                        self._q_cmd = q_full.astype(np.float64).copy()
                        with self._target_lock:
                            self._target_T = _fk_T_tcp(self._pk_robot, self._link6_idx, self._q_pyroki)
                            self._target_gripper = float(self._q_cmd[6])
                    with self._target_lock:
                        T_tgt = self._target_T.copy()
                        g_tgt = self._target_gripper
                    ik_dt = dt * float(self._ik_dt_scale)
                    self._q_pyroki = _ik_step_once(
                        self._pk_robot,
                        self._link6_idx,
                        self._step_jit,
                        self._q_pyroki,
                        T_tgt,
                        self._home_cfg,
                        ik_dt,
                    )
                    q_arm = _pyroki_to_mujoco(self._q_pyroki)
                    a = float(YAM_JOINT_CMD_ALPHA)
                    if a >= 1.0:
                        self._q_cmd[:6] = q_arm
                    else:
                        self._q_cmd[:6] = (1.0 - a) * self._q_cmd[:6] + a * q_arm
                    # Slew-rate limit the gripper so it ramps smoothly toward the target
                    # instead of stepping with the 10 Hz teleop setpoint (the arm joints
                    # already get per-tick IK integration + YAM_JOINT_CMD_ALPHA).
                    # Scale by ik_dt (= dt * ik_dt_scale), NOT raw dt, so the same
                    # set_ik_dt_scale(speed_up_times) knob that lets the arm track sped-up
                    # replay also lets the gripper keep pace — otherwise the fixed-rate
                    # gripper becomes the bottleneck at speed_up_times > 1.
                    g_max_step = float(YAM_GRIPPER_MAX_SPEED) * ik_dt
                    if g_max_step > 0.0:
                        g_prev = float(self._q_cmd[6])
                        self._q_cmd[6] = g_prev + np.clip(g_tgt - g_prev, -g_max_step, g_max_step)
                    else:
                        self._q_cmd[6] = g_tgt
                    self._robot.command_joint_pos(self._q_cmd.copy())
                except Exception as e:
                    print(f"[yam_server] control loop: {e}")
                elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, period - elapsed))

    def reset(self):
        self._ensure_robot()
        self._control_pause.set()
        # Hold _loop_tick_lock for the entire reset: waits for any in-progress tick
        # and blocks any rogue tick that slipped past the _control_pause check before
        # it was set (race: loop passed pause check → reset sets pause → loop acquires
        # lock → concurrent CAN access with _smooth_move_to_joint_q).
        with self._loop_tick_lock:
            _smooth_move_to_joint_q(
                self._robot,
                YAM_INITIAL_JOINTS,
                YAM_RESET_DURATION_S,
                YAM_CONTROL_PERIOD,
            )
            self._q_pyroki = None
            self._q_cmd = None
            q_full = self._robot.get_joint_pos()
            q_pk = _mujoco_to_pyroki(q_full[:6])
            with self._target_lock:
                self._target_T = _fk_T_tcp(self._pk_robot, self._link6_idx, q_pk)
                self._target_gripper = float(q_full[6])
        self._control_pause.clear()

    def execute_action(self, action):
        self._ensure_robot()
        T = _pose_to_T_tcp(action["arm_pos"], action["arm_quat"])
        grip = _grip_downstream_to_yam(np.asarray(action["gripper_pos"]).reshape(-1)[0])
        with self._target_lock:
            self._target_T[:] = T
            self._target_gripper = grip

    def set_ik_dt_scale(self, scale):
        # Multiplier on the per-tick J-PARSE integration dt. Larger values let the arm
        # slew further toward the latest target each control tick; useful when commanded
        # TCP targets move faster than the default servo can follow (e.g. sped-up replay).
        # The gripper slew-rate limit (YAM_GRIPPER_MAX_SPEED) scales by the same factor,
        # so set_ik_dt_scale(speed_up_times) speeds up the arm and gripper together.
        # Trade-off: per-tick joint deltas grow linearly with the scale, which can hit
        # joint velocity limits or destabilize J-PARSE near singularities.
        s = float(scale)
        if not (s > 0.0):
            raise ValueError(f"ik_dt_scale must be > 0, got {scale!r}")
        self._ik_dt_scale = s

    def _apply_stiffness_scale(self):
        # Scale only the 6 arm joints' kp; leave the gripper kp (last element) and
        # all kd untouched. command_joint_pos reads self._kp every tick, so this
        # takes effect on the next control tick.
        kp = self._base_kp.copy()
        kp[:6] *= self._stiffness_kp_scale
        self._robot.update_kp_kd(kp, self._base_kd.copy())

    def set_stiffness_scale(self, kp_scale):
        # Multiplier on the arm-joint position gain (kp), applied relative to the
        # default gains from get_yam_robot (kp[:6] = [80, 80, 80, 40, 10, 10]).
        # kp_scale == 1.0 is the default; >1.0 makes the arm hold its commanded
        # joint positions more stiffly. Applied to the 6 arm joints only (gripper
        # kp and all kd are left at default), and relative to the base so repeated
        # calls don't compound. Trade-off: too-large kp can cause buzzing /
        # oscillation, especially on contact.
        s = float(kp_scale)
        if not (s > 0.0):
            raise ValueError(f"stiffness kp_scale must be > 0, got {kp_scale!r}")
        self._stiffness_kp_scale = s
        self._ensure_robot()
        self._apply_stiffness_scale()

    def get_state(self):
        self._ensure_robot()
        q_full = self._robot.get_joint_pos()
        q_pyroki = _mujoco_to_pyroki(q_full[:6])
        arm_pos, arm_quat = _fk_tcp(self._pk_robot, self._link6_idx, q_pyroki)
        return {
            "arm_pos": arm_pos,
            "arm_quat": arm_quat,
            "gripper_pos": np.array([_grip_yam_to_downstream(q_full[6])]),
        }

    def close(self):
        self._control_stop.set()
        if self._control_thread is not None:
            self._control_thread.join(timeout=3.0)
            self._control_thread = None
        if self._robot is not None:
            self._robot.close()
            self._robot = None
        self._q_pyroki = None
        self._q_cmd = None


class YamArmManager(MPBaseManager):
    pass


YamArmManager.register("YamArm", YamArm)


if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Single YAM arm RPC server (drop-in for the Kinova arm_server.py)")
    parser.add_argument("--channel", type=str, default=YAM_CAN_CHANNEL, help="SocketCAN interface for the arm.")
    parser.add_argument("--rpc-host", type=str, default=ARM_RPC_HOST, help="RPC bind address.")
    parser.add_argument(
        "--rpc-port",
        type=int,
        default=ARM_RPC_PORT,
        help="RPC port (defaults to the shared arm RPC port).",
    )
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    YamArm.server_channel = args.channel
    manager = YamArmManager(address=(args.rpc_host, args.rpc_port), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(
        f"YAM arm manager at {args.rpc_host}:{args.rpc_port} "
        f"(channel={args.channel!r}; spawn; JAX-safe)"
    )
    server.serve_forever()
