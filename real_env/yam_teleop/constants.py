import numpy as np

################################################################################
# Teleop and imitation learning

# Arm RPC server (yam_server.py)
ARM_RPC_HOST = 'localhost'
ARM_RPC_PORT = 50001
RPC_AUTHKEY = b'secret password'

# Luxonis OAK (DepthAI) camera mounted on the YAM arm wrist. Set this to the
# camera's DepthAI device id / MXID (see depthai.Device.getAllAvailableDevices())
# to pin a specific unit, or leave as 'TODO' / None to open the first OAK found.
# When no camera is available, get_obs returns a black frame.
WRIST_CAMERA_SERIAL = 'TODO'

# Policy
POLICY_SERVER_HOST = 'localhost'
POLICY_SERVER_PORT = 5555
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ

# Raw camera capture resolution (what OAKCamera streams and what gets recorded).
RAW_IMAGE_WIDTH = 640
RAW_IMAGE_HEIGHT = 480
# Resolution the policy server expects (images are resized to this before inference).
POLICY_IMAGE_WIDTH = 84
POLICY_IMAGE_HEIGHT = 84

################################################################################
# YAM arm (i2rt MotorChainRobot + yam_server.py, single arm)
#
# yam_server.py serves the YAM arm on the same RPC host/port as the arm above
# (ARM_RPC_HOST / ARM_RPC_PORT), so real_env.py talks to "the arm" exactly as
# it did with the Kinova. Run a single server process:
#   python yam_server.py                       # CAN channel = YAM_CAN_CHANNEL
#   python yam_server.py --channel can0        # override the SocketCAN interface

# SocketCAN interface name for the arm (see `ip link`; errno 19 / ENODEV
# means the name is wrong or the link is down).
YAM_CAN_CHANNEL = 'can0'
YAM_ARM_TYPE_STR = 'yam'
YAM_GRIPPER_TYPE_STR = 'linear_4310'
# Gripper convention bridge. The teleop/policy stack (inherited from the Kinova arm)
# uses 0 = open, 1 = closed, but the YAM linear_4310 gripper uses 0 = closed,
# 1 = open. When True, yam_server.py inverts the gripper at the RPC boundary so the
# downstream convention (and any recorded demos) stay 0 = open, 1 = closed.
YAM_GRIPPER_INVERT = True
# Fixed-rate arm servo (i2rt minimum_gello follower uses ~100 Hz joint commands).
YAM_CONTROL_HZ = 100.0
YAM_CONTROL_PERIOD = 1.0 / YAM_CONTROL_HZ
# 1.0 = command raw IK joints each tick; lower = extra low-pass on top of velocity IK (smoother, more lag).
YAM_JOINT_CMD_ALPHA = 1.0
# Gripper command slew-rate limit (gripper units per second, range is [0, 1]) at 1x speed.
# The teleop/policy gripper setpoint only updates at POLICY_CONTROL_FREQ (10 Hz) and a fast
# phone drag can jump nearly the full range in one step; without limiting, the gripper
# slams between setpoints. The control loop ramps the command toward the target at this
# rate each tick instead. e.g. 3.0 = full open/close in ~0.33 s. Set <= 0 to disable.
# NOTE: this limit is scaled by set_ik_dt_scale (i.e. speed_up_times during sped-up
# replay), so the effective max speed is YAM_GRIPPER_MAX_SPEED * ik_dt_scale and the
# gripper keeps pace with the arm instead of bottlenecking fast replays. Set it near the
# gripper's hardware max speed so it smooths teleop without clipping 1x replays.
YAM_GRIPPER_MAX_SPEED = 3.0
YAM_USE_SIM = False
YAM_RESET_DURATION_S = 3.0
# MuJoCo joint order (same as MotorChainRobot.get_joint_pos): joint1..joint6 (rad) + gripper [0, 1].
# yam_server.YamArm.reset() ramps here with command_joint_pos; tune for your workspace.
YAM_INITIAL_JOINTS = np.array([0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
