# Author: Haoyu Xiong
# Date: June 8, 2026

import numpy as np
from cameras import OAKCamera
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import WRIST_CAMERA_SERIAL, RAW_IMAGE_WIDTH, RAW_IMAGE_HEIGHT
from multiprocessing.managers import BaseManager as MPBaseManager


class YamArmManager(MPBaseManager):
    pass


YamArmManager.register("YamArm")

class RealEnv:
    def __init__(self, use_cameras=True, stiffness_kp_scale=1.0):
        # RPC server connection for the single YAM arm
        arm_manager = YamArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        try:
            arm_manager.connect()
        except ConnectionRefusedError as e:
            raise Exception('Could not connect to arm RPC server, is yam_server.py running?') from e

        # RPC proxy object
        self.arm = arm_manager.YamArm()

        # Stiffen (or soften) the arm's joint position gain. 1.0 = server default;
        # >1.0 makes the arm hold commanded joint positions more stiffly.
        if float(stiffness_kp_scale) != 1.0:
            self.arm.set_stiffness_scale(float(stiffness_kp_scale))

        # Wrist camera (Luxonis OAK / DepthAI); optional. Configure the device id in
        # constants.py. Pass use_cameras=False (main.py --no-cameras) to skip it.
        serial = str(WRIST_CAMERA_SERIAL).strip() if WRIST_CAMERA_SERIAL else ''
        if use_cameras:
            device_id = serial if serial and serial != 'TODO' else None
            self.wrist_camera = OAKCamera(device_id=device_id)
        else:
            self.wrist_camera = None

    def get_obs(self):
        obs = {}
        obs.update(self.arm.get_state())
        if self.wrist_camera is not None:
            obs['wrist_image'] = self.wrist_camera.get_image()
        else:
            obs['wrist_image'] = np.zeros((RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH, 3), dtype=np.uint8)
        return obs

    def get_cameras(self):
        """Return a name→OAKCamera dict for visualization (empty if cameras disabled)."""
        cameras = {}
        if self.wrist_camera is not None:
            cameras['wrist'] = self.wrist_camera
        return cameras

    def reset(self):
        print('Resetting arm...')
        self.arm.reset()
        print('Robot has been reset')

    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        self.arm.execute_action(action)  # Non-blocking

    def close(self):
        self.arm.close()
        if self.wrist_camera is not None:
            self.wrist_camera.close()

if __name__ == '__main__':
    import time
    from constants import POLICY_CONTROL_PERIOD
    env = RealEnv()
    try:
        while True:
            env.reset()
            for _ in range(100):
                action = {
                    'arm_pos': 0.1 * np.random.rand(3) + np.array([0.4, 0.0, 0.3]),
                    'arm_quat': np.random.rand(4),
                    'gripper_pos': np.random.rand(1),
                }
                env.step(action)
                obs = env.get_obs()
                print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
