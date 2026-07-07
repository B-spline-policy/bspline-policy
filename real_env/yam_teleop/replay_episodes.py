# Author: Haoyu Xiong
# Date: June 8, 2026

import argparse
import time
from itertools import count
from pathlib import Path
import cv2 as cv
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeReader

def replay_episode(env, episode_dir, show_images=False, execute_obs=False):
    # Reset env
    env.reset()

    # Load episode data
    reader = EpisodeReader(episode_dir)
    print(f'Loaded episode from {episode_dir}')

    start_time = time.time()
    for step_idx, (obs, action) in enumerate(zip(reader.observations, reader.actions)):
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Show image observations
        if show_images:
            window_idx = 0
            for k, v in obs.items():
                if v.ndim == 3:
                    cv.imshow(k, cv.cvtColor(v, cv.COLOR_RGB2BGR))
                    cv.moveWindow(k, 640 * window_idx, 0)
                    window_idx += 1
            cv.waitKey(1)

        # Execute in action in env
        if execute_obs:
            env.step(obs)
        else:
            env.step(action)

def main(args):
    # Create env (no live camera needed; images come from the saved episode)
    from real_env import RealEnv
    env = RealEnv(use_cameras=False)

    try:
        episode_dirs = sorted([child for child in Path(args.input_dir).iterdir() if child.is_dir()])
        for episode_dir in episode_dirs:
            replay_episode(env, episode_dir, show_images=args.show_images, execute_obs=args.execute_obs)
            # input('Press <Enter> to continue...')
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/demos')
    parser.add_argument('--show-images', action='store_true')
    parser.add_argument('--execute-obs', action='store_true')
    main(parser.parse_args())
