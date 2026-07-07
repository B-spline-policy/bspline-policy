# Author: Haoyu Xiong
# Date: June 8, 2026

import argparse
import subprocess
import time
from itertools import count
from constants import POLICY_CONTROL_PERIOD
from episode_storage import EpisodeWriter
from policies import TeleopPolicy, RemotePolicy

def say(text):
    # Spoken prompt via spd-say (speech-dispatcher); no-op if not installed
    try:
        subprocess.Popen(['spd-say', text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass

def run_episode(env, policy, writer=None):
    # Reset the env
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    # Wait for user to press "Start episode"
    print('Press "Start episode" in the web app when ready to start new episode')
    say('please start')
    policy.reset()
    print('Starting new episode')

    episode_ended = False
    start_time = time.time()
    for step_idx in count():
        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Get latest observation
        obs = env.get_obs()

        # Get action
        action = policy.step(obs)

        # No action if teleop not enabled
        if action is None:
            continue

        # Execute valid action on robot
        if isinstance(action, dict):
            env.step(action)

            if writer is not None and not episode_ended:
                # Record executed action
                writer.step(obs, action)

        # Episode ended
        elif not episode_ended and action == 'end_episode':
            episode_ended = True
            print('Episode ended')

            if writer is not None:
                # Always save the episode (no prompt); skip only if it's empty
                if len(writer) == 0:
                    print('Discarding empty episode')
                else:
                    # Save to disk in background thread
                    writer.flush_async()

            print('Teleop is now active. Press "Reset env" in the web app when ready to proceed.')
            say('please reset')

        # Ready for env reset
        elif action == 'reset_env':
            break

    if writer is not None:
        # Wait for writer to finish saving to disk
        writer.wait_for_flush()

def main(args):
    # Create env
    from real_env import RealEnv
    env = RealEnv()

    # Create policy
    if args.teleop:
        policy = TeleopPolicy()
    else:
        policy = RemotePolicy()

    try:
        while True:
            writer = EpisodeWriter(args.output_dir) if args.save else None
            run_episode(env, policy, writer)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output-dir', default='data/demos')
    main(parser.parse_args())
