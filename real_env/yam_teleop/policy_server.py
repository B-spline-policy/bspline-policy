# Author: Haoyu Xiong
# Date: June 8, 2026
#
# Note: This file is intended to be run within the diffusion_policy repository

import argparse
import math
import queue
import threading
import time
from collections import deque
import cv2 as cv
import dill
import hydra
import numpy as np
import torch
import zmq
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

POLICY_CONTROL_PERIOD = 0.1  # 100 ms (10 Hz)
LATENCY_BUDGET = 0.2  # 200 ms including policy inference and communication
LATENCY_STEPS = math.ceil(LATENCY_BUDGET / POLICY_CONTROL_PERIOD)  # Up to 3 is okay, 4 is too high

class StubDiffusionPolicy:
    def reset(self):
        pass

    def step(self, obs_sequence):
        obs = obs_sequence[-1]
        act_sequence = [f'{obs + i} (inference {obs})' for i in range(8)]
        time.sleep(0.115)  # 115 ms
        return act_sequence

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/eval_real_robot.py
class DiffusionPolicy:
    def __init__(self, ckpt_path):
        # Load checkpoint
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)

        # Load policy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)

        # Store attributes
        self.policy = policy
        self.device = device
        self.obs_shape_meta = cfg.shape_meta['obs']
        self.rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='quaternion')
        self.warmed_up = False

    def reset(self):
        self.policy.reset()

    def step(self, obs_sequence):
        obs_dict = self._convert_obs(obs_sequence)
        with torch.no_grad():
            if not self.warmed_up:
                print('Warming up policy...')
                self.policy.predict_action(obs_dict)
                self.warmed_up = True
            # start_time = time.time()
            result = self.policy.predict_action(obs_dict)
            # elapsed_time = time.time() - start_time
            # print(f'Inference time: {1000 * elapsed_time:.1f} ms')  # 115 ms (RTX 4080 Laptop)
            action = result['action'][0].detach().to('cpu').numpy()
        act_sequence = self._convert_action(action)
        return act_sequence

    def _convert_obs(self, obs_sequence):
        obs_dict_np = {}
        for key, value in self.obs_shape_meta.items():
            if value.get('type') == 'rgb':
                images = np.stack([obs[key] for obs in obs_sequence], axis=0)
                assert images.dtype == np.uint8
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))
                assert images.shape[1:] == tuple(value['shape'])
                obs_dict_np[key] = images
            else:
                obs_dict_np[key] = np.stack([obs[key] for obs in obs_sequence], axis=0).astype(np.float32)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        return obs_dict

    def _convert_action(self, action):
        act_sequence = []
        for act in action:
            action_dict = {
                'arm_pos': act[0:3],
                'arm_quat': self.rotation_transformer.forward(act[3:9])[[1, 2, 3, 0]],  # (w, x, y, z) -> (x, y, z, w)
                'gripper_pos': act[9:10],
            }
            act_sequence.append(action_dict)
        return act_sequence

class PolicyWrapper:
    def __init__(self, policy, n_obs_steps=2, n_action_steps=8):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()

        # Start inference loop
        threading.Thread(target=self.inference_loop, args=(policy,), daemon=True).start()

    def reset(self):
        self.obs_queue.put('reset')

    def step(self, obs):
        self.obs_queue.put(obs)
        action = None if self.act_queue.empty() else self.act_queue.get()
        if action is None:
            print('Warning: Unexpected idle action queue. Is the latency budget set too low?')
        return action

    def inference_loop(self, policy):
        obs_history = deque(maxlen=self.n_obs_steps)
        start_of_episode = True
        while True:
            # Check for new obs
            if not self.obs_queue.empty():
                obs = self.obs_queue.get()

                # Reset policy
                if obs == 'reset':
                    policy.reset()
                    obs_history.clear()
                    start_of_episode = True
                    while not self.act_queue.empty():
                        self.act_queue.get()
                    continue

                # Append obs to history
                obs_history.append(obs)

            if self.act_queue.qsize() < LATENCY_STEPS and len(obs_history) == self.n_obs_steps:
                obs_sequence = list(obs_history)
                act_sequence = policy.step(obs_sequence)
                if not self.act_queue.empty():
                    print('Warning: Unexpected action queue backlog. Is the latency budget set too high?')
                if start_of_episode:
                    act_sequence = act_sequence[:self.n_action_steps - LATENCY_STEPS]
                    start_of_episode = False
                else:
                    act_sequence = act_sequence[LATENCY_STEPS:self.n_action_steps]
                for action in act_sequence:
                    self.act_queue.put(action)

            time.sleep(0.001)

class PolicyServer:
    def __init__(self, policy):
        self.policy = policy

        # Set up ZMQ server
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        port = 5555
        self.socket.bind(f'tcp://*:{port}')
        print(f'Server started on port {port}')

    def step(self, obs):
        # Decode images
        for k, v in obs.items():
            if k.endswith('image'):
                v = cv.imdecode(v, cv.IMREAD_COLOR)  # Note: Interprets RGB as BGR
                # cv.imwrite(f'{k}.jpg', cv.cvtColor(v, cv.COLOR_RGB2BGR))
                obs[k] = v

        # Get action
        action = self.policy.step(obs)

        return action

    def run(self):
        while True:
            # Wait for request from client
            req = self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
            rep = {}

            # Reset policy
            if 'reset' in req:
                self.policy.reset()
                print('Policy has been reset')

            # Get action
            elif 'obs' in req:
                obs = req['obs']
                action = self.step(obs)
                rep['action'] = action

            # Send reply to client
            self.socket.send_pyobj(rep)

def main(ckpt_path):
    policy = PolicyWrapper(DiffusionPolicy(ckpt_path))
    server = PolicyServer(policy)
    server.run()

if __name__ == '__main__':
    # policy = PolicyWrapper(StubDiffusionPolicy())
    # policy.reset()
    # for step_num in range(1, 9999):
    #     print(f'obs: {step_num}, action: {policy.step(step_num)}')
    #     time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default='data/outputs/2024.10.08/23.42.04_train_diffusion_unet_hybrid_sim-v1/checkpoints/epoch=0500-train_loss=0.001.ckpt')
    args = parser.parse_args()
    main(args.ckpt_path)
