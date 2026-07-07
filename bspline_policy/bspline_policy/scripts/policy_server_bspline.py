# Date: May 2026
#
# B-spline policy server for simple_mobile.
#
# This is separate from policy_server.py on purpose. It returns raw B-spline
# parameters under the "bspline" key, plus metadata under "bspline_meta".

import argparse
import sys
from collections import deque
from pathlib import Path

BSPLINE_POLICY_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BSPLINE_POLICY_DIR.parent
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"
for path in (BSPLINE_POLICY_DIR, DIFFUSION_POLICY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cv2 as cv
import dill
import hydra
import numpy as np
import torch
import zmq

from bspline_policy.common.knots import decode_relative_knots
from bspline_policy.common.bspline_action import decode_bspline_action
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def safer_knots(knots):
    knots = np.asarray(knots, dtype=np.float64).copy()
    for idx in range(1, len(knots)):
        if knots[idx] < knots[idx - 1]:
            knots[idx] = knots[idx - 1] + 1e-6
    return knots


def infer_action_meta(cfg, degree):
    action_shape = cfg.shape_meta["action"]["shape"]
    action_dim = action_shape[0] if len(action_shape) == 1 else action_shape[-1]
    dataset_cfg = _cfg_get(_cfg_get(cfg, "task", None), "dataset", None)
    return {
        "action_format": "real_bimanual_base_rot6d",
        "action_layout": (
            "base_velocity,left_pos,left_rot6d,left_gripper,"
            "right_pos,right_rot6d,right_gripper"
        ),
        "action_dim": int(action_dim),
        "bspline_channels": int(action_dim) + 1,
        "degree": int(degree),
        "relative_knots": bool(_cfg_get(dataset_cfg, "relative_knots", False)),
    }


class BSplinePolicy:
    def __init__(self, ckpt_path, degree=None, device="cuda"):
        with open(ckpt_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        self.device = torch.device(device)
        policy.eval().to(self.device)

        self.policy = policy
        self.obs_shape_meta = cfg.shape_meta["obs"]
        self.degree = int(degree if degree is not None else cfg.policy.get("bspline_degree", 3))
        self.action_meta = infer_action_meta(cfg, self.degree)
        self.n_action_steps = int(cfg.get("n_action_steps", 8))
        self.warmed_up = False

        print(f"Loaded B-spline policy from: {ckpt_path}")
        print(f"  - device: {self.device}")
        print(f"  - degree: {self.degree}")
        print(f"  - relative_knots: {self.action_meta['relative_knots']}")
        print(f"  - action_dim: {self.action_meta['action_dim']}")

    def get_action_metadata(self):
        return self.action_meta

    def reset(self):
        self.policy.reset()
        self.warmed_up = False

    def step(self, obs_sequence):
        obs_dict = self._convert_obs(obs_sequence)
        with torch.no_grad():
            if not self.warmed_up:
                print("Warming up policy...")
                self.policy.predict_action(obs_dict)
                self.warmed_up = True
                print("Policy warmed up")

            result = self.policy.predict_action(obs_dict)
            action = result["action"][0].detach().to("cpu").numpy()
            if self.action_meta["relative_knots"]:
                action = decode_relative_knots(action, degree=self.degree)

        if action.ndim != 2 or action.shape[1] != self.action_meta["bspline_channels"]:
            raise ValueError(
                "Expected B-spline action shape "
                f"(T, {self.action_meta['bspline_channels']}), got {action.shape}"
            )
        return action

    def _convert_obs(self, obs_sequence):
        obs_dict_np = {}
        for key, value in self.obs_shape_meta.items():
            if value.get("type") == "rgb":
                images = np.stack([obs[key] for obs in obs_sequence], axis=0)
                if images.dtype != np.uint8:
                    raise AssertionError(f"{key} image dtype must be uint8")
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))
                if images.shape[1:] != tuple(value["shape"]):
                    raise AssertionError(
                        f"{key} shape mismatch: expected {value['shape']}, "
                        f"got {images.shape[1:]}"
                    )
                obs_dict_np[key] = images
            else:
                obs_dict_np[key] = np.stack(
                    [obs[key] for obs in obs_sequence], axis=0
                ).astype(np.float32)
        return dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device),
        )


class YamBSplineActionPolicy(BSplinePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation_transformer = RotationTransformer(
            from_rep="rotation_6d",
            to_rep="quaternion",
        )

    def step(self, obs_sequence):
        return self.decode_action_sequence(super().step(obs_sequence))

    def decode_action_sequence(self, bspline):
        if self.action_meta["action_dim"] != 10:
            raise ValueError(
                "YAM action protocol expects 10D single-arm actions, "
                f"got action_dim={self.action_meta['action_dim']}"
            )
        bspline = np.asarray(bspline, dtype=np.float64).copy()
        bspline[:, 0] = safer_knots(bspline[:, 0])
        action = decode_bspline_action(
            bspline,
            degree=self.degree,
            num_actions=self.n_action_steps,
            relative_knots=False,
        )
        act_sequence = []
        for act in action:
            act_sequence.append(
                {
                    "arm_pos": act[0:3],
                    "arm_quat": self.rotation_transformer.forward(act[3:9])[[1, 2, 3, 0]],
                    "gripper_pos": act[9:10],
                }
            )
        return act_sequence


class PolicyWrapper:
    def __init__(self, policy, n_obs_steps=2):
        self.policy = policy
        self.n_obs_steps = n_obs_steps
        self.obs_history = deque(maxlen=n_obs_steps)

    def reset(self):
        self.policy.reset()
        self.obs_history.clear()

    def step(self, obs, model_infer=True):
        self.obs_history.append(obs)
        if len(self.obs_history) < self.n_obs_steps:
            print(f"Collecting observations: {len(self.obs_history)}/{self.n_obs_steps}")
            return None
        if not model_infer:
            return None
        return self.policy.step(list(self.obs_history))


class PolicyServer:
    def __init__(self, policy, port=5555):
        self.policy = policy
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"B-spline policy server started on port {port}")

    def _process_single_obs(self, obs):
        processed = {}
        image_sizes = {}
        policy_impl = getattr(self.policy, "policy", None)
        if policy_impl is not None:
            for key, value in policy_impl.obs_shape_meta.items():
                if value.get("type") == "rgb":
                    _, h, w = value["shape"]
                    image_sizes[key] = (w, h)

        for key, value in obs.items():
            if key.endswith("image"):
                if isinstance(value, np.ndarray) and value.ndim <= 2:
                    decoded = cv.imdecode(value, cv.IMREAD_COLOR)
                    if decoded is not None:
                        value = decoded
                    elif value.ndim == 2:
                        value = np.expand_dims(value, axis=-1)
                elif isinstance(value, np.ndarray) and value.ndim == 3:
                    if value.shape[0] in (1, 3):
                        value = np.moveaxis(value, 0, -1)
                    if value.dtype in (np.float32, np.float64):
                        value = (np.clip(value, 0, 1) * 255).astype(np.uint8)
                    elif value.dtype != np.uint8:
                        value = value.astype(np.uint8)
                if key in image_sizes:
                    value = cv.resize(value, image_sizes[key])
            processed[key] = value
        return processed

    def step(self, obs):
        if isinstance(obs, list):
            obs_sequence = obs
        else:
            obs_sequence = [obs]

        bspline = None
        for idx, single_obs in enumerate(obs_sequence):
            bspline = self.policy.step(
                self._process_single_obs(single_obs),
                model_infer=(idx == len(obs_sequence) - 1),
            )
        return bspline

    def run(self):
        while True:
            req = self.socket.recv_pyobj()
            rep = {}

            if "reset" in req:
                self.policy.reset()
                print("Policy has been reset")
            elif "obs" in req:
                rep["bspline"] = self.step(req["obs"])
                policy_impl = getattr(self.policy, "policy", None)
                if hasattr(policy_impl, "get_action_metadata"):
                    rep["bspline_meta"] = policy_impl.get_action_metadata()

            self.socket.send_pyobj(rep)


def main(
    ckpt_path,
    n_obs_steps=2,
    n_action_steps=None,
    degree=None,
    port=5555,
    device="cuda",
    response_format="bspline",
):
    policy_cls = YamBSplineActionPolicy if response_format == "action" else BSplinePolicy
    bspline_policy = policy_cls(ckpt_path, degree=degree, device=device)
    if n_action_steps is None:
        n_action_steps = bspline_policy.n_action_steps
    if response_format == "action":
        yam_teleop_dir = REPO_ROOT / "simple_mobile" / "yam_teleop"
        if str(yam_teleop_dir) not in sys.path:
            sys.path.insert(0, str(yam_teleop_dir))
        from policy_server import PolicyWrapper as ActionQueuePolicyWrapper
        from policy_server import PolicyServer as ActionPolicyServer

        policy = ActionQueuePolicyWrapper(
            bspline_policy,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
        )
        server = ActionPolicyServer(policy)
    else:
        policy = PolicyWrapper(
            bspline_policy,
            n_obs_steps=n_obs_steps,
        )
        server = PolicyServer(policy, port=port)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="B-spline policy server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", required=True, help="Path to checkpoint file")
    parser.add_argument("--n-obs-steps", type=int, default=2)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--response-format",
        choices=("bspline", "action"),
        default="bspline",
        help="Use action for compatibility with simple_mobile/yam_teleop RemotePolicy",
    )
    args = parser.parse_args()

    main(
        ckpt_path=args.ckpt_path,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        degree=args.degree,
        port=args.port,
        device=args.device,
        response_format=args.response_format,
    )
