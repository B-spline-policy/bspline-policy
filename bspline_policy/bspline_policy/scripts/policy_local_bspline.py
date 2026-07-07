import inspect
import math
import queue
import sys
import threading
import time
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
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

YAM_TELEOP_DIR = REPO_ROOT / "real_env" / "yam_teleop"
if str(YAM_TELEOP_DIR) not in sys.path:
    sys.path.insert(0, str(YAM_TELEOP_DIR))

from policy_local_utils import (
    _add_diffusion_policy_to_path,
    _cfg_get,
    decode_action_vector,
    infer_action_meta,
)


def safer_knots(knots):
    knots = np.asarray(knots, dtype=np.float64).copy()
    for idx in range(1, len(knots)):
        if knots[idx] < knots[idx - 1]:
            knots[idx] = knots[idx - 1] + 1e-6
    return knots


class CudaGraphDDIMSampler:
    """Replay the whole DDIM denoising loop as a single CUDA graph.

    Batch-1 diffusion inference is dominated by CPU kernel-launch overhead:
    each denoising step issues hundreds of tiny kernels and the GPU idles
    between launches. Capturing the fixed-shape denoising loop once and
    replaying it collapses all launches into one driver call. The replayed
    kernels are identical to eager ones, so outputs match bit-for-bit up to
    float32 rounding in the precomputed DDIM coefficients.

    Only the fixed deployment configuration is captured: DDIMScheduler with
    epsilon prediction, eta=0, an all-False conditioning mask, and the tensor
    shapes / num_inference_steps seen at capture time. Any other call falls
    back to the original eager `conditional_sample`.
    """

    def __init__(self, policy, device):
        scheduler = policy.noise_scheduler
        if type(scheduler).__name__ != "DDIMScheduler":
            raise RuntimeError(
                f"only DDIMScheduler is supported, got {type(scheduler).__name__}"
            )
        if scheduler.config.prediction_type != "epsilon":
            raise RuntimeError(
                f"only epsilon prediction is supported, got {scheduler.config.prediction_type}"
            )
        if getattr(scheduler.config, "thresholding", False):
            raise RuntimeError("dynamic thresholding is not supported")
        leftover_kwargs = getattr(policy, "kwargs", None)
        if leftover_kwargs:
            raise RuntimeError(
                f"policy passes extra scheduler kwargs {sorted(leftover_kwargs)}"
            )

        self.policy = policy
        self.device = torch.device(device)
        self.eager_conditional_sample = policy.conditional_sample
        self.num_steps = int(policy.num_inference_steps)

        scheduler.set_timesteps(self.num_steps)
        step_ratio = scheduler.config.num_train_timesteps // self.num_steps
        alphas = scheduler.alphas_cumprod
        self.timesteps = [int(t) for t in scheduler.timesteps]
        self.t_gpu = [
            torch.full((1,), t, dtype=torch.long, device=self.device)
            for t in self.timesteps
        ]
        # Per-step DDIM coefficients (eta=0): resident on GPU so the loop has
        # no CPU involvement and can be captured.
        self.coefs = []
        for t in self.timesteps:
            prev_t = t - step_ratio
            a_t = alphas[t]
            a_prev = alphas[prev_t] if prev_t >= 0 else scheduler.final_alpha_cumprod
            self.coefs.append(
                tuple(
                    c.to(device=self.device, dtype=torch.float32)
                    for c in (
                        a_t.sqrt(),
                        (1 - a_t).sqrt(),
                        a_prev.sqrt(),
                        (1 - a_prev).sqrt(),
                    )
                )
            )
        self.clip_sample = bool(scheduler.config.clip_sample)
        self.clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
        self.takes_global_cond = (
            "global_cond" in inspect.signature(policy.model.forward).parameters
        )

        self._hint_causal_decoder()

        self.graph = None
        self.static_traj = None
        self.static_cond = None
        self.graph_out = None

    def install(self):
        self.policy.conditional_sample = self._conditional_sample

    def uninstall(self):
        self.policy.__dict__.pop("conditional_sample", None)

    def _hint_causal_decoder(self):
        """Pass tgt_is_causal explicitly to nn.TransformerDecoder.

        Its mask auto-detection (`_detect_is_causal_mask`) runs a GPU->CPU sync
        on every forward, which both stalls the eager pipeline and aborts CUDA
        graph capture. The hint is only installed when the model mask really is
        the standard causal mask.
        """
        model = self.policy.model
        decoder = getattr(model, "decoder", None)
        mask = getattr(model, "mask", None)
        if decoder is None or mask is None:
            return
        if getattr(decoder, "_bspline_causal_hint", False):
            return
        # tgt_is_causal was added to TransformerDecoder.forward in torch 2.0.
        # On older torch (e.g. 1.12) the _detect_is_causal_mask CPU sync this
        # hint works around does not exist, so the optimization is a no-op.
        if "tgt_is_causal" not in inspect.signature(decoder.forward).parameters:
            return
        try:
            causal = torch.nn.Transformer.generate_square_subsequent_mask(
                mask.shape[0], device=mask.device, dtype=mask.dtype
            )
        except TypeError:
            # torch < 2.0 does not accept device/dtype kwargs
            causal = torch.nn.Transformer.generate_square_subsequent_mask(
                mask.shape[0]
            ).to(device=mask.device, dtype=mask.dtype)
        if mask.shape != causal.shape or not torch.equal(mask, causal):
            return
        orig_forward = decoder.forward

        def forward_with_hint(tgt, memory, tgt_mask=None, memory_mask=None, **kwargs):
            if tgt_mask is mask:
                kwargs.setdefault("tgt_is_causal", True)
            return orig_forward(
                tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, **kwargs
            )

        decoder.forward = forward_with_hint
        decoder._bspline_causal_hint = True

    def _model_eps(self, trajectory, t, cond):
        if self.takes_global_cond:
            return self.policy.model(trajectory, t, local_cond=None, global_cond=cond)
        return self.policy.model(trajectory, t, cond)

    def _denoise(self, trajectory, cond):
        for t, (sqrt_a_t, sqrt_1m_a_t, sqrt_a_prev, sqrt_1m_a_prev) in zip(
            self.t_gpu, self.coefs
        ):
            eps = self._model_eps(trajectory, t, cond)
            x0 = (trajectory - sqrt_1m_a_t * eps) / sqrt_a_t
            if self.clip_sample:
                x0 = x0.clamp(-self.clip_range, self.clip_range)
            trajectory = sqrt_a_prev * x0 + sqrt_1m_a_prev * eps
        return trajectory

    def _capture(self, condition_data, cond):
        self.static_traj = torch.empty_like(condition_data)
        self.static_cond = torch.empty_like(cond)
        self.static_traj.normal_()
        self.static_cond.copy_(cond)

        # Warmup on a side stream: cuDNN autotuning and allocator growth must
        # happen before capture, they are illegal inside a graph.
        side_stream = torch.cuda.Stream(self.device)
        side_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(side_stream):
            for _ in range(3):
                self._denoise(self.static_traj, self.static_cond)
        torch.cuda.current_stream(self.device).wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            self.graph_out = self._denoise(self.static_traj, self.static_cond)
        self.graph = graph
        print(
            f"Captured CUDA graph: {self.num_steps} DDIM steps, "
            f"trajectory {tuple(self.static_traj.shape)}, cond {tuple(self.static_cond.shape)}"
        )

    def _conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        cond=None,
        generator=None,
        **kwargs,
    ):
        cond_feat = global_cond if self.takes_global_cond else cond
        fast = (
            not kwargs
            and generator is None
            and local_cond is None
            and cond_feat is not None
            and condition_data.is_cuda
            and int(self.policy.num_inference_steps) == self.num_steps
            and not bool(condition_mask.any())
        )
        if fast and self.graph is not None:
            fast = (
                condition_data.shape == self.static_traj.shape
                and cond_feat.shape == self.static_cond.shape
            )
        if not fast:
            if self.takes_global_cond:
                return self.eager_conditional_sample(
                    condition_data,
                    condition_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    generator=generator,
                    **kwargs,
                )
            return self.eager_conditional_sample(
                condition_data,
                condition_mask,
                cond=cond,
                generator=generator,
                **kwargs,
            )

        if self.graph is None:
            self._capture(condition_data, cond_feat)

        self.static_cond.copy_(cond_feat)
        self.static_traj.normal_()  # fresh initial noise, drawn outside the graph
        self.graph.replay()
        return self.graph_out.clone()


class DiffusionBSplineModel:
    def __init__(
        self,
        ckpt_path,
        diffusion_policy_dir=None,
        degree=None,
        device="cuda",
        use_ema=None,
        rotation_output=None,
        num_inference_steps=None,
        use_cuda_graph=False,
    ):
        self.diffusion_policy_dir = _add_diffusion_policy_to_path(diffusion_policy_dir)
        from diffusion_policy.common.pytorch_util import dict_apply

        self.dict_apply = dict_apply
        with open(ckpt_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)

        policy = workspace.model
        if use_ema is None:
            use_ema = bool(_cfg_get(cfg.training, "use_ema", False))
        if use_ema and _cfg_get(cfg.training, "use_ema", False):
            policy = workspace.ema_model

        self.device = torch.device(device)
        policy.eval().to(self.device)

        if num_inference_steps is not None:
            policy.num_inference_steps = int(num_inference_steps)
        print(f"  num_inference_steps: {policy.num_inference_steps}")

        self.policy = policy
        self.cfg = cfg
        self.obs_shape_meta = cfg.shape_meta["obs"]
        self.degree = int(degree if degree is not None else _cfg_get(cfg.policy, "bspline_degree", 3))
        self.action_meta = infer_action_meta(
            cfg,
            self.degree,
            rotation_output_override=rotation_output,
        )
        self.inference_times = []

        print(f"Loaded local B-spline policy from: {ckpt_path}")
        print(f"  diffusion_policy_dir: {self.diffusion_policy_dir}")
        print(f"  device: {self.device}")
        print(f"  degree: {self.degree}")
        print(f"  action_meta: {self.action_meta}")

        self._warmup()

        self.cuda_graph_sampler = None
        if use_cuda_graph:
            self._install_cuda_graph()

    def _dummy_obs_dict(self):
        n_obs_steps = int(_cfg_get(self.cfg, "n_obs_steps", 2))
        dummy_obs = {}
        for key, meta in self.obs_shape_meta.items():
            shape = tuple(int(v) for v in meta["shape"])
            if meta.get("type") == "rgb":
                _, h, w = shape
                dummy_obs[key] = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                dummy_obs[key] = np.zeros(shape, dtype=np.float32)
        return self._convert_obs([dummy_obs] * n_obs_steps)

    def _warmup(self):
        """Run one dummy inference at load time so CUDA init never pollutes timings."""
        obs_dict = self._dummy_obs_dict()
        print("Warming up policy...")
        start = time.perf_counter()
        with torch.no_grad():
            self.policy.predict_action(obs_dict)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        print(f"Policy warmed up in {time.perf_counter() - start:.4f} seconds")

    def _install_cuda_graph(self):
        """Capture the DDIM loop as a CUDA graph"""
        if self.device.type != "cuda":
            print("CUDA graph requested but device is not cuda; keeping eager path")
            return
        try:
            sampler = CudaGraphDDIMSampler(self.policy, self.device)
        except RuntimeError as exc:
            print(f"CUDA graph disabled: {exc}")
            return

        obs_dict = self._dummy_obs_dict()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)
        try:
            capture_start = time.perf_counter()
            sampler.install()
            with torch.no_grad():
                self.policy.predict_action(obs_dict)  # triggers capture
                torch.cuda.synchronize(self.device)
                capture_time = time.perf_counter() - capture_start

                torch.manual_seed(0)
                graph_pred = self.policy.predict_action(obs_dict)["action_pred"].clone()

                sampler.uninstall()
                torch.manual_seed(0)
                eager_pred = self.policy.predict_action(obs_dict)["action_pred"]
                sampler.install()

                diff = float((graph_pred - eager_pred).abs().max().item())
            print(
                f"CUDA graph ready in {capture_time:.1f}s, "
                f"max |graph - eager| on warmup obs: {diff:.2e}"
            )
            # Matmul/conv kernels may differ between capture and eager streams
            # (cuDNN TF32 algorithm selection), which alone yields ~1e-3 in
            # normalized action space. Anything beyond that indicates a real
            # capture bug (e.g. mis-wired conditioning).
            if diff > 5e-3:
                raise RuntimeError(f"CUDA graph verification diff too large: {diff:.2e}")
        except Exception as exc:  # noqa: BLE001 - always fall back to eager
            sampler.uninstall()
            print(f"CUDA graph disabled: {type(exc).__name__}: {exc}")
            return
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)

        self.cuda_graph_sampler = sampler

    def reset(self):
        self.policy.reset()

    def print_inference_summary(self):
        times = np.asarray(self.inference_times, dtype=np.float64)
        if times.size == 0:
            print("Inference time summary: no inference calls recorded")
            return
        print(
            f"Inference time summary: n={times.size} "
            f"mean={times.mean():.4f}s min={times.min():.4f}s "
            f"max={times.max():.4f}s"
        )

    def predict_bspline(self, obs_sequence):
        obs_dict = self._convert_obs(obs_sequence)
        from bspline_policy.common.knots import decode_relative_knots

        with torch.no_grad():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            infer_start = time.perf_counter()
            result = self.policy.predict_action(obs_dict)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_time = time.perf_counter() - infer_start
            self.inference_times.append(inference_time)
            action = result["action"][0].detach().to("cpu").numpy()

        if self.action_meta["relative_knots"]:
            action = decode_relative_knots(action, degree=self.degree)
        expected_channels = self.action_meta["raw_bspline_dim"]
        if action.ndim != 2 or action.shape[1] != expected_channels:
            raise ValueError(
                f"Expected B-spline action shape (T, {expected_channels}), "
                f"got {action.shape}"
            )
        print(f"Model inference time: {inference_time:.4f} seconds")
        return action, dict(self.action_meta)

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
        return self.dict_apply(
            obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )


class PolicyLocalBSpline:
    def __init__(
        self,
        ckpt_path,
        diffusion_policy_dir=None,
        device="cuda",
        use_ema=None,
        rotation_output=None,
        num_inference_steps=None,
        use_cuda_graph=False,
        n_obs_steps=2,
        obs_stride=20,
        degree=None,
        speed_up_times=1.0,
        predict_before_end=0.06,
        origin_time_scale=10.0,
        use_action_derivatives=False,
        disable_time_align=False,
        time_align_error_threshold=0.1,
        time_align_larger_t=0.2,
        restart_on_time_align_error=False,
        consider_gripper_during_align=False,
        gripper_slowdown_enabled=False,
        gripper_slowdown_threshold=0.08,
        gripper_slowdown_steps=7,
    ):
        self.model = DiffusionBSplineModel(
            ckpt_path=ckpt_path,
            diffusion_policy_dir=diffusion_policy_dir,
            degree=degree,
            device=device,
            use_ema=use_ema,
            rotation_output=rotation_output,
            num_inference_steps=num_inference_steps,
            use_cuda_graph=use_cuda_graph,
        )
        self.n_obs_steps = int(n_obs_steps)
        self.obs_stride = max(1, int(obs_stride))
        self.obs_history = deque(maxlen=self.n_obs_steps * self.obs_stride)
        self.speed_up_times = float(speed_up_times)
        self.predict_before_end = float(predict_before_end)
        self.origin_time_scale = float(origin_time_scale)
        self.use_action_derivatives = bool(use_action_derivatives)
        self.degree = self.model.degree
        self.disable_time_align = bool(disable_time_align)
        self.time_align_error_threshold = float(time_align_error_threshold)
        self.time_align_larger_t = time_align_larger_t
        self.restart_on_time_align_error = bool(restart_on_time_align_error)
        self.consider_gripper_during_align = bool(consider_gripper_during_align)
        self.gripper_slowdown_enabled = bool(gripper_slowdown_enabled)
        self.gripper_slowdown_threshold = float(gripper_slowdown_threshold)
        self.gripper_slowdown_steps = int(gripper_slowdown_steps)

        self.predictor_lock = threading.Lock()
        self.req_queue = queue.Queue(maxsize=1)
        self.latest_obs_raw = None
        self.predictor = None
        self.vel_predictor = None
        self.acc_predictor = None
        self.min_t = None
        self.max_t = None
        self.last_obs_time_to_predict = None
        self.last_t_normalized = None
        self.action_meta = {}
        self.action_format = None
        self.getting_spline = False
        self._last_step_time = None
        self._accumulated_t = 0.0
        self._last_gripper_pos = None
        self._slowdown_remaining_steps = 0
        self._reset_epoch = 0

        for key, value in self.model.obs_shape_meta.items():
            if value.get("type") == "rgb":
                _, h, w = value["shape"]
                print(f"Policy expects {key}: {w}x{h}")

        threading.Thread(target=self._predict_process, daemon=True).start()

    def reset(self):
        with self.predictor_lock:
            self._reset_epoch += 1
            self.predictor = None
            self.vel_predictor = None
            self.acc_predictor = None
            self.min_t = None
            self.max_t = None
            self.last_obs_time_to_predict = None
            self.last_t_normalized = None
            self.action_meta = {}
            self.action_format = None
            self.getting_spline = False
            self._last_step_time = None
            self._accumulated_t = 0.0
            self._last_gripper_pos = None
            self._slowdown_remaining_steps = 0
        self.obs_history.clear()
        self.latest_obs_raw = None
        while not self.req_queue.empty():
            self.req_queue.get_nowait()
        self.model.reset()

    def step(self, obs):
        if isinstance(obs, (list, tuple)):
            self.latest_obs_raw = obs[-1]
            obs_sequence = [self._process_obs(single_obs) for single_obs in obs]
            if len(obs_sequence) < self.n_obs_steps:
                return None
        else:
            self.latest_obs_raw = obs
            self.obs_history.append(self._process_obs(obs))
            if len(self.obs_history) < self.n_obs_steps * self.obs_stride:
                return None
            obs_sequence = [
                self.obs_history[index]
                for index in range(self.obs_stride - 1, len(self.obs_history), self.obs_stride)
            ]

        self._request_spline_if_needed(obs_sequence)
        return self._sample_action()

    def print_inference_summary(self):
        self.model.print_inference_summary()

    def wait_for_pending_inference(self, timeout=3.0):
        """Block until no spline request is in flight.

        Call before process exit: tearing down the interpreter while the
        daemon predictor thread is inside a CUDA call aborts shutdown
        ("terminate called without an active exception", SIGABRT).
        """
        deadline = time.perf_counter() + float(timeout)
        while time.perf_counter() < deadline:
            with self.predictor_lock:
                if not self.getting_spline:
                    return True
            time.sleep(0.01)
        return False

    def waiting_for_first_plan(self):
        """True while the first spline request is still in flight (no plan installed yet)."""
        with self.predictor_lock:
            return self.predictor is None and self.getting_spline

    def poll_action(self):
        """Sample the current plan without feeding a new observation."""
        return self._sample_action()

    def discard_plan(self):
        with self.predictor_lock:
            self.predictor = None
            self.vel_predictor = None
            self.acc_predictor = None
            self.min_t = None
            self.max_t = None
            self.last_obs_time_to_predict = None
            self.last_t_normalized = None
            self.getting_spline = False
        while not self.req_queue.empty():
            self.req_queue.get_nowait()

    def _process_obs(self, obs):
        processed = {}
        for key, meta in self.model.obs_shape_meta.items():
            if key not in obs:
                if meta.get("type") == "rgb":
                    _, h, w = meta["shape"]
                    processed[key] = np.zeros((h, w, 3), dtype=np.uint8)
                    continue
                raise KeyError(f"Observation is missing required policy key: {key}")
            value = obs[key]
            if isinstance(value, dict):
                continue
            if meta.get("type") == "rgb":
                _, h, w = meta["shape"]
                if value is None:
                    value = np.zeros((h, w, 3), dtype=np.uint8)
                elif value.ndim == 3 and value.shape[:2] != (h, w):
                    value = cv.resize(value, (w, h))
                if value.dtype != np.uint8:
                    value = value.astype(np.uint8)
            processed[key] = value
        return processed

    def _request_spline_if_needed(self, obs_sequence):
        with self.predictor_lock:
            needs_spline = self.predictor is None
            if not needs_spline and self.last_obs_time_to_predict is not None:
                elapsed = time.perf_counter() - self.last_obs_time_to_predict
                time_remaining = (
                    self.max_t / self.origin_time_scale
                    - elapsed * self.speed_up_times
                )
                # time_remaining is in origin-trajectory seconds, which are
                # consumed at speed_up_times x wall-clock rate. Scale the
                # threshold so the wall-clock lead given to the predictor stays
                # constant (predict_before_end seconds) regardless of speed-up;
                # otherwise the plan runs out before inference finishes at high
                # speed-up and the motion stalls segment-by-segment.
                needs_spline = time_remaining < self.predict_before_end * self.speed_up_times
            elif not needs_spline:
                needs_spline = True

            if self.getting_spline or not needs_spline:
                return
            self.getting_spline = True
            reset_epoch = self._reset_epoch

        req = {
            "obs_sequence": [dict(obs) for obs in obs_sequence],
            "latest_obs": self.latest_obs_raw,
            "obs_time": time.perf_counter(),
            "reset_epoch": reset_epoch,
        }
        try:
            self.req_queue.put_nowait(req)
            print("get bspline")
        except queue.Full:
            with self.predictor_lock:
                self.getting_spline = False

    def _predict_process(self):
        while True:
            req = self.req_queue.get()
            try:
                start = time.perf_counter()
                bspline, meta = self.model.predict_bspline(req["obs_sequence"])
                elapsed = time.perf_counter() - start
                print(f"Time to get B-spline: {elapsed:.4f} seconds")
                self._install_bspline(bspline, meta, req)
            except Exception:
                import traceback

                traceback.print_exc()
                with self.predictor_lock:
                    self.getting_spline = False

    def _install_bspline(self, bspline, meta, req):
        with self.predictor_lock:
            if req["reset_epoch"] != self._reset_epoch:
                self.getting_spline = False
                return

            self.action_meta = meta or {}
            self.action_format = self.action_meta.get("action_format")
            if self.predictor is None or self.disable_time_align:
                self._flush_predictor(bspline)
                self.last_obs_time_to_predict = time.perf_counter()
                new_t_normalized = 0.0
                error = 0.0
            else:
                old_t = self.last_t_normalized
                old_action_raw = self.predictor(old_t)
                self._flush_predictor(bspline)
                new_t_normalized, error = self._align_new_plan(
                    old_action_raw,
                    req.get("obs_time", time.perf_counter()),
                )
                if (
                    self.restart_on_time_align_error
                    and error > self.time_align_error_threshold
                ):
                    print(
                        "Time-align error too large; restarting new B-spline "
                        f"at t=0.0: {error:.6f}"
                    )
                    new_t_normalized = 0.0
                self.last_obs_time_to_predict = (
                    -new_t_normalized
                    / self.speed_up_times
                    / self.origin_time_scale
                    + time.perf_counter()
                )

            if self.gripper_slowdown_enabled:
                self._accumulated_t = float(np.asarray(new_t_normalized).reshape(-1)[0])
                self._last_step_time = time.perf_counter()
            print(
                "New last obs time to predict: "
                f"{float(np.asarray(new_t_normalized).reshape(-1)[0]) / self.speed_up_times / self.origin_time_scale:.4f}s"
            )
            if error > self.time_align_error_threshold:
                print(f"Warning: B-spline time-align error too large: {error:.6f}")
            self.getting_spline = False

    def _flush_predictor(self, bspline_raw):
        knots = safer_knots(bspline_raw[..., 0])
        control_points = np.asarray(bspline_raw[..., 1:], dtype=np.float64)
        self.predictor = BSpline(
            t=knots,
            c=control_points[: -(self.degree + 1)],
            k=self.degree,
        )
        if self.predictor.c.ndim != 2:
            raise ValueError(f"Expected 2D B-spline control points, got {self.predictor.c.shape}")
        if self.use_action_derivatives:
            self.vel_predictor = self.predictor.derivative(1)
            self.acc_predictor = self.predictor.derivative(2) if self.degree >= 2 else None
        else:
            self.vel_predictor = None
            self.acc_predictor = None
        self.min_t, self.max_t = self.predictor.t[[self.degree, -self.degree - 1]]

    def _align_new_plan(self, old_action_raw, obs_time):
        new_max_t = (
            (time.perf_counter() - obs_time)
            * self.speed_up_times
            * self.origin_time_scale
        )
        new_max_t = float(np.clip(new_max_t, self.min_t, self.max_t))
        max_t_allowed = self.max_t - self.predict_before_end * self.origin_time_scale - 0.1
        if self.time_align_larger_t is not None:
            max_t_allowed = min(
                max_t_allowed,
                self.max_t * float(self.time_align_larger_t)
                + self.min_t * (1.0 - float(self.time_align_larger_t)),
            )
        max_t_allowed = max(max_t_allowed, self.min_t + 1e-3)

        lam = 1.0
        best_t = self.min_t
        best_error = math.inf
        while best_error > self.time_align_error_threshold:
            this_max_t = min(new_max_t * lam, max_t_allowed)
            if this_max_t <= self.min_t:
                break
            best_t, best_error = self._find_closest_t_to_target(
                self.predictor,
                old_action_raw,
                self.min_t,
                this_max_t,
            )
            if lam * new_max_t > max_t_allowed or lam > 20:
                break
            lam *= 1.5
        return best_t, best_error

    def _find_closest_t_to_target(self, predictor, target, min_t, max_t):
        if self.action_format == "single_yam_rot6d":
            compare_dim = 10 if self.consider_gripper_during_align else 9
        elif self.action_format in ("dual_arm_ee_rot6d", "dual_arm_ee_rot6d_next"):
            compare_dim = 20 if self.consider_gripper_during_align else 18
        else:
            compare_dim = 14 if self.consider_gripper_during_align else 12

        target = np.asarray(target).reshape(-1)

        def dist(t):
            current = predictor(t).squeeze()
            return np.sqrt((current[:compare_dim] - target[:compare_dim]) ** 2).sum()

        res = minimize_scalar(dist, bounds=(min_t, max_t), method="bounded")
        error = np.abs(predictor(res.x).squeeze()[:compare_dim] - target[:compare_dim])
        return res.x, float(error.max())

    def _sample_action(self):
        with self.predictor_lock:
            if self.predictor is None or self.last_obs_time_to_predict is None:
                return None

            current_time = time.perf_counter()
            if self.gripper_slowdown_enabled:
                t_normalized = np.array([self._step_gripper_slowdown_time(current_time)])
            else:
                t_normalized = (
                    np.array([current_time - self.last_obs_time_to_predict])
                    * self.speed_up_times
                    * self.origin_time_scale
                )

            if t_normalized < self.min_t:
                t_normalized = np.array([self.min_t])
            if t_normalized > self.max_t:
                print("Warning: all available action plan has been used")
                return None

            self.last_t_normalized = t_normalized
            action_raw = self.predictor(t_normalized).squeeze()
            if self.vel_predictor is None:
                action_velocity = None
            else:
                action_velocity = (
                    self.vel_predictor(t_normalized).squeeze()
                    / self.origin_time_scale
                    * self.speed_up_times
                )
            if self.acc_predictor is None:
                action_acceleration = None
            else:
                action_acceleration = (
                    self.acc_predictor(t_normalized).squeeze()
                    / (self.origin_time_scale ** 2)
                    * (self.speed_up_times ** 2)
                )

        return self._process_raw_action(action_raw, action_velocity, action_acceleration)

    def _step_gripper_slowdown_time(self, current_time):
        if self._last_step_time is None:
            self._last_step_time = current_time
            self._accumulated_t = 0.0
            return self._accumulated_t

        delta_real_time = current_time - self._last_step_time
        tentative_t = self._accumulated_t + delta_real_time * self.speed_up_times * self.origin_time_scale
        tentative_t = float(np.clip(tentative_t, self.min_t, self.max_t))
        tentative_action = self.predictor(np.array([tentative_t])).squeeze()
        if self.action_format == "single_yam_rot6d":
            gripper_idx = [9]
        elif self.action_format in ("dual_arm_ee_rot6d", "dual_arm_ee_rot6d_next"):
            gripper_idx = [18, 19]
        else:
            gripper_idx = [12, 13]
        current_gripper = tentative_action[gripper_idx]
        if self._last_gripper_pos is not None:
            gripper_change = np.abs(current_gripper - self._last_gripper_pos).max()
            if gripper_change > self.gripper_slowdown_threshold:
                self._slowdown_remaining_steps = self.gripper_slowdown_steps
        self._last_gripper_pos = current_gripper.copy()

        if self._slowdown_remaining_steps > 0:
            current_speedup = 1.0 + (
                self.gripper_slowdown_steps - self._slowdown_remaining_steps
            ) / max(self.gripper_slowdown_steps, 1) * (self.speed_up_times - 1.0)
            self._slowdown_remaining_steps -= 1
        else:
            current_speedup = self.speed_up_times
        self._accumulated_t += delta_real_time * current_speedup * self.origin_time_scale
        self._last_step_time = current_time
        return self._accumulated_t

    def _process_raw_action(self, action_raw, action_velocity=None, action_acceleration=None):
        del action_velocity, action_acceleration
        return decode_action_vector(action_raw, self.action_meta, self.latest_obs_raw)
