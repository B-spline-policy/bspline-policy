"""CUDA graph DDIM vs eager DDIM numerical-equivalence test on real dataset obs.

Loads observation windows from the YAM demo hdf5, runs the same checkpoint
through the eager `conditional_sample` and the CUDA graph replay path with
identical RNG seeds, and asserts the predicted B-spline chunks match.

Run standalone (no pytest needed):

    conda run -n bsp-simple python bspline_policy/tests/test_cuda_graph_ddim.py

or via pytest if installed. Requires a CUDA device; each sub-test is skipped
when its checkpoint or the dataset is missing.

Env overrides: BSP_TEST_DATA, BSP_TEST_CKPT_TRANSFORMER, BSP_TEST_CKPT_UNET.
"""

import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BSPLINE_POLICY_DIR = TESTS_DIR.parent
REPO_ROOT = BSPLINE_POLICY_DIR.parent
for path in (BSPLINE_POLICY_DIR, REPO_ROOT / "diffusion_policy"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import h5py
import numpy as np
import torch

DATA_PATH = Path(
    os.environ.get(
        "BSP_TEST_DATA", REPO_ROOT / "diffusion_policy" / "data" / "yam-demo-0627.hdf5"
    )
)
CKPT_PATHS = {
    "transformer": Path(
        os.environ.get(
            "BSP_TEST_CKPT_TRANSFORMER",
            REPO_ROOT
            / "bspline_policy/outputs/2026-07-05/18-12-04/checkpoints/epoch=1000-train_loss=0.017.ckpt",
        )
    ),
    "unet": Path(
        os.environ.get(
            "BSP_TEST_CKPT_UNET",
            REPO_ROOT
            / "bspline_policy/data/outputs/2026.06.23/14.56.01_train_diffusion_unet_hybrid_bspline_yam-v1-bspline/checkpoints/latest.ckpt",
        )
    ),
}
N_WINDOWS = 4
N_OBS_STEPS = 2
NUM_INFERENCE_STEPS = 10
# With cuDNN TF32 disabled the only difference between the two paths is
# float32 rounding in the precomputed DDIM coefficients (measured ~1e-6).
ATOL = 1e-4


class SkipTest(Exception):
    pass


def _require(condition, reason):
    if not condition:
        raise SkipTest(reason)


def load_obs_windows(data_path, n_windows=N_WINDOWS, n_obs_steps=N_OBS_STEPS, seed=0):
    """Sample (demo, t) windows spread across demos; returns policy obs dicts."""
    windows = []
    with h5py.File(data_path, "r") as f:
        demos = sorted(f["data"].keys())
        rng = np.random.RandomState(seed)
        demo_picks = rng.choice(len(demos), size=n_windows, replace=False)
        for demo_idx in demo_picks:
            demo = f["data"][demos[demo_idx]]["obs"]
            length = demo["arm_pos"].shape[0]
            t0 = int(rng.randint(0, max(1, length - n_obs_steps)))
            obs_seq = []
            for i in range(n_obs_steps):
                t = min(t0 + i, length - 1)
                obs_seq.append(
                    {
                        "arm_pos": np.asarray(demo["arm_pos"][t], dtype=np.float32),
                        "arm_quat": np.asarray(demo["arm_quat"][t], dtype=np.float32),
                        "gripper_pos": np.asarray(demo["gripper_pos"][t], dtype=np.float32),
                        "wrist_image": np.asarray(demo["wrist_image"][t]),  # HWC uint8
                    }
                )
            windows.append(obs_seq)
    return windows


def _predict(policy, obs_dict, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        pred = policy.predict_action(obs_dict)["action_pred"]
    torch.cuda.synchronize()
    return pred.clone()


def run_equivalence_check(ckpt_name):
    _require(torch.cuda.is_available(), "CUDA device required")
    ckpt_path = CKPT_PATHS[ckpt_name]
    _require(ckpt_path.exists(), f"checkpoint not found: {ckpt_path}")
    _require(DATA_PATH.exists(), f"dataset not found: {DATA_PATH}")

    # cuDNN TF32 lets capture and eager streams pick different conv kernels,
    # which adds ~1e-3 noise unrelated to the graph mechanism; disable for a
    # strict comparison.
    torch.backends.cudnn.allow_tf32 = False

    from bspline_policy.scripts.policy_local_bspline import DiffusionBSplineModel

    model = DiffusionBSplineModel(
        ckpt_path=str(ckpt_path),
        num_inference_steps=NUM_INFERENCE_STEPS,
        use_cuda_graph=True,
    )
    try:
        assert model.cuda_graph_sampler is not None, (
            "CUDA graph sampler failed to install (see 'CUDA graph disabled' log)"
        )
        sampler = model.cuda_graph_sampler
        policy = model.policy
        windows = load_obs_windows(DATA_PATH)

        graph_preds = []
        for idx, obs_seq in enumerate(windows):
            obs_dict = model._convert_obs(obs_seq)
            seed = 1000 + idx

            sampler.uninstall()
            eager_pred = _predict(policy, obs_dict, seed)
            sampler.install()
            graph_pred = _predict(policy, obs_dict, seed)
            graph_pred_again = _predict(policy, obs_dict, seed)

            diff = float((eager_pred - graph_pred).abs().max().item())
            replay_diff = float((graph_pred - graph_pred_again).abs().max().item())
            print(
                f"[{ckpt_name}] window {idx}: max|eager - graph| = {diff:.3e}, "
                f"max|replay - replay| = {replay_diff:.3e}"
            )
            assert diff <= ATOL, (
                f"window {idx}: eager vs graph diff {diff:.3e} exceeds atol {ATOL:.0e}"
            )
            assert replay_diff == 0.0, (
                f"window {idx}: graph replay is not deterministic ({replay_diff:.3e})"
            )
            graph_preds.append(graph_pred)

            # Different seed must give a different sample (graph is not frozen).
            other_seed_pred = _predict(policy, obs_dict, seed + 7777)
            seed_diff = float((graph_pred - other_seed_pred).abs().max().item())
            assert seed_diff > 1e-3, (
                f"window {idx}: identical output across seeds ({seed_diff:.3e}); "
                "initial noise is not being refreshed"
            )

        # Different obs with the same seed must give different outputs
        # (conditioning really is copied into the graph's static buffer).
        cond_diff = float((graph_preds[0] - graph_preds[1]).abs().max().item())
        assert cond_diff > 1e-3, (
            f"identical output across different obs ({cond_diff:.3e}); "
            "cond static buffer is not being updated"
        )
    finally:
        del model
        torch.cuda.empty_cache()


def test_cuda_graph_matches_eager_transformer():
    run_equivalence_check("transformer")


def test_cuda_graph_matches_eager_unet():
    run_equivalence_check("unet")


def main():
    tests = [
        ("transformer", test_cuda_graph_matches_eager_transformer),
        ("unet", test_cuda_graph_matches_eager_unet),
    ]
    failures = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            print(f"[{name}] PASS")
        except SkipTest as exc:
            print(f"[{name}] SKIP: {exc}")
        except AssertionError as exc:
            print(f"[{name}] FAIL: {exc}")
            failures.append(name)
    if failures:
        sys.exit(f"FAILED: {failures}")
    print("\nOK")


if __name__ == "__main__":
    main()
