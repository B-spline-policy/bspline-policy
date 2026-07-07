# Clean Policy Data and Training

This README covers data conversion and training for DP and B-spline policies.
Haoyu-left is the main task. X5 stack-cube is included as a second example.

## 1. Prepare Environment

Run from the repo root:

```bash
REPO_ROOT=$PWD
export PYTHONPATH="$REPO_ROOT/bspline_policy:$REPO_ROOT/diffusion_policy:$REPO_ROOT/simple_mobile/tidybot2:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
```

## 2. Haoyu-left Data

Convert recorded episodes to robomimic HDF5:

```bash
cd "$REPO_ROOT/simple_mobile/tidybot2"
python convert_clean_bspline_policy_haoyu_left_to_robomimic_real_hdf5.py \
  --input-dir <EPISODE_DIR> \
  --output-path "$REPO_ROOT/diffusion_policy/data/clean_bspline_policy_haoyu_left.hdf5" \
  --overwrite
```

Check the converted file:

```bash
cd "$REPO_ROOT"
python - <<'PY'
import h5py
path = "diffusion_policy/data/clean_bspline_policy_haoyu_left.hdf5"
with h5py.File(path, "r") as f:
    demo = f["data"][sorted(f["data"].keys())[0]]
    print("actions:", demo["actions"].shape)
    print("obs keys:", sorted(demo["obs"].keys()))
PY
```

Expected action shape is `(T, 7)`. The dataset converts it to 10D:

```text
raw:    arm_pos_l(3), arm_rotvec_l(3), gripper_l(1)
policy: arm_xyz(3), arm_rot6d(6), gripper(1)
```

## 3. Train Haoyu-left DP

```bash
cd "$REPO_ROOT/diffusion_policy"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python train.py \
  --config-name=clean_bspline_policy_unet_dp \
  task=clean_bspline_policy_haoyu_left \
  hydra.run.dir=<OUTPUT_DIR> \
  training.resume=false \
  logging.mode=offline \
  task.dataset.cache_suffix=clean_bspline_policy_haoyu_left_action10_rot6d_v2
```

## 4. Train Haoyu-left B-spline

```bash
cd "$REPO_ROOT/bspline_policy"
export PYTHONPATH="$PWD:../diffusion_policy:${PYTHONPATH:-}"
python train.py \
  --config-name=clean_bspline_policy_unet_bspline \
  task=clean_bspline_policy_haoyu_left_bspline \
  task.dataset_path=../diffusion_policy/data/clean_bspline_policy_haoyu_left.hdf5 \
  task.dataset.dataset_path=../diffusion_policy/data/clean_bspline_policy_haoyu_left.hdf5 \
  hydra.run.dir=<OUTPUT_DIR> \
  training.resume=false \
  logging.mode=offline \
  task.dataset.cache_suffix=clean_bspline_policy_haoyu_left_action10_rot6d_v2
```

## 5. X5 Stack-cube Data Check

Use a robomimic HDF5 whose raw action shape is `(T, 14)`:

```text
left_pose6d(6), right_pose6d(6), left_gripper(1), right_gripper(1)
```

The dataset converts it to 20D:

```text
left_xyz,left_rot6d,right_xyz,right_rot6d,left_gripper,right_gripper
```

Check the file:

```bash
python - <<'PY'
import h5py
path = "<STACK_CUBE_HDF5>"
with h5py.File(path, "r") as f:
    demo = f["data"][sorted(f["data"].keys())[0]]
    print("actions:", demo["actions"].shape)
    print("obs keys:", sorted(demo["obs"].keys()))
PY
```

## 6. Train X5 Stack-cube DP

```bash
cd "$REPO_ROOT/diffusion_policy"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python train.py \
  --config-name=clean_bspline_policy_unet_dp \
  task=clean_bspline_policy_stack_cube_teleop_10hz_fix_cam \
  task.dataset_path=<STACK_CUBE_HDF5> \
  task.dataset.dataset_path=<STACK_CUBE_HDF5> \
  hydra.run.dir=<OUTPUT_DIR> \
  training.resume=false \
  logging.mode=offline \
  task.dataset.cache_suffix=clean_bspline_policy_stack_cube_action20_rot6d_v2
```

## 7. Train X5 Stack-cube B-spline

```bash
cd "$REPO_ROOT/bspline_policy"
export PYTHONPATH="$PWD:../diffusion_policy:${PYTHONPATH:-}"
python train.py \
  --config-name=clean_bspline_policy_unet_bspline \
  task=clean_bspline_policy_stack_cube_teleop_10hz_fix_cam_bspline \
  task.dataset_path=<STACK_CUBE_HDF5> \
  task.dataset.dataset_path=<STACK_CUBE_HDF5> \
  hydra.run.dir=<OUTPUT_DIR> \
  training.resume=false \
  logging.mode=offline \
  task.dataset.cache_suffix=clean_bspline_policy_stack_cube_action20_rot6d_v2
```

## Cache Note

After changing action conversion or dataset contents, do not reuse old cache.
Delete old cache files or use a new `cache_suffix`.

Cache files are created next to the HDF5:

```text
<DATASET_HDF5>.<cache_suffix>.zarr.zip
<DATASET_HDF5>.<cache_suffix>.zarr.zip.lock
```
