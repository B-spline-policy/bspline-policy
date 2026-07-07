# Clean Policy Rollout

This README covers local rollout for DP and B-spline checkpoints.

## 1. Prepare Environment

Run from the repo root:

```bash
REPO_ROOT=$PWD
export PYTHONPATH="$REPO_ROOT/bspline_policy:$REPO_ROOT/diffusion_policy:$REPO_ROOT/simple_mobile/tidybot2:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
```

## 2. Download Shared B-spline Rollout Data

Browser download:

```text
https://drive.google.com/file/d/1KGDvhKZQPkr7u83hPs8O3GIFpXFZGV5g/view?usp=sharing
```

Command-line download:

```bash
python -m pip install gdown
gdown --id 1KGDvhKZQPkr7u83hPs8O3GIFpXFZGV5g -O <ROLLOUT_DATA_FILE>
```

If the file is an archive, unpack it and use the checkpoint path inside as
`<CKPT_PATH>`.

## 3. Haoyu-left DP Rollout

Use this for checkpoints trained with `task=clean_bspline_policy_haoyu_left`.

```bash
python "$REPO_ROOT/simple_mobile/tidybot2/rollout_local_policy.py" \
  --env tidybot2 \
  --policy dp \
  --ckpt-path <CKPT_PATH> \
  --diffusion-policy-dir "$REPO_ROOT/diffusion_policy" \
  --control-freq 200 \
  --data-freq 10 \
  --save \
  --output-dir <ROLLOUT_OUTPUT_DIR>
```

## 4. Haoyu-left B-spline Rollout

Use this for checkpoints trained with `task=clean_bspline_policy_haoyu_left_bspline`.

```bash
python "$REPO_ROOT/simple_mobile/tidybot2/rollout_local_policy.py" \
  --env tidybot2 \
  --policy bspline \
  --ckpt-path <CKPT_PATH> \
  --diffusion-policy-dir "$REPO_ROOT/diffusion_policy" \
  --control-freq 200 \
  --data-freq 10 \
  --origin-time-scale 10 \
  --predict-before-end 0.06 \
  --save \
  --output-dir <ROLLOUT_OUTPUT_DIR> \
  --speed-up-times 1.0
```

## 5. X5 Stack-cube DP Rollout

Use this for checkpoints trained with `task=clean_bspline_policy_stack_cube_teleop_10hz_fix_cam`.

```bash
python "$REPO_ROOT/simple_mobile/tidybot2/rollout_local_policy.py" \
  --env x5 \
  --policy dp \
  --ckpt-path <CKPT_PATH> \
  --diffusion-policy-dir "$REPO_ROOT/diffusion_policy" \
  --control-freq 200 \
  --data-freq 10 \
  --save \
  --output-dir <ROLLOUT_OUTPUT_DIR>
```

## 6. X5 Stack-cube B-spline Rollout

Use this for checkpoints trained with `task=clean_bspline_policy_stack_cube_teleop_10hz_fix_cam_bspline`.

```bash
python "$REPO_ROOT/simple_mobile/tidybot2/rollout_local_policy.py" \
  --env x5 \
  --policy bspline \
  --ckpt-path <CKPT_PATH> \
  --diffusion-policy-dir "$REPO_ROOT/diffusion_policy" \
  --control-freq 200 \
  --data-freq 10 \
  --origin-time-scale 10 \
  --predict-before-end 0.06 \
  --save \
  --output-dir <ROLLOUT_OUTPUT_DIR>
```

## Quick Checks

Before rollout, confirm:

- The checkpoint was trained after the action-conversion fix.
- Haoyu-left checkpoint action shape is `[10]`.
- Stack-cube checkpoint action shape is `[20]`.
- The selected `--env` matches the robot and checkpoint task.
