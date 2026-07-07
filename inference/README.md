## Policy Rollout



### Installation


We follow [Diffusion Policy](https://github.com/haoyu-x/diffusion_policy/tree/main?tab=readme-ov-file#%EF%B8%8F-installation) to set up the required dependencies for policy training. 
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 

```bash
# if you haven't install ``robodiff`` env

sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
cd ~/bspline-policy/diffusion_policy
mamba env create -f conda_environment.yaml
```

```bash
conda activate robodiff
cd ~/bspline-policy/real_env/i2rt
pip install -e .
```
```bash
conda activate robodiff
cd ~/bspline-policy/real_env/pyroki
pip install -e .
```



### YAM arms CAN setup

> [!NOTE]
> 1. Remeber to power on the robot arms first.
> 1. Follow [instructions](https://github.com/haoyu-x/bspline-policy/blob/main/real_env/i2rt/docs/getting-started/hardware-setup.md#persistent-can-ids) to set CAN name to can_follower_r and can_follower_l

Quick CAN connection:
```bash
sudo ip link set can_follower_r up type can bitrate 1000000
```


Test:
```bash
conda activate robodiff
cd ~/bspline-policy/real_env/i2rt
python i2rt/robots/motor_chain_robot.py --channel can_follower_r --gripper_type linear_4310
```




### Real-world Deployment:
> [!NOTE]
> 1. Update CAMERA_SERIAL in [constants.py](tidybot2/constants.py). (ignore if you did)
> 1. Download and open the XR Browser app on your iPhone.
> 1. Follow [XR Browser useage](https://tidybot2.github.io/docs/usage/#connecting-the-client)
> 1. We will use one iPhone to start, end episodes, and reset the robot during policy rollouts. Make sure the iPhone is connected to the same Wi-Fi network as the computer.



Open a new tab

```bash
conda activate robodiff
cd ~/bspline-policy/real_env/yam_teleop
python yam_server.py --channel can_follower_r 
```


open another tab, rollout:
```bash
conda activate robodiff
cd ~/bspline-policy
python real_env/yam_teleop/rollout_local_policy.py \
      --env yam \
      --policy bspline \
      --ckpt-path "your/ckpt/path" \
      --device cuda \
      --num-inference-steps 10 \
      --control-freq 200 \
      --data-freq 10 \
      --origin-time-scale 10 \
      --predict-before-end 0.1 \
      --speed-up-times 2.0 \
      --max-steps 3000 \
      --cuda-graph

```
