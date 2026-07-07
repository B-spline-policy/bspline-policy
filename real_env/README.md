
## Installation


```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
cd ~/bspline-policy/real_env
sudo apt install -y cmake build-essential
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo apt-get install ffmpeg
```
```bash
cd ~/bspline-policy/real_env
uv sync
source .venv/bin/activate
```

```bash
cd ~/bspline-policy/real_env/i2rt
uv pip install -e .

cd ~/bspline-policy/real_env/pyroki
uv pip install -e .

```

## Teleop and data collection with a singel YAM:

### YAM arms CAN setup

> [!NOTE]
> 1. Remeber to power on the robot arms first.
> 1. Follow [instructions](https://github.com/haoyu-x/simple_mobile_bsp/blob/main/simple_mobile/i2rt/docs/getting-started/hardware-setup.md#persistent-can-ids) to set CAN name to can_follower_r

Quick CAN connection:
```bash
sudo ip link set can_follower_r up type can bitrate 1000000
```


Test:
```bash
cd ~/bspline-policy/real_env/i2rt
uv run python i2rt/robots/motor_chain_robot.py --channel can_follower_r --gripper_type linear_4310
```





### Real-world Teleop:


> [!NOTE]
> 1. Update CAMERA_SERIAL in [constants.py](tidybot2/constants.py).
> 1. Make sure the onboard computer and the two iPhones are under the same wifi network.
> 2. Download and open the XR Browser app on your two iPhones.
> 1. Follow [XR Browser useage](https://tidybot2.github.io/docs/usage/#connecting-the-client)
> 1. The first connected iphone will control the left arm and the base xy; the second connected iphone will control the right arm and the base yaw.
> 1. Place iPhone pose before pressing start episode:
(For proper coordinate frame alignment, the phone should face the same direction as the robot when you press "Start episode".)




```bash
cd ~/bspline-policy/real_env
source .venv/bin/activate
cd ~/bspline-policy/real_env/yam_teleop
uv run python yam_server.py  --channel can_follower_r
```

open another tab:

```bash
cd ~/bspline-policy/real_env
source .venv/bin/activate
cd ~/bspline-policy/real_env/yam_teleop
uv run python main.py --teleop --save
```

> [!NOTE]
> 1. If you meet this error:  
   `[2026-04-05 01:43:20.437] [host] [error] Searched, but no actual device found by given DeviceInfo: DeviceInfo(name=, deviceId=19443010B128714800, X_LINK_ANY_STATE, X_LINK_ANY_PROTOCOL, X_LINK_ANY_PLATFORM, X_LINK_SUCCESS) Check your USB connection.`
Check your USB connection.


Review collected data:
```bash
cd ~/bspline-policy/real_env
source .venv/bin/activate
cd ~/bspline-policy/real_env/yam_teleop
uv run python reviewer.py
```


Sort collected data:
```bash
cd ~/bspline-policy/real_env
source .venv/bin/activate
cd ~/bspline-policy/real_env/yam_teleop
uv run python sort_demos_from_review.py  # data/demos/review_results_20260407_171358.json
```


Replay BSP data:
```bash
cd ~/bspline-policy
source .venv/bin/activate
PYTHONPATH="$PWD/bspline-policy:$PWD/diffusion_policy:${PYTHONPATH:-}" \
python -m bspline_policy.scripts.yam_replay_episodes_bspline \
  --input-dir simple_mobile/yam_teleop/data/demos \
  --speed-up-times 4

```
Next --> [Bspline policy training](../bspline_policy)
