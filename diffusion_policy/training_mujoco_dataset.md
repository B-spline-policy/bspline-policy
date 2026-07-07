## Policy training
  


### Data Processing

you can use your own collected data or directly download a dataset:
```bash
cd ~/simple-mobile/simple_mobile
source .venv/bin/activate

cd ~/simple-mobile/simple_mobile/tidybot2
uv run gdown https://drive.google.com/uc?id=10CEUUrz2BYksbkSrV_U2ZO08FVWEP2hS
unzip mujoco_cube_tidybot_vel_data.zip 
```


Before training a policy you need to first convert the data into a format compatible with the `diffusion_policy` codebase:


  ```bash
  cd ~/simple-mobile/simple_mobile
  source .venv/bin/activate
  cd ~/simple-mobile/simple_mobile/tidybot2
  uv run python convert_to_robomimic_mujoco_hdf5.py --input-dir data/succ --output-path data/sim-v1.hdf5

  ```

Copy the generated `.hdf5` file to the GPU machine for policy training.

Next, go to the GPU laptop, and follow the steps below to train a diffusion policy using the `sim-v1` data.

Move the generated `.hdf5` file to the `data` directory in the `diffusion_policy` repo:

  ```bash
  mkdir ~/simple-mobile/diffusion_policy/data
  mv ~/simple-mobile/simple_mobile/tidybot2/data/sim-v1.hdf5 ~/simple-mobile/diffusion_policy/data
  ```

### Training


Here, we follow [Diffusion Policy](https://github.com/haoyu-x/diffusion_policy/tree/main?tab=readme-ov-file#%EF%B8%8F-installation) to set up the required dependencies for policy training. 
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 

```bash
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
cd ~/simple-mobile/diffusion_policy
mamba env create -f conda_environment.yaml
```
```bash
# you can use conda as well: 
conda env create -f conda_environment.yaml
```

Open [`diffusion_policy/diffusion_policy/config/task/square_image_abs.yaml`](diffusion_policy/config/task/square_image_abs.yaml) and use the `sim-v1` part config for your task.

Open [`diffusion_policy/diffusion_policy/dataset/robomimic_replay_image_dataset.py`](diffusion_policy/dataset/robomimic_replay_image_dataset.py) and use the `# sim` part code for your `def _convert_actions`.


Start the training run:

  ```bash
  conda activate robodiff
  cd ~/simple-mobile/diffusion_policy
  python train.py --config-name=train_diffusion_unet_real_hybrid_workspace
  ```

### Next
Jump to [policy rollout in mujoco](../inference/rollout_mujoco.md).

