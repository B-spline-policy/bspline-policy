## Policy training
  


### Data Processing

you can use your own collected data or directly download a dataset:
```bash
cd ~/simple_mobile_bsp/simple_mobile
source .venv/bin/activate

# download an example dataset
cd ~/simple_mobile_bsp/simple_mobile/yam_teleop
uv run gdown "https://drive.google.com/uc?id=1hGzU4CBUUrWGT-pSi77bfD1CMEutuUpa"
unzip yam_demo.zip 
```


Before training a policy you need to first convert the data into a format compatible with the `diffusion_policy` codebase:


  ```bash
  cd ~/simple_mobile_bsp/simple_mobile
  source .venv/bin/activate
  cd ~/simple_mobile_bsp/simple_mobile/yam_teleop
  uv run python convert_to_robomimic_hdf5.py --input-dir demos --output-path demos/yam-v1.hdf5

  ```


Copy the generated `.hdf5` file to the GPU machine for policy training.

Next, go to the GPU laptop, and follow the steps below to train a diffusion policy using the `real-v1` data.

Move the generated `.hdf5` file to the `data` directory in the `diffusion_policy` repo:

  ```bash
  mkdir ~/simple_mobile_bsp/diffusion_policy/data
  mv ~/simple_mobile_bsp/simple_mobile/yam_teleop/demos/yam-v1.hdf5 ~/simple_mobile_bsp/diffusion_policy/data
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



### Start training a diffusion policy:



We updated [`diffusion_policy/diffusion_policy/config/task/square_image_abs.yaml`](diffusion_policy/config/task/square_image_abs.yaml) and used the `sim-v1` part config for the task.

We updated [`diffusion_policy/diffusion_policy/dataset/robomimic_replay_image_dataset.py`](diffusion_policy/dataset/robomimic_replay_image_dataset.py) code for `def _convert_actions`.


> [!NOTE]
> 1. Remeber to name your dataset `yam-v1.hdf5` under `~/simple_mobile_bsp/diffusion_policy/data`

Open a new tab, start the training run:

  ```bash
  conda activate robodiff
  cd ~/simple_mobile_bsp/diffusion_policy
  python train.py --config-name=train_diffusion_unet_real_hybrid_workspace
  ```


Next --> [Robot Deplyment and Model Inference](../inference/README.md)
