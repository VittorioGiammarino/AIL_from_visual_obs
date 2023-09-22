# AIL_from_visual_obs

## Instructions

### Use anaconda to create a virtual environment

**Step 1.** Install miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2.** Install [MuJoCo](https://github.com/deepmind/mujoco)

**Step 3.** Clone repo and create conda environment

```shell
conda env create -f environment.yml
conda activate AdvIL_from_videos
```

### Train expert

```shell
python train_expert.py task=walker_walk seed=0 agent=ddpg frame_skip=1
```
Create a new directory `expert_policies`, move the trained expert policy in `expert_policies`.

Alternatively, download the policies [here](https://figshare.com/s/22de566de2229068fb75) and unzip in main directory.

### Train imitation from experts

**DAC**
```shell
python train_w_expert_MDP.py task=walker_walk seed=0 GAN_loss=bce from_dem=true
```

**DACfO**
```shell
python train_w_expert_MDP.py task=walker_walk seed=0 GAN_loss=bce from_dem=false
```

**patchAIL**
```shell
python train_LAIL.py task=walker_walk agent=patchAIL seed=0 GAN_loss=bce discriminator_lr=1e-4
```

**LAIfO**
```shell
python train_LAIL.py task=walker_walk agent=lail seed=0 GAN_loss=bce from_dem=false
```

**VMAIL**
```shell
python train_VMAIL.py task=walker_walk seed=0 GAN_loss=bce from_dem=true
```

**LAIL**
```shell
python train_LAIL.py task=walker_walk agent=lail seed=0 GAN_loss=bce from_dem=true
```

