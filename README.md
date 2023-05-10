# AIL_from_visual_obs

## Instructions

### Use anaconda to create a virtual environment

Install [MuJoCo](https://github.com/deepmind/mujoco)

**Step 1.** install miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2.** clone repo and create conda environment

```shell
conda env create -f environment.yml
conda activate AdvIL_from_videos
```

### train expert

```shell
python train_expert.py task=walker_walk seed=0 agent=ddpg frame_skip=1
```
create a new directory `expert_policies`, move the trained expert policy in `expert_policies`.

Alternatively, download the policies [here](https://figshare.com/s/22de566de2229068fb75) and unzip in main directory.

### train imitation from experts

**DAC**
```shell
python train_w_expert_MDP.py task=walker_walk seed=0 GAN_loss=least-square from_dem=true
```
```shell
python train_w_expert_MDP.py task=walker_walk seed=0 GAN_loss=bce from_dem=true
```

**DACfO**
```shell
python train_w_expert_MDP.py task=walker_walk seed=0 GAN_loss=least-square from_dem=false
```
```shell
python train_w_expert_MDP.py task=walker_walk seed=0 GAN_loss=bce from_dem=false
```

**DRAIL**
```shell
python train_DRAIL.py task=walker_walk seed=0 GAN_loss=least-square from_dem=true
```
```shell
python train_DRAIL.py task=walker_walk seed=0 GAN_loss=bce from_dem=true
```

**DRAIfO**
```shell
python train_DRAIL.py task=walker_walk seed=0 GAN_loss=least-square from_dem=false
```
```shell
python train_DRAIL.py task=walker_walk seed=0 GAN_loss=bce from_dem=false
```

**VAIL**
```shell
python train_VAIL.py task=walker_walk seed=0 GAN_loss=least-square from_dem=true
```
```shell
python train_VAIL.py task=walker_walk seed=0 GAN_loss=bce from_dem=true
```

**VAIfO**
```shell
python train_VAIL.py task=walker_walk seed=0 GAN_loss=least-square from_dem=false
```
```shell
python train_VAIL.py task=walker_walk seed=0 GAN_loss=bce from_dem=false
```

**VMAIL**
```shell
python train_VMAIL.py task=walker_walk seed=0 GAN_loss=bce from_dem=true
```
