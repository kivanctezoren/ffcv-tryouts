#!/bin/bash
#SBATCH --nodelist=dl3
#SBATCH --job-name=ffcv-train-imagenet-givenconfig
#SBATCH --workdir=/home/ktezoren/ffcv/imagenet-lt/
#SBATCH --gres=gpu:4
#SBATCH --output=/home/ktezoren/ffcv/imagenet/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/imagenet/slurm/out/%x-%j.err

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 train_imagenet.py --config-file rn50_configs/rn50_88_epochs.yaml 

echo "--- FFCV Job (ffcv-train-imagenet-givenconfig) Done ---"

