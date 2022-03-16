#!/bin/bash
#SBATCH --nodelist=dl3
#SBATCH --job-name=ffcv-train-imagenet-lt
#SBATCH --workdir=/home/ktezoren/ffcv/imagenet-lt/
#SBATCH --gres=gpu:4
#SBATCH --output=/home/ktezoren/ffcv/imagenet-lt/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/imagenet-lt/slurm/out/%x-%j.err

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 train_imagenet_lt.py --config-file default_config.yaml 

echo "--- FFCV Job (ffcv-train-imagenet-lt) Done ---"

