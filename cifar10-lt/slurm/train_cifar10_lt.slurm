#!/bin/bash
#SBATCH --nodelist=dl3
#SBATCH --job-name=ffcv-cifar10-lt
#SBATCH --workdir=/home/ktezoren/ffcv/cifar10-lt/
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ktezoren/ffcv/cifar10-lt/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/cifar10-lt/slurm/out/%x-%j.err

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 train_cifar10_lt.py --config-file default_config.yaml 

echo "--- FFCV Job (ffcv-cifar10-lt) Done ---"

