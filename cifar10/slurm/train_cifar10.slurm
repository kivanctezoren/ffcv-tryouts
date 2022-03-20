#!/bin/bash
#SBATCH --nodelist=dl3
#SBATCH --job-name=ffcv-cifar10-test
#SBATCH --workdir=/home/ktezoren/ffcv/cifar10/
#SBATCH --gres=gpu:1
#SBATCH --output=/home/ktezoren/ffcv/cifar10/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/cifar10/slurm/out/%x-%j.err

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 train_cifar.py --config-file default_config.yaml 

echo "--- FFCV Job (ffcv-cifar10-test) Done ---"

