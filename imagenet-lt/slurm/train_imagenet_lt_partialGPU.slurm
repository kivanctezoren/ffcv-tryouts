#!/bin/bash
#SBATCH --nodelist=dl3
#SBATCH --job-name=ffcv-train-imagenet-lt-partialGPU
#SBATCH --workdir=/home/ktezoren/ffcv/imagenet-lt/
#SBATCH --gres=gpu:2
#SBATCH --output=/home/ktezoren/ffcv/imagenet-lt/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/imagenet-lt/slurm/out/%x-%j.err

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 train_imagenet_lt.py --config-file partial_gpu_config.yaml 

echo "--- FFCV Job (ffcv-train-imagenet-lt-partialGPU) Done ---"

