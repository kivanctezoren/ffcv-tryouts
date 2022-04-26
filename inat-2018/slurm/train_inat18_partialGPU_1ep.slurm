#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ffcv-train-inat18-partialGPU-1ep
#SBATCH --workdir=/home/ktezoren/ffcv/inat-2018/
#SBATCH --gres=gpu:2
#SBATCH --output=/home/ktezoren/ffcv/inat-2018/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/inat-2018/slurm/out/%x-%j.out  # Redirect err and out to the same file

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 train_inat18.py --config-file partial_gpu_config_1ep.yaml 

echo "--- FFCV Job (ffcv-train-inat18-partialGPU_1ep) Done ---"

