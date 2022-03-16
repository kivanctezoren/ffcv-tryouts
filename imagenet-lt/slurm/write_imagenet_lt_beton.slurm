#!/bin/bash
#SBATCH --nodelist=dl3
#SBATCH --job-name=write-imagenet-lt-beton
#SBATCH --workdir=/home/ktezoren/ffcv/imagenet-lt/
#SBATCH --gres=gpu:0
#SBATCH --output=/home/ktezoren/ffcv/imagenet-lt/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/imagenet-lt/slurm/out/%x-%j.err

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 write-imagenet-lt-beton.py 

echo "--- FFCV Job (write-imagenet-lt-beton) Done ---"

