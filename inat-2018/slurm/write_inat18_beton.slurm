#!/bin/bash
#SBATCH --nodelist=dl3
#SBATCH --job-name=write-inat18-beton
#SBATCH --workdir=/home/ktezoren/ffcv/inat-2018/
#SBATCH --gres=gpu:0
#SBATCH --output=/home/ktezoren/ffcv/inat-2018/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/inat-2018/slurm/out/%x-%j.err

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 write-inat18-beton.py 

echo "--- FFCV Job (write-inat18-beton) Done ---"

