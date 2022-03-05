#!/bin/bash
#SBATCH --nodelist=dl3
#SBATCH --job-name=ffcv-inat18-test
#SBATCH --workdir=/home/ktezoren/ffcv/inat-2018/
#SBATCH --gres=gpu:4
#SBATCH --output=/home/ktezoren/ffcv/inat-2018/slurm/out/%x-%j.out
#SBATCH --error=/home/ktezoren/ffcv/inat-2018/slurm/out/%x-%j.err

#export PATH=/usr/local/cuda-10.0/bin:$PATH
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64

source activate ffcv
python3 train_inat18.py --config-file default_config.yaml 

echo "--- FFCV Job (ffcv-inat18-test) Done ---"

