#!/bin/bash
#SBATCH -n 32
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH -t 47:00:00
#SBATCH --mail-user=mabdel03@mit.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/6.8611_DeepLearning

python -u Train_Model_OG.py


