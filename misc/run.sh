#! /bin/bash

#SBATCH --job-name=Ksz
#SBATCH --partition=sxmq
#SBATCH --nodelist=sxm001

source /home/farhadik/miniforge3/bin/activate
conda activate torch

python3 cpt_sal.py