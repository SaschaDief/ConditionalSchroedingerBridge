#!/bin/bash
#SBATCH -A m3246
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus 4 

module load python 
conda activate SDiefenbacher_pytorch201

cd /global/homes/d/diefenbs/SBUnfold/BSBUnfold/

srun python sample.py --n-gpu-per-node 1  --sample-name 'OmniFold_train_large.h5'  --ckpt 'BSBUnfold_OmniFold_train_large_KL1fix_iter10M_6x256' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold' --n-ensemble 50 --n-ensemble-start 40 --n-eval 5309543 --hidden-dim 256 --nresnet 6

# python plot_ensemble.py --sample-name 'OmniFold_train_small.h5' --data-name 'BSBUnfold_OmniFold_train_small_KL01_iter10M_4x45' --directory '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold'