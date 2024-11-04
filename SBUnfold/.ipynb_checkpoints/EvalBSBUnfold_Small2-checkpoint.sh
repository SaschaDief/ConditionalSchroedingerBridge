#!/bin/bash
#SBATCH -A m3246
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus 4 

module load python 
conda activate SDiefenbacher_pytorch201

#/global/cfs/cdirs/m3246/sdiefenbacher/conda/envs/SDiefenbacher_pytorch201

cd /global/homes/d/diefenbs/SBUnfold/BSBUnfold/

srun python sample.py --n-gpu-per-node 1  --sample-name 'OmniFold_train_small.h5'  --ckpt 'BSBUnfold_OmniFold_train_small_KL001fix_iter10M_4x45_SelfCond' --batch-size 100000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold' --n-ensemble 25 --n-ensemble-start 0 --n-eval 5309543 --hidden-dim 45 --nresnet 4

# python plot_ensemble.py --sample-name 'OmniFold_train_small.h5' --data-name 'BSBUnfold_OmniFold_train_small_KL01_iter10M_4x45' --directory '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold'