#!/bin/bash
#SBATCH -A m3246
#SBATCH --qos=regular
#SBATCH --time=11:00:00
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --gpus 4 

module load python 
conda activate SDiefenbacher_pytorch201

#/global/cfs/cdirs/m3246/sdiefenbacher/conda/envs/SDiefenbacher_pytorch201

cd /global/homes/d/diefenbs/SBUnfold/BSBUnfold/

srun python train_physics.py --corrupt 'BSBUnfold_OmniFold_train_large_KL100fix_iter10M_6x256' --n-gpu-per-node 1  --beta-max 0.1 --sample-name 'OmniFold_train_large.h5' --num-itr 1000000 --kl 100.0 --hidden-dim 256 --nresnet 6

srun python sample.py --n-gpu-per-node 1  --sample-name 'OmniFold_train_large.h5'  --ckpt 'BSBUnfold_OmniFold_train_large_KL100fix_iter10M_6x256' --batch-size 10000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold' --n-ensemble 1 --n-ensemble-start 0 --n-eval 5309543 --hidden-dim 256 --nresnet 6 --bnn-map


#srun python sample.py --n-gpu-per-node 1  --sample-name 'OmniFold_train_small.h5'  --ckpt 'BSBUnfold_OmniFold_train_small_KL01_iter10M' --batch-size 10000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold'

# python plot_ensemble.py --sample-name 'OmniFold_train_small.h5' --data-name 'BSBUnfold_OmniFold_train_small_KL01_iter10M' --directory '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold'