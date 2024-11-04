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

srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBUnfold_Z2Jet_KL1_iter5M_6x256_SelfCond' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 50 --n-ensemble-start 25 --n-eval 600000 --hidden-dim 256 --nresnet 6
