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

srun python train_physics.py --corrupt 'BSBU_Z2Jet_KL01_i500k_Bmin1e7_lr1e2_4x3x512_SC' --n-gpu-per-node 1 --beta-max 0.1 --beta-min 1e-7 --sample-name 'Z_2j' --num-itr 500000 --kl 0.1 --cond-x1 --lr 1e-2 --hidden-dim 512 --nresnet 4 --layerresnet 3

srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBU_Z2Jet_KL01_i500k_Bmin1e7_lr1e2_4x3x512_SC' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 1 --n-ensemble-start 0 --n-eval 600000 --hidden-dim 512 --nresnet 4 --layerresnet 3 --bnn-map
srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBU_Z2Jet_KL01_i500k_Bmin1e7_lr1e2_4x3x512_SC' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 2 --n-ensemble-start 0 --n-eval 600000 --hidden-dim 512 --nresnet 4 --layerresnet 3



#srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBU_Z2Jet_KL01_i5M_Bmin1e6_lr1e4_4x3x512_SC' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 5  --n-ensemble-start 0 --n-eval 600000 --hidden-dim 512 --nresnet 4 --layerresnet 3
#srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBU_Z2Jet_KL01_i5M_Bmin1e6_lr1e4_4x3x512_SC' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 10 --n-ensemble-start 5 --n-eval 600000 --hidden-dim 512 --nresnet 4 --layerresnet 3

#python plot_ensembleZ2Jet.py --sample-name 'Z_2j' --data-name 'BSBU_Z2Jet_KL01_i5M_Bmin1e6_lr1e4_4x3x512_SC' --directory '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet'