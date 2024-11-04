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

srun python train_physics.py --corrupt 'BSBUnfold_Z2Jet_KL1_iter5M_4x3x512_SelfCond' --n-gpu-per-node 1 --beta-max 0.1 --beta-min 1e-6 --sample-name 'Z_2j' --num-itr 5000000 --kl 1.0 --hidden-dim 512 --nresnet 4 --layerresnet 3 --cond-x1 --lr 1e-3

srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBUnfold_Z2Jet_KL1_iter5M_6x256_SelfCond' --batch-size 10000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 1  --n-ensemble-start 0  --n-eval 600000 --hidden-dim 256 --nresnet 6 --bnn-map
#srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBUnfold_Z2Jet_KL1_iter1M_5x128_SelfCond' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 2  --n-ensemble-start 0 --n-eval 600000 --hidden-dim 128 --nresnet 5
srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBUnfold_Z2Jet_KL1_iter5M_6x256_SelfCond' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 5  --n-ensemble-start 0 --n-eval 600000 --hidden-dim 256 --nresnet 6
srun python sample.py --n-gpu-per-node 1  --sample-name 'Z_2j'  --ckpt 'BSBUnfold_Z2Jet_KL1_iter5M_6x256_SelfCond' --batch-size 50000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet' --n-ensemble 10 --n-ensemble-start 5--n-eval 600000 --hidden-dim 256 --nresnet 6

#python plot_ensembleZ2Jet.py --sample-name 'Z_2j' --data-name 'BSBUnfold_Z2Jet_KL1_iter5M_6x256_SelfCond' --directory '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold/Z_2Jet'