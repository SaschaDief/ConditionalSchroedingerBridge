module load python 
conda activate SDiefenbacher_pytorch201

python sample.py --n-gpu-per-node 1  --sample-name 'OmniFold_train_small.h5'  --ckpt 'BSBUnfold_OmniFold_train_small_KL1_iter10M' --batch-size 10000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold'

python sample.py --n-gpu-per-node 1  --sample-name 'OmniFold_train_small.h5'  --ckpt 'BSBUnfold_OmniFold_train_small_KL01_iter10M' --batch-size 10000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold'

python sample.py --n-gpu-per-node 1  --sample-name 'OmniFold_train_small.h5'  --ckpt 'BSBUnfold_OmniFold_train_small_KL001_iter10M' --batch-size 10000 --save-target '/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_results/BSBUnfold'