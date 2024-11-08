# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
import runner_physics
from torch.utils.data import TensorDataset, DataLoader, Dataset

import colored_traceback.always
from ipdb import set_trace as debug

sys.path.append('../')
import utils

import energyflow as ef

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--name",           type=str,   default=None,        help="experiment ID")
    parser.add_argument("--corrupt",        type=str,   default=None,        help="restoration task")
    parser.add_argument("--gpu",            type=int,   default=None,        help="set only if you wish to run on a particular device")
    parser.add_argument("--ckpt",           type=str,   default=None,        help="resumed checkpoint name")
    parser.add_argument("--sample-name",      type=str,  default="Pythia26",  help="Load data sample")
        
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,           help="The number of nodes in multi node env")
    # parser.add_argument("--amp",            action="store_true")

    # --------------- SB model ---------------
    parser.add_argument("--t0",             type=float, default=1e-4,        help="sigma start time in network parametrization")
    parser.add_argument("--T",              type=float, default=1.,          help="sigma end time in network parametrization")
    parser.add_argument("--interval",       type=int,   default=1000,        help="number of interval")
    parser.add_argument("--beta-max",       type=float, default=0.3,         help="max diffusion for the diffusion model")
    parser.add_argument("--beta-min",       type=float, default=1e-5,         help="min diffusion for the diffusion model")
    # parser.add_argument("--beta-min",       type=float, default=0.1)    
    parser.add_argument("--ot-ode",         action="store_true",             help="use OT-ODE model")
    parser.add_argument("--clip-denoise",   action="store_true",             help="clamp predicted image to [-1,1] at each")
    # optional configs for conditional network
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",   action="store_true",             help="add noise to conditional network")
    parser.add_argument("--noise-preprocessing",        action="store_true", help="add noise to network")
    parser.add_argument("--basic-preprocessing",        action="store_true", help="use basic preprocessing")
    parser.add_argument("--noise-muon",        action="store_true", help="add noise to muon angle variables")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",     type=int,   default=128)
    parser.add_argument("--microbatch",     type=int,   default=128,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",        type=int,   default=250000,     help="training iteration")
    parser.add_argument("--lr",             type=float, default=1e-3,        help="learning rate")
    parser.add_argument("--lr-gamma",       type=float, default=0.99,        help="learning rate decay ratio")    
    parser.add_argument("--lr-cos-anneal",  action="store_true",            help="learning rate decay cos-anneal flag")
    parser.add_argument("--lr-step",        type=int,   default=500,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0)
    parser.add_argument("--ema",            type=float, default=0.999)
    parser.add_argument("--kl",             type=float, default=1.0)

    # --------------- path and logging ---------------
    parser.add_argument("--log-dir",        type=Path,  default=".log",      help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default=None,        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")
    parser.add_argument("--hidden-dim",     type=int,   default=45,          help="size of hidden dims")
    parser.add_argument("--nresnet",        type=int,   default=4,           help="number of resnet layers")
    parser.add_argument("--layerresnet",    type=int,   default=1,           help="number of layers in resnet")
    
    # --------------- transformer ---------------

    parser.add_argument("--trans", action="store_true",          help="use transformer model")
    parser.add_argument("--transV2", action="store_true",          help="use transformer model")
    parser.add_argument("--transV3", action="store_true",          help="use transformer model")
    parser.add_argument("--dim-embedding", type=int, default=64, help="dimension of embedding")
    parser.add_argument("--n-head", type=int, default=4, help="number of heads in transformer")
    parser.add_argument("--n-encoder-layers", type=int, default=6, help="number of encoder layers in transformer")
    parser.add_argument("--n-decoder-layers", type=int, default=6, help="number of decoder layers in transformer")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="dimension of feedforward in transformer")
    parser.add_argument("--bayes", action="store_true", help="use bayesian transformer")
    
    
    opt = parser.parse_args()

    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt / "latest.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("         Unfolding Schrodinger Bridge")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    
    #Loading and preprocessing baseline OmniFold data
    
    if 'OmniFold_train' in opt.sample_name:
        train_gen, train_sim, test_gen,test_sim = utils.DataLoaderComparison(
            opt.sample_name,
            json_path = '../JSON',
            cache_dir="/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_data/OmniFold_big/",
            test_name='OmniFold_test.h5'
        )
    elif 'Z_2j' in opt.sample_name:
        train_gen, train_sim, test_gen,test_sim = utils.DataLoaderZ2Jet(
            opt.sample_name,
            json_path = '../JSON',
            cache_dir="/global/ml4hep/spss/sdiefenbacher/Schroedinger_Bridge/Unfolding/Z_2Jet_new_delphes/",
            test_name=None,
            noise_preprocessing=opt.noise_preprocessing,
            noise_muon=opt.noise_muon,
            basic_preprocessing=opt.basic_preprocessing
        )
    else:
        train_gen, train_sim, test_gen,test_sim = utils.DataLoader(opt.sample_name,json_path = '../JSON')

    
    train_dataset = TensorDataset(torch.tensor(train_gen).float(), torch.tensor(train_sim).float())
    val_dataset   = TensorDataset(torch.tensor(test_gen).float(), torch.tensor(test_sim).float())

    
    run = runner_physics.Runner_physics(opt, log)
    run.train(opt, train_dataset, val_dataset)
    log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()

    assert opt.corrupt is not None

    if opt.distributed:
        size = opt.n_gpu_per_node

        processes = []
        for rank in range(size):
            opt = copy.deepcopy(opt)
            opt.local_rank = rank
            global_rank = rank + opt.node_rank * opt.n_gpu_per_node
            global_size = opt.num_proc_node * opt.n_gpu_per_node
            opt.global_rank = global_rank
            opt.global_size = global_size
            print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
            p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)
        #main(opt)
        
        
        
        
#python train_physics.py --name 'SBUnfold' --n-gpu-per-node 1 --batch-size 256 --microbatch 128 --beta-max 0.3 --log-writer None

