# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os,sys
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu

from logger import Logger
import distributed_util as dist_util
import ckpt_util

import colored_traceback.always
from ipdb import set_trace as debug
import runner_physics

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

def build_subset_per_gpu(opt, dataset, log):
    n_data = len(dataset)
    n_gpu  = opt.global_size
    n_dump = (n_data % n_gpu > 0) * (n_gpu - n_data % n_gpu)

    # create index for each gpu
    total_idx = np.concatenate([np.arange(n_data), np.zeros(n_dump)]).astype(int)
    idx_per_gpu = total_idx.reshape(-1, n_gpu)[:, opt.global_rank]
    log.info(f"[Dataset] Add {n_dump} data to the end to be devided by {n_gpu=}. Total length={len(total_idx)}!")

    # build subset
    indices = idx_per_gpu.tolist()
    subset = Subset(dataset, indices)
    log.info(f"[Dataset] Built subset for gpu={opt.global_rank}! Now size={len(subset)}!")
    return subset

def collect_all_subset(sample, log):
    batch, *xdim = sample.shape
    gathered_samples = dist_util.all_gather(sample, log)
    gathered_samples = [sample.cpu() for sample in gathered_samples]
    # [batch, n_gpu, *xdim] --> [batch*n_gpu, *xdim]
    return torch.stack(gathered_samples, dim=1).reshape(-1, *xdim)

def build_partition(opt, full_dataset, log):
    n_samples = len(full_dataset)

    part_idx, n_part = [int(s) for s in opt.partition.split("_")]
    assert part_idx < n_part and part_idx >= 0
    assert n_samples % n_part == 0

    n_samples_per_part = n_samples // n_part
    start_idx = part_idx * n_samples_per_part
    end_idx = (part_idx+1) * n_samples_per_part

    indices = [i for i in range(start_idx, end_idx)]
    subset = Subset(full_dataset, indices)
    log.info(f"[Dataset] Built partition={opt.partition}, {start_idx=}, {end_idx=}! Now size={len(subset)}!")
    return subset


def get_recon_imgs_fn(opt, nfe):
    sample_dir = RESULT_DIR / opt.ckpt / "samples_nfe{}{}".format(
        nfe, "_clip" if opt.clip_denoise else ""
    )
    os.makedirs(sample_dir, exist_ok=True)

    recon_imgs_fn = sample_dir / "recon{}.pt".format(
        "" if opt.partition is None else f"_{opt.partition}"
    )
    return recon_imgs_fn

def compute_batch(ckpt_opt, out):
    clean_img, corrupt_img = out
    mask = None
    
    cond = corrupt_img.detach() if ckpt_opt.cond_x1 else None
    if ckpt_opt.add_x1_noise: # only for decolor
        x1 = x1 + torch.randn_like(x1)

    return corrupt_img, clean_img, mask, cond

@torch.no_grad()
def main(opt):
    log = Logger(opt.global_rank, ".log")

    print(RESULT_DIR / opt.ckpt)
    # get (default) ckpt option
    ckpt_opt = ckpt_util.build_ckpt_option(opt, log, RESULT_DIR / opt.ckpt)
    corrupt_type = ckpt_opt.corrupt
    nfe = opt.nfe or ckpt_opt.interval-1
    
    
    
    
    if 'OmniFold_train' in opt.sample_name:
        _, _, test_gen,test_sim = utils.DataLoaderComparison(
            opt.sample_name,
            json_path = '../JSON',
            cache_dir="/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_data/OmniFold_big/",
            test_name='OmniFold_test.h5'
        )
    elif 'Z_2j' in opt.sample_name:
        _, _, test_gen,test_sim = utils.DataLoaderZ2Jet(
            opt.sample_name,
            json_path = '../JSON',
            #cache_dir="/global/cfs/projectdirs/m3246/sdiefenbacher/Schroedinger_Bridge/SBUnfold_data/Z_2Jet_new_delphes/",
            cache_dir="/global/ml4hep/spss/sdiefenbacher/Schroedinger_Bridge/Unfolding/Z_2Jet_new_delphes/",
            test_name=None,
            basic_preprocessing=ckpt_opt.basic_preprocessing if hasattr(ckpt_opt, 'basic_preprocessing') else False         
        )
    else:
        _, _, test_gen,test_sim = utils.DataLoader(opt.sample_name,json_path = '../JSON')

    
    
    print("AAAAAAAAAAAAA")
    val_dataset   = TensorDataset(torch.tensor(test_gen).float(), torch.tensor(test_sim).float())
        
    n_samples = len(val_dataset)
    
    if n_samples > opt.n_eval:
        n_samples = opt.n_eval

    # build dataset per gpu and loader
    subset_dataset = build_subset_per_gpu(opt, val_dataset, log)
    val_loader = DataLoader(subset_dataset,
        batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False,
    )

    # build runner
    runner = runner_physics.Runner_physics(ckpt_opt, log, save_opt=False)

    # handle use_fp16 for ema
    if opt.use_fp16:
        runner.ema.copy_to() # copy weight from ema to net
        runner.net.diffusion_model.convert_to_fp16()
        runner.ema = ExponentialMovingAverage(runner.net.parameters(), decay=0.99) # re-init ema with fp16 weight

    # create save folder
    recon_imgs_fn = get_recon_imgs_fn(opt, nfe)
    log.info(f"Recon images will be saved to {recon_imgs_fn}!")

    if opt.bnn_map:
        runner.bayes_set_map()

        recon_imgs = []
        ys = []
        num = 0
        for loader_itr, out in enumerate(val_loader):
            corrupt_img, clean_img, mask, cond= compute_batch(ckpt_opt, out)

            xs, _ = runner.ddpm_sampling(
                ckpt_opt, corrupt_img, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe,
                verbose=opt.n_gpu_per_node==1
            )
            recon_img = xs[:, 0, ...].to(opt.device)

            assert recon_img.shape == corrupt_img.shape

            # [-1,1]
            gathered_recon_img = collect_all_subset(recon_img, log)
            recon_imgs.append(gathered_recon_img)

            num += len(gathered_recon_img)
            log.info(f"Collected {num} recon images!")
            dist.barrier()

            if num > opt.n_eval:
                break


        recon_imgs = torch.cat(recon_imgs, axis=0)[:n_samples]

        if opt.global_rank == 0:
            np.save('{}/clean_{}_map.npy'.format(opt.save_target, opt.ckpt), test_gen[:num])
            np.save('{}/corrupt_{}_map.npy'.format(opt.save_target, opt.ckpt), test_sim[:num])
            np.save('{}/recon_{}_map.npy'.format(opt.save_target, opt.ckpt), recon_imgs.detach().cpu().numpy())

        dist.barrier()
        
    else:
        for i_ens in range(opt.n_ensemble_start, opt.n_ensemble):
            try:
                runner.bayes_reset_random()
            except: 
                print("not BNN")

            recon_imgs = []
            ys = []
            num = 0
            for loader_itr, out in enumerate(val_loader):
                corrupt_img, clean_img, mask, cond= compute_batch(ckpt_opt, out)

                xs, _ = runner.ddpm_sampling(
                    ckpt_opt, corrupt_img, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, nfe=nfe,
                    verbose=opt.n_gpu_per_node==1
                )
                recon_img = xs[:, 0, ...].to(opt.device)

                assert recon_img.shape == corrupt_img.shape

                # [-1,1]
                gathered_recon_img = collect_all_subset(recon_img, log)
                recon_imgs.append(gathered_recon_img)

                num += len(gathered_recon_img)
                log.info(f"Collected {num} recon images!")
                dist.barrier()

                if num > opt.n_eval:
                    break


            recon_imgs = torch.cat(recon_imgs, axis=0)[:n_samples]

            if opt.global_rank == 0:
                np.save('{}/clean_{}_ens{}.npy'.format(opt.save_target, opt.ckpt, i_ens), test_gen[:num])
                np.save('{}/corrupt_{}_ens{}.npy'.format(opt.save_target, opt.ckpt, i_ens), test_sim[:num])
                np.save('{}/recon_{}_ens{}.npy'.format(opt.save_target, opt.ckpt, i_ens), recon_imgs.detach().cpu().numpy())

            dist.barrier()
    
    
    del runner

    log.info(f"Sampling complete! Collect recon_imgs={recon_imgs.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,  default=0)
    parser.add_argument("--n-gpu-per-node", type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",      type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--partition",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")
    parser.add_argument("--sample-name",      type=str,  default="Herwig",  help="Load data sample")

    # sample
    parser.add_argument("--batch-size",     type=int,  default=100000)
    parser.add_argument("--ckpt",           type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",            type=int,  default=None,        help="sampling steps")
    parser.add_argument("--clip-denoise",   action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",       action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--n-ensemble",     type=int,  default=10,            help="number of times the BNN is sampled")
    parser.add_argument("--n-ensemble-start",type=int,  default=0,            help="number of times the BNN is sampled")
    parser.add_argument("--n-eval",     type=int,  default=600000,            help="number data point in the test set that are evaluate")
    parser.add_argument("--save-target",     type=str,  default='.',            help="directory to save the outputs to")
    parser.add_argument("--hidden-dim",     type=int,   default=45,          help="size of hidden dims")
    parser.add_argument("--nresnet",        type=int,   default=4,           help="number of resnet layers")
    parser.add_argument("--layerresnet",    type=int,   default=1,           help="number of layers in resnet")
    parser.add_argument("--bnn-map", action="store_true", help="sets bnn to use mean instead of sample")
    parser.add_argument("--cond-x1",        action="store_true",             help="conditional the network on degraded images")

    parser.add_argument("--trans", action="store_true",          help="use transformer model")
    parser.add_argument("--dim-embedding", type=int, default=64, help="dimension of embedding")
    parser.add_argument("--n-head", type=int, default=4, help="number of heads in transformer")
    parser.add_argument("--n-encoder-layers", type=int, default=6, help="number of encoder layers in transformer")
    parser.add_argument("--n-decoder-layers", type=int, default=6, help="number of decoder layers in transformer")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="dimension of feedforward in transformer")
    parser.add_argument("--bayes", action="store_true", help="use bayesian transformer")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))


    set_seed(opt.seed)

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
            p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, opt))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        dist_util.init_processes(0, opt.n_gpu_per_node, main, opt)
