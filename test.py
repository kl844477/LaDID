import os
import glob
import sys
sys.path.append('./utils')

import glob
import h5py
from einops import rearrange
import numpy as np
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import json
from types import SimpleNamespace


import argparse
from tqdm import tqdm
import time
from einops import rearrange

import dataset # from sys path
from LaDID import SerializedModel

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def normalized_mse(y_true, y_pred):
    diff = np.linalg.norm(rearrange(y_true, 'a n l c h w -> a n l (c h w)')-rearrange(y_pred, 'a n l c h w -> a n l (c h w)'), ord=2, axis=-1)**2
    gt_norm = np.linalg.norm(rearrange(y_true, 'a n l c h w -> a n l (c h w)'), ord=2, axis=-1)**2
    return diff / gt_norm

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_inference_data(t: torch.Tensor, y: torch.Tensor, delta_inf: float) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        t_inf, y_inf = [], []
        for i in range(t.shape[0]):
            inf_inds = torch.argwhere(t[[i]] <= delta_inf)[:, 1]
            t_inf.append(t[[i]][:, inf_inds, :])
            y_inf.append(y[[i]][:, inf_inds, :, :])
        return t_inf, y_inf


def get_x0(t: list[torch.Tensor], y: list[torch.Tensor], model: nn.Module) -> torch.Tensor:
    x0 = []
    for ti, yi in zip(t, y):
        model.rec_net.update_time_grids(ti)
        gamma, tau = model.rec_net(yi)
        noise = torch.randn(size=tau.size(), generator=model.rng, dtype=tau.dtype, device=tau.device, requires_grad=False)
        s = gamma + tau * noise 

        x0.append(s[:, [0], :])
    return torch.cat(x0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--WORKERS', default=0, type=int)
    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    parser.add_argument('--EXP_CHECKPOINT_DIR', type=str, required=True)
    test_args = parser.parse_args()    

    RANK = int(os.environ.get('RANK'))
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK'))
    WORLD_SIZE = int(os.environ.get('WORLD_SIZE'))
    DEVICE_ORDINAL = 0 #This is managed by CUDA_VISIBLE_DEVICES

    device = torch.device("cuda:{}".format(DEVICE_ORDINAL))

    assert RANK != None 
    assert LOCAL_RANK != None 
    assert WORLD_SIZE != None 

    if WORLD_SIZE > 0:
        test_args.is_distributed = True

    if '.chkpt' not in test_args.EXP_CHECKPOINT_DIR:   
        checkpoint_path = glob.glob(os.path.join(test_args.EXP_CHECKPOINT_DIR, 'checkpoints', 'best_model_*.chkpt'))
        assert len(checkpoint_path) == 1, "Too many checkpoints in your EXP DIR"
        checkpoint_path = checkpoint_path[0]
        
        with open(os.path.join(test_args.EXP_CHECKPOINT_DIR, "hyper_parameters"), 'r') as f:
            args = SimpleNamespace(**json.load(f))
    else:
        checkpoint_path = test_args.EXP_CHECKPOINT_DIR
        hyperparam_dir = test_args.EXP_CHECKPOINT_DIR.split('checkpoints/')[0]
        with open(os.path.join(hyperparam_dir, "hyper_parameters"), 'r') as f:
            args = SimpleNamespace(**json.load(f))
    
    checkpoint = torch.load(checkpoint_path)

    print("################################## restored args ###########################################")
    print("Architeture: ", args.ARCHITECTURE)
    print("DATASET:", args.DATASET)
    print("DATA FOLDER", args.data_folder)
    print(args)
    print("############################################################################################")

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    dist.barrier()

    if args.unset_random_seed:
        pass
    else:
        set_random_seeds(args.seed)
    
    if WORLD_SIZE > 0:
        args.is_distributed = True

    # load and prepare data
    args = dataset.load_and_prepare_data(args)

    if '.chkpt' not in test_args.EXP_CHECKPOINT_DIR:
        test_args.PREDICTION_PATH = os.path.join(args.LOG_PATH, 'predictions.h5')
    else:
        if test_args.autoregressive:
            pred_name = test_args.EXP_CHECKPOINT_DIR.split('/')[-1].split('.')[0] + '_AR.h5'
        else:
            pred_name = test_args.EXP_CHECKPOINT_DIR.split('/')[-1].split('.')[0] + '.h5'
        test_args.PREDICTION_PATH = os.path.join(args.LOG_PATH, pred_name)

    rng_gen = torch.Generator()
    rng_gen.manual_seed(0)
    
    _, _, test_dataset  = dataset.create_datasets(args)
    args.t_min, args.t_max = torch.min(test_dataset.t[0]), test_dataset.t[0][-1]    

    test_sampler = torch.utils.data.DistributedSampler(test_dataset, shuffle=False) if args.is_distributed else None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_args.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=test_sampler,
        drop_last=True,
        collate_fn=None,
        worker_init_fn=seed_worker,
        generator=rng_gen,
        )       

    model = SerializedModel(args)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)   
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ORDINAL], output_device=DEVICE_ORDINAL)
            
    # Test
    model.eval()
    model.module.set_rng(0)

    if RANK == 0:
        test_pbar = tqdm(enumerate(test_loader), total=test_loader.__len__(), desc='TEST', disable=False)
    else:
        test_pbar = tqdm(enumerate(test_loader), total=test_loader.__len__(), desc='TEST', disable=True)
    
    y_pred = []
    y_true = []
    x_latent = []
    
    with torch.no_grad():
        for i, test_batch in test_pbar:  
            if not args.read_data_and_info:
                t, y, traj_inds = test_batch
            else:
                t, y, traj_inds, y_info = test_batch

            y = rearrange(y, 'n l c hw -> n l hw c')    
            y = y.to(device)
            t = t.to(device)

            if args.DATASET in ['reaction_diffusion', "vortex_street"]:
                overall_query_points = 40
            else:
                overall_query_points = 50

            _, _, x_traj, y_full_traj = model(t=t, y=y, batch_ids=traj_inds, block_size=1, scaler=1., num_input_steps=args.num_input_steps, num_query_points=overall_query_points, testing=True)
            
            y_pred.append(y_full_traj.detach().cpu().numpy())
            y_true.append(y[:,args.num_input_steps:, :, :].detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    with h5py.File(test_args.PREDICTION_PATH, 'w') as hf:
        hf.create_dataset('y_pred', data=y_pred, dtype=np.float32)
        hf.create_dataset('y_true', data=y_true, dtype=np.float32)            

if __name__ == '__main__':
    main()
    
    