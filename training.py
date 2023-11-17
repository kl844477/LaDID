import os
import sys
sys.path.append('./utils')

import glob

from einops import rearrange
import numpy as np
import random
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import json

import argparse
from tqdm import tqdm
from typing import Tuple
import shutil 
from einops import rearrange
from datetime import datetime

import dataset # from sys path
from LaDID import SerializedModel

def set_random_seeds(random_seed=0):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--WORKERS', default=0, type=int)
    parser.add_argument('--EPOCHS', default=1000, type=int)
    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    
    parser.add_argument('--RESULT_BASE_PATH', default='./results/tests/', type=str)
    parser.add_argument('--unset_random_seed', action='store_true')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Data
    parser.add_argument('--DATASET', type=str, required=True)
    parser.add_argument('--read_data_and_info', default=False, type=eval)
    parser.add_argument('--checkpoint_path', type=str, default=None)

    # Optimizer
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='reducelronplateau')
    parser.add_argument('--reduce_factor', type=float, default=0.2)
    parser.add_argument('--patience_level', type=int, default=15)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--lr_schedule', nargs='+', default=[20, 50 , 100, 300])
   
    parser.add_argument("--max_len", type=int, default=60, help="Truncation length for trajectories.")
    parser.add_argument("--norm_strategy", type=str, default="images", help="normalization strategy in data preprocessing")
    parser.add_argument("--sigY", type=float, default=1e-3, help="Observation noise.")

    # Model
    parser.add_argument("--K", type=int, default=32, help="Latent space dimension.")
    parser.add_argument("--Xi", type=float, default=1e-4, help="Used to set variance for the continuity prior.")
    parser.add_argument("--block_size", type=int, default=1, help="Number of time points in each block.")
    parser.add_argument("--g_cnn_channels", type=int, default=8, help="Channels in CNNDecoder.")
    parser.add_argument("--m_F", type=int, default=8, help="Dimensionality scaler for F.")
    parser.add_argument("--F_nonlin", type=str, default="relu", help="Nonlinearity for F.")
    parser.add_argument("--dyn_order", type=int, default=1, help="Order of the dynamcis function, must be 1 or 2.")
    parser.add_argument("--m_h", type=int, default=4, help="Dimensionality scaler for h.")
    parser.add_argument("--h_enc_cnn_channels", type=int, default=8, help="Channels in CNNEncoder.")
    parser.add_argument("--h_agg_attn", type=str, default="dp", help="Attention type (dp, t, tdp, tdp_b).")
    parser.add_argument("--h_agg_pos_enc", type=str, default="rpeNN", help="Position encoding type (csc, rpeNN, rpeInterp).")
    parser.add_argument("--h_agg_stat_layers", type=int, default=4, help="Number of TFEncoder layers in static aggregation net.")
    parser.add_argument("--h_agg_dyn_layers", type=int, default=8, help="Number of TFEncoder layers in dynamic aggregation net.")
    parser.add_argument("--h_agg_max_tokens", type=int, default=51, help="Maximum expected number of tokens.")
    parser.add_argument("--h_agg_max_time", type=float, default=3.0, help="Maximum expected observation time.")
    parser.add_argument("--h_agg_delta_r", type=float, default=0.45, help="Attention time span at training time.")
    parser.add_argument("--h_agg_p", type=float, default=10**15, help="Exponent for temporal attention (use -1 for p=inf).")
    parser.add_argument("--n", type=int, default=1, help="Number of nearest neighbors used for baseline aggregation net.")
    parser.add_argument("--drop_prob", type=float, default=0.0, help="Attention dropout probability.")  # 0.1
    parser.add_argument("--tau_min", type=float, default=2e-2, help="Lower bound on the variance of q(s_i).")  # 2e-2
    parser.add_argument("--sigT", type=float, default=0.0, help="Scale of the noise added to the time grids for temporal neighborhood adjustment.")  # 0.00025
    parser.add_argument("--delta_inf", type=float, default=0.45, help="Fraction of obsevations used for x0 inference at test time.")
    parser.add_argument("--aggregation_heads", type=int, default=4, help="Number of attention heads in aggregation module.")
    parser.add_argument("--m_AGG", type=int, default=1, help="Dimensionality scaler for the dimension of the aggregation module.")
    parser.add_argument("--F_drop", type=float, default=0.5, help="Dropout in query function")
    parser.add_argument("--variational", type=eval, default=True, help="Variational encoder")

    # MS training
    parser.add_argument("--increase_block_size", type=eval, default=False, help="Increase block size sequentially")
    parser.add_argument("--epochs_per_block_size_increase", type=int, default=250, help="Number of epochs before block size is increased.")
    parser.add_argument("--num_input_steps", type=int, default=10, help="Number of timesteps used as input.")
    parser.add_argument("--num_query_points", type=int, default=1, help="Number of timesteps to query.")
    parser.add_argument("--num_overlap_points", type=int, default=0, help="Number of timesteps which should overlap.")
    parser.add_argument("--t_min", type=float, default=0.0, help="Minimum time of trajectory --- used for temporal encoding")
    parser.add_argument("--t_max", type=float, default=0.45, help="Maximium time of trajectory --- used for temporal encoding.")

    # Trajectory encoding and loss
    parser.add_argument("--t_pos_enc", type=str, default="rel_pe_sincos")
    parser.add_argument("--learnable_pos_enc", type=eval, default=True, help="Attach linear layer to sin/cos pos embedding")
    parser.add_argument("--relative_temporal_encoding", type=eval, default=True, help="Use relative encoding")
    parser.add_argument("--encode_full_trajectory", type=eval, default=True, help="Activate encoding of full trajectory")
    parser.add_argument("--static_representation", type=eval, default=True, help="Add a static image representation")
    parser.add_argument("--smoothness_scaling", type=float, default=1.0, help="Dropout in query function")
    parser.add_argument("--use_L2_2", type=eval, default=True, help="Use L2_2 loss.")

    args = parser.parse_args()

    maxEntry_power2 = int(np.ceil(np.log2(args.max_len)))
    inc_query_steps = 2 ** np.arange(maxEntry_power2+1)
    inc_query_steps[-1] = args.max_len - args.num_input_steps
    if args.increase_block_size and args.num_overlap_points != 0:           
        overlap_steps = [1, 1, 2, 4, 8, 16, 0]
        args.num_overlap_points = overlap_steps[0]

    max_len = args.num_input_steps + ((args.max_len - args.num_input_steps - args.num_overlap_points) // args.num_query_points ) * args.num_query_points + args.num_overlap_points
    args.max_len = max_len 

    RANK = int(os.environ.get('RANK'))
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK'))
    WORLD_SIZE = int(os.environ.get('WORLD_SIZE'))
    DEVICE_ORDINAL = 0

    device = torch.device("cuda:{}".format(DEVICE_ORDINAL))

    assert RANK != None 
    assert LOCAL_RANK != None 
    assert WORLD_SIZE != None 

    if args.unset_random_seed:
        pass
    else:
        set_random_seeds(args.seed)

    if WORLD_SIZE > 0:
        args.is_distributed = True

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    dist.barrier()

    # Save directories and logging
    if RANK == 0:
        training_begin = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        args.LOG_PATH = os.path.join(args.RESULT_BASE_PATH, "%s_%s_%s"%(args.DATASET, training_begin))
        args.CHECKPOINT_PATH = os.path.join(args.LOG_PATH, 'checkpoints')
        args.RESULT_HDF_PATH = os.path.join(args.LOG_PATH, 'predictions.h5')

        if os.path.exists(args.LOG_PATH):
            shutil.rmtree(args.LOG_PATH)

        if not os.path.exists(args.CHECKPOINT_PATH):
            os.makedirs(args.CHECKPOINT_PATH)

    # load and prepare data
    args = dataset.load_and_prepare_data(args)
    
    if RANK == 0:
        writer = SummaryWriter(log_dir=args.LOG_PATH)
        with open(os.path.join(args.LOG_PATH, 'hyper_parameters'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        log_file = open(os.path.join(args.LOG_PATH, 'log.txt'), "w")

    rng_gen = torch.Generator()
    rng_gen.manual_seed(0)

    train_dataset, val_dataset, _  = dataset.create_datasets(args)
    args.t_min, args.t_max = torch.min(train_dataset.t[0]), train_dataset.t[0][args.max_len-1]

    train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True) if args.is_distributed else None
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=4 if args.DATASET == 'vortex_street' else args.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=None,
        worker_init_fn=seed_worker,
        generator=rng_gen,
        )

    val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False) if args.is_distributed else None
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=val_sampler,
        drop_last=True,
        collate_fn=None,
        worker_init_fn=seed_worker,
        generator=rng_gen,
        )      

    model = SerializedModel(args)

    if args.checkpoint_path != None:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True) 
        print('Checkpoint loaded: ', args.checkpoint_path)

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[DEVICE_ORDINAL], output_device=DEVICE_ORDINAL)
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
         
    if args.scheduler in ['multisteplr']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.int_(args.lr_schedule), gamma=0.5, verbose=True)
    elif args.scheduler in ['exponentiallr']:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1e-5/args.lr)**(1.0/args.EPOCHS), verbose=True)
    elif args.scheduler in ['cosinelr']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.reduce_factor, patience=args.patience_level, min_lr=args.min_lr, verbose=True)
            
    best_val_metric = 1e32
    print('*************************   start training   *************************', flush=True)
    print(args)
    print(args, file=sys.stderr)

    
    metric_train_loss = torchmetrics.MeanMetric(dist_sync_on_step=True, nan_strategy='ignore').to(device)
    metric_train_L1_loss = torchmetrics.MeanMetric(dist_sync_on_step=True, nan_strategy='ignore').to(device)
    metric_train_L2_loss = torchmetrics.MeanMetric(dist_sync_on_step=True, nan_strategy='ignore').to(device)

    metric_val_loss = torchmetrics.MeanMetric(dist_sync_on_step=True, nan_strategy='ignore').to(device)
    metric_val_L1_loss = torchmetrics.MeanMetric(dist_sync_on_step=True, nan_strategy='ignore').to(device)
    metric_val_L2_loss = torchmetrics.MeanMetric(dist_sync_on_step=True, nan_strategy='ignore').to(device)

    inc_idx = 0

    for epoch in range(args.EPOCHS):
        #training
        if epoch % args.epochs_per_block_size_increase == 0 and epoch > 0 and args.increase_block_size and args.num_query_points < inc_query_steps[-1]:
            inc_idx = inc_idx + 1
            args.num_query_points = inc_query_steps[inc_idx]
            if args.num_overlap_points != 0:
                args.num_overlap_points = overlap_steps[inc_idx]

            if inc_idx == len(inc_query_steps) - 1 :
                args.max_len = args.num_input_steps + args.num_query_points
            else:
                max_len = args.num_input_steps + ((args.max_len - args.num_input_steps - args.num_overlap_points) // args.num_query_points ) * args.num_query_points + args.num_overlap_points
                args.max_len = max_len

            train_dataset, val_dataset, test_dataset  = dataset.create_datasets(args)
            args.t_min, args.t_max = torch.min(train_dataset.t[0]), train_dataset.t[0][args.max_len-1]

            train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True) if args.is_distributed else None
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=4 if args.DATASET == 'vortex_street' else args.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                sampler=train_sampler,
                drop_last=True,
                collate_fn=None,
                worker_init_fn=seed_worker,
                generator=rng_gen,
                )

            val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False) if args.is_distributed else None
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                sampler=val_sampler,
                drop_last=True,
                collate_fn=None,
                worker_init_fn=seed_worker,
                generator=rng_gen,
                )

            print('#################### data loader udpate - block size increased !!! ################### - block size: ', args.num_query_points)

            best_val_metric = 1e32
            print('#################### validation metric resetted !!! ###################')
        
        model.train()

        if RANK == 0:
            train_pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc='Epoch %s / %s TRAINING'%(epoch +1, args.EPOCHS))
        else:
            train_pbar = tqdm(enumerate(train_loader), total=train_loader.__len__(), desc='Epoch %s / %s TRAINING'%(epoch +1, args.EPOCHS), disable=True)
        for i, train_batch in train_pbar:       
            if not os.environ.get('VAL_BATCH') is None:
                os.environ.pop('VAL_BATCH')
            os.environ['TRAIN_BATCH'] = str(i)


            if not args.read_data_and_info:
                ts, y, traj_inds = train_batch
            else:
                ts, y, traj_inds, y_info = train_batch
                
            y = rearrange(y, 'n l c hw -> n l hw c')    
            y = y.to(device)

            L1, L2, x = model(t=ts, y=y, batch_ids=traj_inds, block_size=args.block_size, scaler=1., num_input_steps=args.num_input_steps, num_query_points=args.num_query_points, overlap=args.num_overlap_points)
            L1 *= len(train_dataset) / args.BATCH_SIZE
            L2 *= len(train_dataset) / args.BATCH_SIZE
            
            loss = -(L1 - L2 )

            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()

            batch_loss = metric_train_loss(loss)
            batch_L1_loss = metric_train_L1_loss(L1)
            batch_L2_loss = metric_train_L2_loss(L2)
            loss = loss.item()
            
            postfix_str = 'loss: %.2e avg_loss: %.2e L1: %.2e L2: %.2e' %(batch_loss.item(), metric_train_loss.compute().item(), batch_L1_loss.item(), batch_L2_loss.item())
            
            train_pbar.set_postfix_str(postfix_str)

        avg_train_loss = metric_train_loss.compute()
        avg_train_L1_loss = metric_train_L1_loss.compute()
        avg_train_L2_loss = metric_train_L2_loss.compute()

        metric_train_loss.reset()
        metric_train_L1_loss.reset()
        metric_train_L2_loss.reset()

        # validation
        model.eval()
        model.module.set_rng(epoch + RANK * 10000)

        if RANK == 0:
            val_pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Epoch %s / %s VALIDATION'%(epoch +1, args.EPOCHS))
        else:
            val_pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Epoch %s / %s VALIDATION'%(epoch +1, args.EPOCHS), disable=False)
        
        for i, val_batch in val_pbar:  
            if not os.environ.get('TRAIN_BATCH') is None:
                os.environ.pop('TRAIN_BATCH')
            os.environ['VAL_BATCH'] = str(i)


            if not args.read_data_and_info:
                ts, y, traj_inds = val_batch
            else:
                ts, y, traj_inds, y_info = val_batch
            
            y = rearrange(y, 'n l c hw -> n l hw c')    
            y = y.to(device)

            with torch.no_grad():
                L1, L2, x = model(t=ts, y=y, batch_ids=traj_inds, block_size=args.block_size, scaler=1., num_input_steps=args.num_input_steps, num_query_points=args.num_query_points, overlap=args.num_overlap_points)
                L1 *= len(val_dataset) / args.BATCH_SIZE
                L2 *= len(val_dataset) / args.BATCH_SIZE
                loss = -(L1 - L2 )

                batch_loss = metric_val_loss(loss)
                batch_L1_loss = metric_val_L1_loss(L1)
                batch_L2_loss = metric_val_L2_loss(L2)

            postfix_str = 'loss: %.2e avg_loss: %.2e L1: %.2e L2: %.2e ' %(batch_loss.item(), metric_val_loss.compute().item(), batch_L1_loss.item(), batch_L2_loss.item())

            val_pbar.set_postfix_str(postfix_str)

        avg_val_loss = metric_val_loss.compute()
        avg_val_L1_loss = metric_val_L1_loss.compute()
        avg_val_L2_loss = metric_val_L2_loss.compute()

        metric_val_loss.reset()
        metric_val_L1_loss.reset()
        metric_val_L2_loss.reset()                     
            
        if RANK == 0:
            writer.add_scalar('training/loss', avg_train_loss.item(), epoch)
            writer.add_scalar('val/loss', avg_val_loss.item(), epoch)
            writer.add_scalar('training/L1-loss', avg_train_L1_loss.item(), epoch)
            writer.add_scalar('training/L2-loss', avg_train_L2_loss.item(), epoch)       
            writer.add_scalar('val/L1-loss', avg_val_L1_loss.item(), epoch)
            writer.add_scalar('val/L2-loss', avg_val_L2_loss.item(), epoch)
            logging_str_train = "Epoch: %i TRAINING: avg_train_loss: %.2e avg_L1: %.2e avg_L2: %.2e "%(epoch, avg_train_loss.item(), avg_train_L1_loss.item(), avg_train_L2_loss.item())
            logging_str_val = "Epoch: %i VALIDATION: avg_val_loss: %.2e avg_L1: %.2e avg_L2: %.2e "%(epoch, avg_val_loss.item(), avg_val_L1_loss.item(), avg_val_L2_loss.item())
            
            tqdm.write(logging_str_train)
            log_file.write(logging_str_train+'\n')
            tqdm.write(logging_str_val)
            log_file.write(logging_str_val+'\n')
            log_file.flush()
            
            if avg_val_loss.item() < best_val_metric:
                if args.increase_block_size:
                        best_val_metric = avg_val_loss.item()
                        ckpt_name = "best_model_query_length_" + str(inc_query_steps[inc_idx]) + "_*.chkpt"
                        filelist = glob.glob(os.path.join(args.CHECKPOINT_PATH, ckpt_name))
                        for f in filelist:
                            os.remove(f)
                        
                        torch.save({'model_state_dict': model.module.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'scheduler_state_dict': scheduler.state_dict(),
                                    'epoch': epoch},
                                    os.path.join(args.CHECKPOINT_PATH, 'best_model_query_length_' + str(inc_query_steps[inc_idx]) + '_epoch_%s_%.2e.chkpt' % (epoch, best_val_metric)))

                else:
                    best_val_metric = avg_val_loss.item()
                    filelist = glob.glob(os.path.join(args.CHECKPOINT_PATH, "best_model_*.chkpt"))
                    for f in filelist:
                        os.remove(f)
                    
                    torch.save({'model_state_dict': model.module.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'epoch': epoch},
                                os.path.join(args.CHECKPOINT_PATH, 'best_model_%s_%.2e.chkpt' % (epoch, best_val_metric)))
        
        if args.scheduler not in ['multisteplr', 'cosinelr', 'exponentiallr']:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
                    

    

if __name__ == '__main__':
    main()
    
    