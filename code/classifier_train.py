import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt
import wandb
import copy

from config import Config_Classifier
from dataset import hcp_dataset, NSDDataset, NSDDataset2
from classifier.trainer import NativeScalerWithGradNormCount as NativeScaler
from classifier.trainer import train_one_epoch_classifier
from classifier.classifier import *
from classifier.utils import save_model

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class wandb_logger:
    def __init__(self, config):
        wandb.init(
                    project="mind-vis",
                    anonymous="allow",
                    group='stageA_sc-mbm',
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

def get_args_parser():
    parser = argparse.ArgumentParser('MBM pre-training for fMRI', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    # Model Parameters
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--depth', type=int)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--roi', type=str)
    parser.add_argument('--aug_times', type=int)
    parser.add_argument('--num_sub_limit', type=int)

    parser.add_argument('--include_hcp', type=bool)
    parser.add_argument('--include_kam', type=bool)

    parser.add_argument('--use_nature_img_loss', type=bool)
    parser.add_argument('--img_recon_weight', type=float)
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
    
    parser.add_argument('--resume',default='',type=str)
                        
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def fmri_transform(x):
    # x: 1, num_voxels
    return torch.FloatTensor(x)

def my_collate(batch):
    collate_subj = []
    collate_voxels = []
    max_len = max([len(sample['voxels'][-1]) for sample in batch])
    for sample in batch:
        diff = max_len - len(sample['voxels'][-1])
        if diff>0:
            zero_pad = torch.zeros(size=(sample['voxels'].shape[0], diff))
            collate_voxels.append(torch.cat([sample['voxels'], zero_pad], dim=1))
            collate_subj.append(sample['subj'])
        else:
            collate_voxels.append(sample['voxels'])
            collate_subj.append(sample['subj'])
    return {'subj' : torch.Tensor(collate_subj), 'voxels' : torch.stack(collate_voxels)}
            

def main(config):
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    output_path = os.path.join(config.root_path, 'results', 'classifier',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # create dataset and dataloader
    NSD_not_2 = True
    if NSD_not_2:
        dataset_pretrain = NSDDataset(root_dir = '/home/smaller/mind_vis_contrastive/data/webdataset_avg_split/train', folder_list=['subj01', 'subj02', 'subj05'])
    
        print(f'Dataset size: {len(dataset_pretrain)}')
        sampler = torch.utils.data.DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

        dataloader = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, shuffle=(sampler is None), pin_memory=True, collate_fn=my_collate)
        
        # Validation Dataset & DataLoader
        val_dataset = NSDDataset(root_dir = '/home/smaller/mind_vis_contrastive/data/webdataset_avg_split/val', folder_list=['subj01', 'subj02', 'subj05'])
        
        print(f'Validation Dataset size: {len(val_dataset)}')
        
        sampler = torch.utils.data.DistributedSampler(val_dataset, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=sampler, shuffle=None, pin_memory=True, collate_fn=my_collate)
    else:
        dataset_pretrain = NSDDataset2(root_dir = '/home/smaller/mind_vis_contrastive/data/webdataset_avg_split/train', folder_list=['subj01', 'subj02', 'subj05'])
    
        print(f'Dataset size: {len(dataset_pretrain)}')
        sampler = torch.utils.data.DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

        dataloader = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, shuffle=(sampler is None), pin_memory=True)
        
        # Validation Dataset & DataLoader
        val_dataset = NSDDataset2(root_dir = '/home/smaller/mind_vis_contrastive/data/webdataset_avg_split/val', folder_list=['subj01', 'subj02', 'subj05'])
        
        print(f'Validation Dataset size: {len(val_dataset)}')
        
        sampler = torch.utils.data.DistributedSampler(val_dataset, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=sampler, shuffle=None, pin_memory=True)
    
    
    
    
    
    # create model
    model = SubjectClassifier(in_chans = config.in_chans, dim = config.dim, depth = config.depth, kernel_size = config.kernel_size, patch_size = config.patch_size, n_classes = config.n_classes)  
    model.to(device)
    model_without_ddp = model
    
    if config.resume:
        ckpt = torch.load(config.resume)
        start = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
    else:
        start = 0
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank)
        
    number_of_model_parameters = count_parameters(model)
    print(f'Number of Parameters in Model is {number_of_model_parameters}')

    param_groups = optim_factory.param_groups_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    loss_list = []
    val_acc_list = []
    start_time = time.time()
    print('Start Training the Classifier ... ...')

    for ep in range(start, config.num_epoch):
        if torch.cuda.device_count() > 1: 
            sampler.set_epoch(ep) # to shuffle the data at every epoch
        loss = train_one_epoch_classifier(model, dataloader, optimizer, device, ep, loss_scaler, logger, config, start_time, model_without_ddp)
        loss_list.append(loss)
        if (ep % 2 == 0 or ep + 1 == config.num_epoch) and ep != 0 and config.local_rank == 0:
            #validation
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for val_dict in val_dataloader:
                    voxels = val_dict['voxels'].to(device)
                    target = val_dict['subj'].type(torch.LongTensor).to(device)
                    
                    output = F.softmax(model(voxels.float()))
                    total += target.size(0)
                    correct += (torch.max(output.data, 1)[1] == target).sum().item()
            correct = correct / total
            val_acc_list.append(correct)
            model.train()
            
            # save models
            save_model(config, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,'checkpoints'))
            print(f'accuracy is : {correct*100}')
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
        logger.log('max cor', np.max(loss_list), step=config.num_epoch-1)
        logger.finish()
    return

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Classifier()
    config = update_config(args, config)
    main(config)
    