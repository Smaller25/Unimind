import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt
import wandb
import copy

from config import Config_MBM_fMRI
from dataset import hcp_dataset, NSDDatasetAE, NSDDatasetImage, NSDDatasetSiam
import torchvision.transforms as transforms
from einops import rearrange
from autoencoder.autoencoder import fmri_encoder
from autoencoder.trainer import train_one_epoch, train_one_epoch_finetune, train_one_epoch_finetune_s_p
from autoencoder.trainer import NativeScalerWithGradNormCount as NativeScaler
from autoencoder.utils import save_model

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
    parser = argparse.ArgumentParser('Autoencoder training for fMRI', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    # Model Parameters
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--decoder_embed_dim', type=int)
    parser.add_argument('--depth', type=int)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--roi', type=str)
    parser.add_argument('--aug_times', type=int)
    parser.add_argument('--num_sub_limit', type=int)
    parser.add_argument('--pretrain_encoder_path', type=str)
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--resume',default='',type=str)
                        
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def my_collate_img(batch):
    collate_img = []
    collate_voxels = []
    max_len = max([len(sample['fmri'][-1]) for sample in batch])
    for sample in batch:
        diff = max_len - len(sample['fmri'][-1])
        if diff>0:
            zero_pad = torch.zeros(size=(sample['fmri'].shape[0], diff))
            collate_voxels.append(torch.cat([sample['fmri'], zero_pad], dim=1))
            collate_img.append(sample['image'])
        else:
            collate_voxels.append(sample['fmri'])
            collate_img.append(sample['image'])
    return {'image' : torch.stack(collate_img), 'fmri' : torch.stack(collate_voxels)}


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img
    
def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def main(config):
    print("ongoing1")

    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    else: #단일 GPU일때
        os.environ["CUDA_VISIBLE_DEVICES"]= "0"

    output_path = os.path.join(config.root_path, 'results', 'stage2_finetune',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    print('ongoing2, device is ', device)
    
    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,
        random_crop(config.img_size-crop_pix, p=0.5),
        transforms.Resize((256, 256)), 
    ])
    
    # create dataset and dataloader
    # dataset_pretrain = NSDDatasetAE(root_dir = '/home/smaller/E_geo_bo_se_yoooooo/data/webdataset_avg_split/train', folder_list=['test_folder'], image_transform = img_transform_train)
    dataset_pretrain = NSDDatasetSiam(root_dir = '/home/smaller/mind_vis_contrastive/data/webdataset_avg_split/train', folder_list=['subj01','subj02', 'subj05'])
   
    print(f'Dataset size: {len(dataset_pretrain)}')
    sampler = torch.utils.data.DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 
    
    dataloader_hcp = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, shuffle=(sampler is None), pin_memory=True)

    # create model
    meta_file = torch.load(config.pretrain_encoder_path)
    config_encoder = meta_file['config']

    model = fmri_encoder(in_chans = config_encoder.in_chans, patch_size=config_encoder.patch_size, kernel_size = config_encoder.kernel_size, embed_dim=config_encoder.embed_dim, depth=config_encoder.depth)
    print(meta_file.keys())
    model.load_state_dict(meta_file['model'], strict = False)
    print(f'Model Loaded Complete')
    model.to(device)
    model_without_ddp = model
    
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

    start_time = time.time()
    print('Start Training the fmri AE ... ...')
    
    total_loss = []
    val_mse_list = []

    for ep in range(start, config.num_epoch):
        if torch.cuda.device_count() > 1: 
            sampler.set_epoch(ep) # to shuffle the data at every epoch
        loss = train_one_epoch_finetune_s_p(model, dataloader_hcp, optimizer, device, ep, loss_scaler, logger, config, start_time)
        total_loss.append(loss)

        if (ep % 5 == 0 or ep + 1 == config.num_epoch) and ep != 0 :
            save_model(config, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,'checkpoints'))
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
        logger.log('average loss', np.mean(total_loss), step=config.num_epoch-1)
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
    config = Config_MBM_fMRI()
    config = update_config(args, config)
    main(config)
    