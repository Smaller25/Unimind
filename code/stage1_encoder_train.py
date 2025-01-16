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
from dataset import hcp_dataset, NSDDatasetAE
from autoencoder.autoencoder import ConvMixerAutoEncoder as ConvAE
from autoencoder.trainer import train_one_epoch
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
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
    
    parser.add_argument('--resume',default='',type=str)
                        
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def my_collate_ae(batch):
    collate_voxels = []
    max_len = max([len(sample[-1]) for sample in batch])
    for sample in batch:
        diff = max_len - len(sample[-1])
        if diff>0:
            zero_pad = torch.zeros(size=(sample.shape[0], diff))
            collate_voxels.append(torch.cat([sample, zero_pad], dim=1))
        else:
            collate_voxels.append(sample)
    return torch.stack(collate_voxels)

def main(config):
    print('ongoing1')
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')

    else: #단일 GPU일때
        os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    output_path = os.path.join(config.root_path, 'results', 'stage1_train',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    # output_path = os.path.join(config.root_path, 'results', 'fmri_pretrain')
    config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    print('ongoing2, device is ', device)

    # create dataset and dataloader
    # option 1 : HCP dataset
    # dataset_pretrain = hcp_dataset(path=os.path.join(config.root_path, 'data/HCP/npz'), roi=config.roi, patch_size=config.patch_size,
    #             transform=fmri_transform, aug_times=config.aug_times, num_sub_limit=config.num_sub_limit, 
    #             include_kam=config.include_kam, include_hcp=config.include_hcp)
    
    # option 2 : NSD dataset
    dataset_pretrain = NSDDatasetAE(root_dir = 'data/webdataset_avg_split/train',
                                    folder_list=['subj01', 'subj02', 'subj05', 'subj07'])
                                    # folder_list= ['subj01'])
   
    print(f'Dataset size: {len(dataset_pretrain)}')
    sampler = torch.utils.data.DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    dataloader_hcp = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, shuffle=(sampler is None), pin_memory=True, collate_fn=my_collate_ae)
    
    # # Validation Dataset & DataLoader
    # val_dataset = NSDDataset(root_dir = '/data/nsd_dataset/val', folder_list=['subj01', 'subj02', 'subj05', 'subj07'])
    
    # print(f'Validation Dataset size: {len(val_dataset)}')
    # sampler = torch.utils.data.DistributedSampler(val_dataset, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    # val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=sampler, shuffle=None, pin_memory=True, collate_fn=my_collate)


    # create model
    # option 1 :원래 ViT
    # model = MAEforFMRI(num_voxels=dataset_pretrain.num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
    #                 decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, 
    #                 num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
    #                 focus_range=config.focus_range, focus_rate=config.focus_rate, 
    #                 img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss)
    
    # option 2 : ConvMixer 
    model = ConvAE(in_chans = 3, 
               kernel_size = config.kernel_size, 
               patch_size = config.patch_size, embed_dim=config.embed_dim,
                depth=config.depth, decoder_embed_dim=config.decoder_embed_dim, decoder_depth = config.decoder_depth)   
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
    
    # parameter 개수 세기
    number_of_model_parameters = count_parameters(model)
    print(f'Number of Parameters in Model is {number_of_model_parameters}')

    param_groups = optim_factory.param_groups_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    start_time = time.time()
    print('Start Training the fmri AutoEncoder ... ...')
    
    total_loss = []
    val_mse_list = []

    for ep in range(start, config.num_epoch):
        if torch.cuda.device_count() > 1: 
            sampler.set_epoch(ep) # to shuffle the data at every epoch
        loss = train_one_epoch(model, dataloader_hcp, optimizer, device, ep, loss_scaler, logger, config, start_time)
        total_loss.append(loss)

        if (ep % 5 == 0 or ep + 1 == config.num_epoch) :
            print("save_model is OK?")
            # #validation
            # mse = 0
            # total = 0
            # model.eval()
            # with torch.no_grad():
            #     for val_dict in val_dataloader:
            #         voxels = val_dict['voxels'].to(device)
            #         target = val_dict['voxels'].to(device).detach()
                    
            #         output = model(voxels)
            #         total += target.size(0)
            #         mse += F.mse_loss(output, target)
            # mean_mse = mse / total
            # val_mse_list.append(mean_mse)
            # model.train()
            # print(f'MSE is : {mean_mse}')
            
            # save models
            save_model(config, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,'checkpoints'))
            # plot_recon_figures(model, device, dataset_pretrain, output_path, 5, config, logger, model_without_ddp)
    

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
        logger.log('average loss', np.mean(total_loss), step=config.num_epoch-1)
        logger.finish()
    return


@torch.no_grad()
def plot_recon_figures(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Masked Ground-truth')
    axs[0,2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['fmri']
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        sample_with_mask = model_without_ddp.patchify(sample).to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).to('cpu').numpy().reshape(-1)
        sample = sample.to('cpu').numpy().reshape(-1)
        mask = mask.to('cpu').numpy().reshape(-1)
        # cal the cor
        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask,mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)

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
    