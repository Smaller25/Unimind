import os
import numpy as np

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 

class Config_Classifier(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.lr = 5e-4
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 200
        self.warmup_epochs = 40
        self.batch_size = 64
        self.clip_grad = 0.8
        
        # Model Parameters
        self.in_chans = 3
        self.patch_size = 3
        self.kernel_size = 3
        self.dim = 512
        self.depth = 4
        self.n_classes = 4

        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.seed = 2024
        self.roi = 'VC'
        self.aug_times = 1
        self.accum_iter = 1

        # distributed training
        self.local_rank = 0
        
        self.resume = ''

class Config_MBM_fMRI(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
    # MAE for fMRI
        # Training Parameters
        self.in_chans = 3
        
        self.lr = 1e-3
        self.min_lr = 0.
        self.weight_decay = 0.05
        self.num_epoch = 100
        self.warmup_epochs = 40
        self.batch_size = 16
        self.clip_grad = 0.8
        
        # Model Parameters
        self.patch_size = 3
        self.kernel_size = 3
        self.embed_dim = 512
        self.decoder_embed_dim = 512
        self.depth = 4
        self.decoder_depth = 8

        # Project setting
        self.root_path = '.'
        self.output_path = self.root_path
        self.seed = 2024
        self.roi = 'VC'
        self.aug_times = 1
        self.accum_iter = 1
        
        self.img_size = 256
        self.crop_ratio = 0.2

        # distributed training
        self.local_rank = 0
        
        self.resume = ''
        
        self.pretrain_encoder_path = '/home/smaller/mind_vis_contrastive/results/stage1_train/06-11-2024-15-46-14/checkpoints/checkpoint_20.pth'
        
class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.in_chans = 3
        
        self.seed = 2024
        self.root_path = '.'
        self.kam_path = os.path.join(self.root_path, 'data/Kamitani/npz')
        self.bold5000_path = os.path.join(self.root_path, 'data/BOLD5000')
        self.roi = 'VC'
        self.patch_size = 3
        
        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/label2img')

        self.pretrain_encoder_path = '/home/smaller/mind_vis_contrastive/results/stage2_finetune/12-11-2024-15-44-34/checkpoints/checkpoint_45.pth'
        self.pretrain_classifier_path = '/home/smaller/mind_vis_contrastive/results/classifier/16-11-2024-22-53-59/checkpoints/checkpoint_6.pth'
        
        self.train_data_root = '/home/smaller/mind_vis_contrastive/data/webdataset_avg_split/train'
        self.test_data_root = '/home/smaller/mind_vis_contrastive/data/webdataset_avg_split/test'
        
        self.dataset = 'NSD' # GOD or BOLD5000
        self.kam_subs = ['sbj_3']
        self.bold5000_subs = ['CSI1']
        self.pretrain_mbm_path = os.path.join(self.root_path, f'pretrains/{self.dataset}/fmri_encoder.pth') 

        self.img_size = 256

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 2
        self.lr = 1e-4
        self.num_epoch = 500
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = False
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None # os.path.join(self.root_path, 'results/generation/25-08-2022-08:02:55/checkpoint.pth')
        
        # Model Parameters
        self.mask_ratio = 0.0
        self.patch_size = 3
        self.kernel_size = 3
        self.embed_dim = 64 # has to be a multiple of num_heads
        self.decoder_embed_dim = 64
        self.depth = 16
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.decoder_depth = 8
        self.mlp_ratio = 1.0
        
        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6
        
