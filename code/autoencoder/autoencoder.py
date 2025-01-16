import autoencoder.utils as ut
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixerAutoEncoder(nn.Module):
    """ Masked Autoencoder with ConvMixer backbone
    """
    def __init__(self, in_chans=1, kernel_size = 3, patch_size = 3, embed_dim=512, 
                 depth=4, decoder_embed_dim=512, decoder_depth=8):
        super().__init__()
        
        self.dim_mapping = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size = patch_size, padding = 'same'),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim)
        )

        # --------------------------------------------------------------------------
        # Encoder

        self.encoder_convmixer = ConvMixer(dim = embed_dim, depth = depth, kernel_size=kernel_size)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder specifics
        
        self.encoder2decoder = nn.Conv1d(embed_dim, decoder_embed_dim, kernel_size = kernel_size, padding = 'same')

        self.decoder_convmixer = ConvMixer(dim = decoder_embed_dim, depth = decoder_depth, kernel_size = kernel_size)
        # --------------------------------------------------------------------------
        
        self.decoding = nn.Conv1d(decoder_embed_dim, in_chans, kernel_size = kernel_size, padding = 'same')

        self.embed_dim = embed_dim
   
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        x = self.dim_mapping(imgs)
        x = self.encoder_convmixer(x)
        x = self.encoder2decoder(x)
        x = self.decoder_convmixer(x)
        x = self.decoding(x)

        return x

class fmri_encoder(nn.Module):
    def __init__(self, in_chans = 3, patch_size=16, kernel_size = 7, embed_dim=1024, depth=24):
        super().__init__()
        self.dim_mapping = nn.Sequential(
            nn.Conv1d(in_chans, embed_dim, kernel_size = patch_size, padding = 'same'),
            nn.GELU(),
            nn.BatchNorm1d(embed_dim)
        )

        # --------------------------------------------------------------------------
        # Encoder

        self.encoder_convmixer = ConvMixer(dim = embed_dim, depth = depth, kernel_size=kernel_size)
        # --------------------------------------------------------------------------
        
        self.encoder2decoder = nn.Conv1d(embed_dim, 512, kernel_size = kernel_size, padding = 'same')
        
        self.aggregate = nn.AdaptiveAvgPool1d(1)
        
        self.embed_dim = embed_dim
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        latent = self.dim_mapping(imgs)
        latent = self.encoder_convmixer(latent)
        latent = self.encoder2decoder(latent)
        latent = self.aggregate(latent)
        return latent
    
    def load_checkpoint(self, state_dict):
        state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 

    
class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9):
        super().__init__()
        self.convmixer =  nn.Sequential(
                            *[nn.Sequential(
                                    Residual(nn.Sequential(
                                        nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"),
                                        nn.GELU(),
                                        nn.BatchNorm1d(dim)
                                    )),
                                    nn.Conv1d(dim, dim, kernel_size=1, padding="same"),
                                    nn.GELU(),
                                    nn.BatchNorm1d(dim)
                            ) for i in range(depth)]
                        )
        
    def forward(self, x):
        return self.convmixer(x)


