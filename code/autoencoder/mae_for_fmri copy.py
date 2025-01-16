import sc_mbm.utils as ut
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

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, num_voxels=224, in_chans=1, dim = 768, patch_size = 7):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x, **kwargs):
        x = self.proj(x)
        x = self.gelu(x)
        x = self.bn(x)
        return x

class MAEforFMRI(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, num_voxels=224, patch_size=16, kernel_size = 9, embed_dim=1024, in_chans=1,
                 depth=24, decoder_embed_dim=512, 
                 decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio = 4., norm_layer=nn.LayerNorm,
                 focus_range=None, focus_rate=None, img_recon_weight=1.0, 
                 use_nature_img_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed1D(num_voxels, in_chans, embed_dim, patch_size)

        num_patches = self.patch_embed.num_patches

        self.encoder_convmixer = ConvMixer(dim = embed_dim, depth = depth, kernel_size=kernel_size)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        # self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1))
        
        self.encoder2decoder = nn.Conv1d(embed_dim, decoder_embed_dim, kernel_size = 1, padding = 'same')

        self.decoder_convmixer = ConvMixer(dim = decoder_embed_dim, depth = decoder_depth, kernel_size = kernel_size)
        
        self.decoder_last_layer = nn.Conv1d(decoder_embed_dim, self.patch_embed.patch_size, kernel_size=1, padding='same')
        # --------------------------------------------------------------------------

        # nature image decoder specifics
        if use_nature_img_loss:
            self.nature_img_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.nature_img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.nature_img_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

            self.nature_img_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(2)])

            self.nature_img_decoder_norm = norm_layer(decoder_embed_dim)
            self.nature_img_decoder_pred = nn.Sequential(
                nn.Conv1d(num_patches, 512, kernel_size=1, stride=1, bias=True),
                nn.Linear(decoder_embed_dim, 28*28, bias=True)
            )
            # --------------------------------------------------------------------------

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.focus_range = focus_range
        self.focus_rate = focus_rate
        self.img_recon_weight = img_recon_weight
        self.use_nature_img_loss = use_nature_img_loss
   
        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
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
    
    
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x.transpose(1, 2).contiguous()

    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[2]
        
        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, D, L], sequence
        """
        N, D, L = x.shape  # batch, dim, length
        len_keep = int(L * (1 - mask_ratio))

        if self.focus_range is not None:
            len_mask = L - len_keep
            weights = [1-self.focus_rate] * L
            weights[self.focus_range[0] // self.patch_size : self.focus_range[1] // self.patch_size
                        ] = [self.focus_rate] * (self.focus_range[1] // self.patch_size - self.focus_range[0] // self.patch_size)
            weights = torch.tensor(weights).repeat(N, 1).to(x.device)
            ids_mask = torch.multinomial(weights, len_mask, replacement=False)
            
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        if self.focus_range is not None:
            for i in range(N):
                noise[i, ids_mask[i,:]] = 1.1  # set mask portion to 1.1 

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(1).repeat(1, D, 1))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        x = self.encoder_convmixer(x)

        return x #, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.encoder2decoder(x)
        
        # append mask tokens to sequence
        # mask_tokens = self.mask_token.repeat(x.shape[0], 1, ids_restore.shape[1] - x.shape[2])
        # x_ = torch.cat([x, mask_tokens], dim=2)
        # x = torch.gather(x_, dim=2, index=ids_restore.unsqueeze(1).repeat(1, x.shape[1], 1))  # unshuffle

        # apply Transformer blocks
        x = self.decoder_convmixer(x)
        x = self.decoder_last_layer(x)
        
        return x

    def forward_nature_img_decoder(self, x, ids_restore):
        # embed tokens
        x = self.nature_img_decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.nature_img_mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.nature_img_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.nature_img_decoder_blocks:
            x = blk(x)
        x = self.nature_img_decoder_norm(x)
        # remove cls token
        x = x[:, 1:, :]
        # predictor projection
        # x = x.mean(dim=1, keepdim=True)
        x = self.nature_img_decoder_pred(x)
        x = x.view(x.shape[0], 512, 28, 28)

        return x # n, 512, 28, 28
        
    def forward_nature_img_loss(self, inputs, reconstructions):
        loss = ((torch.tanh(inputs) - torch.tanh(reconstructions))**2).mean()
        if torch.isnan(reconstructions).sum():
            print('nan in reconstructions')
        if torch.isnan(inputs).sum():
            print('nan in inputs')
    
        return loss   

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss = F.mse_loss(pred, target)
        
        # loss = (pred - target) ** 2
        # loss = loss.transpose(1,2).contiguous()
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches
        return loss

    def forward(self, imgs, img_features=None, valid_idx=None, mask_ratio=0.75):
        # latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        mask = 0
        ids_restore = 0
        latent = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p]
        loss = self.forward_loss(imgs, pred, mask)

        if self.use_nature_img_loss and img_features is not None:
            # valid_idx = torch.nonzero(nature_image.sum(dim=(1,2,3)) != 0).squeeze(1)
            if len(valid_idx) != 0:
                nature_image_recon = self.forward_nature_img_decoder(latent[valid_idx], ids_restore[valid_idx])
                loss_nature_image_recon = self.forward_nature_img_loss(img_features, nature_image_recon)
                if torch.isnan(loss_nature_image_recon).sum():
                    print(loss_nature_image_recon)
                    print("loss_nature_image_recon is nan")
                    
                loss = loss + self.img_recon_weight*loss_nature_image_recon

        return loss, pred, mask



class fmri_encoder(nn.Module):
    def __init__(self, num_voxels=224, patch_size=16, kernel_size = 7, embed_dim=1024, in_chans=1, depth=24):
        super().__init__()
        self.patch_embed = PatchEmbed1D(num_voxels, in_chans, embed_dim, patch_size)

        num_patches = self.patch_embed.num_patches

        self.encoder_convmixer = ConvMixer(embed_dim, depth, kernel_size)
    
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
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

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)       
        x = self.encoder_convmixer(x)
        return x  

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 

    
class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, mode = 1):
        super().__init__()
        g = dim if mode == 1 else 1
        self.convmixer =  nn.Sequential(
                            *[nn.Sequential(
                                    Residual(nn.Sequential(
                                        nn.Conv1d(dim, dim, kernel_size, groups=g, padding="same"),
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