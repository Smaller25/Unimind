import math, sys
import torch
import autoencoder.utils as ut
from torch import inf
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from losses import ContrastiveLoss
from copy import deepcopy
from autoencoder.models import Clipper

def cut_string(x, prompt):
    x = x.replace(prompt, '')
    x = x.replace('USER:  \n\nASSISTANT: ','')
    return x

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

# Siamese Trainer
def train_one_epoch_siamese(model, clip_extractor, data_loader, optimizer, device, epoch, 
                        loss_scaler,log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                        img_feature_extractor=None, preprocess=None):
    model.train(True)
    optimizer.zero_grad()
    CosineSim = nn.CosineSimilarity()
    loss_f = ContrastiveLoss(margin = 1.)
    # loss_f = torch.nn.BCEWithLogitsLoss()

    clip_extractor.eval()
    # print("out_dim:",out_dim)
    
    
    total_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    
    
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples1, samples2, images1, images2 = data_dcit['fmri1'], data_dcit['fmri2'], data_dcit['image1'], data_dcit['image2']
        samples1 = samples1.to(device)
        samples2 = samples2.to(device)
        images1 = images1.to(device)
        images2 = images2.to(device)
        

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            embedding1, embedding2 = model(samples1), model(samples2)
            b = embedding1.shape[0]
            dis = nn.functional.pairwise_distance(embedding1, embedding2).sum(1).reshape(b, -1)
            
            clip_embedding1 = clip_extractor.embed_image(images1).float()
            clip_embedding2 = clip_extractor.embed_image(images2).float()
            
            sim = CosineSim(clip_embedding1, clip_embedding2)
            not_sim = torch.ones_like(sim) - sim
            target = torch.stack([sim, not_sim], dim=-1)
            # target = sim.reshape(b, -1).detach()
            # print(target)
            # print(dis)
            loss = loss_f(embedding1.reshape(b, -1), embedding2.reshape(b, -1), target)
            # loss = loss_f(dis, target)

        loss_value = loss.item()        

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        optimizer.zero_grad()

        total_loss.append(loss_value)
        # total_cor.append(cor)

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        # log_writer.log('cor', np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_loss)


def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                        loss_scaler,log_writer=None, config=None, start_time=None):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    accum_iter = config.accum_iter
    for data_iter_step, (voxels) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = voxels
        
        samples = samples.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            pred = model(samples)
            loss = F.mse_loss(pred, samples.detach())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        optimizer.zero_grad()

        total_loss.append(loss_value)

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_loss)


def train_one_epoch_finetune(model, data_loader, optimizer, device, epoch, 
                        loss_scaler,log_writer=None, config=None, start_time=None):
    # Clipper
    print('Creating Clipper...')
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    clip_variant = "ViT-B/32"
    clip_size = clip_sizes[clip_variant]
    hidden = ''
    norm_embs = True
    if hidden:
        # print("Using hidden layer CLIP space (Versatile Diffusion)")
        if not norm_embs:
            # print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")
            pass
        clip_extractor = Clipper(clip_variant, device=device, hidden_state=True, norm_embs=norm_embs)
        out_dim = 257 * clip_size
    else:
        # print("Using final layer CLIP space (Stable Diffusion Img Variations)")
        if norm_embs:
            # print("WARNING: YOU WANT UN-NORMED EMBEDDINGS FOR IMG VARIATIONS!")
            pass
        clip_extractor = Clipper(clip_variant, device=device, hidden_state=False, norm_embs=norm_embs)
        out_dim = clip_size
    clip_extractor = clip_extractor.to(device).eval()
    
    model = model.to(device)
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    accum_iter = config.accum_iter
    for data_iter_step, (samples) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        voxels = samples['fmri'].to(device)
        voxels = voxels.float()
        image = samples['image'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            pred = model(voxels)
            with torch.no_grad():
                target = clip_extractor.embed_image(image.float()).float()
            pred = pred.reshape(target.shape)
            loss = F.mse_loss(pred, target.detach())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        optimizer.zero_grad()

        total_loss.append(loss_value)

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_loss)


def train_one_epoch_finetune_s_p(model, data_loader, optimizer, device, epoch, 
                        loss_scaler,log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                        img_feature_extractor=None, preprocess=None):
    model.train(True)
    optimizer.zero_grad()
    CosineSim = nn.CosineSimilarity()
    loss_f = ContrastiveLoss(margin = 1.)
    # loss_f = torch.nn.BCEWithLogitsLoss()

    #---------------------------------------------------------------------------
    # Clipper
    print('Creating Clipper...')
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    clip_variant = "ViT-B/32"
    clip_size = clip_sizes[clip_variant]
    hidden = ''
    norm_embs = True
    if hidden:
        # print("Using hidden layer CLIP space (Versatile Diffusion)")
        if not norm_embs:
            # print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")
            pass
        clip_extractor = Clipper(clip_variant, device=device, hidden_state=True, norm_embs=norm_embs)
        out_dim = 257 * clip_size
    else:
        # print("Using final layer CLIP space (Stable Diffusion Img Variations)")
        if norm_embs:
            # print("WARNING: YOU WANT UN-NORMED EMBEDDINGS FOR IMG VARIATIONS!")
            pass
        clip_extractor = Clipper(clip_variant, device=device, hidden_state=False, norm_embs=norm_embs)
        out_dim = clip_size
    clip_extractor = clip_extractor.to(device).eval()
    # print("out_dim:",out_dim)

    #-------------------------------------------------------------------------------
    # Loss 계산
    total_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples1, samples2, images1, images2 = data_dcit['fmri1'], data_dcit['fmri2'], data_dcit['image1'], data_dcit['image2']
        samples1 = samples1.to(device)
        samples2 = samples2.to(device)
        images1 = images1.to(device)
        images2 = images2.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            embedding1, embedding2 = model(samples1), model(samples2)
            b = embedding1.shape[0]
            dis = nn.functional.pairwise_distance(embedding1, embedding2).sum(1).reshape(b, -1)
            
            clip_embedding1 = clip_extractor.embed_image(images1).float()
            clip_embedding2 = clip_extractor.embed_image(images2).float()
            
            sim = CosineSim(clip_embedding1, clip_embedding2)
            not_sim = torch.ones_like(sim) - sim
            target = torch.stack([sim, not_sim], dim=-1)
            # target = sim.reshape(b, -1).detach()
            # print(target)
            # print(dis)

            contrastive_loss = loss_f(embedding1.reshape(b, -1), embedding2.reshape(b, -1), target)
            # print('image_embedding', clip_embedding1.shape)
            # print('fmri_embedding', embedding1.shape)
            projection_loss = F.mse_loss(clip_embedding1, embedding1.squeeze()) + F.mse_loss(clip_embedding2, embedding2.squeeze())

        loss = 0.5 * contrastive_loss + 0.5 * projection_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        loss_scaler(contrastive_loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        optimizer.zero_grad()

        total_loss.append(loss_value)
        # total_cor.append(cor)

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        # log_writer.log('cor', np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_loss)
