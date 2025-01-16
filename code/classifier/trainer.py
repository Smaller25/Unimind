import math, sys
import torch
import classifier.utils as ut
from torch import inf
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from losses import ContrastiveLoss

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

def train_one_epoch_classifier(model, data_loader, optimizer, device, epoch, 
                        loss_scaler,log_writer=None, config=None, start_time=None, model_without_ddp=None):
    model.train(True)
    total_loss = []
    val_accuracy = []
    accum_iter = config.accum_iter
    for data_iter_step, (data_list) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = data_list['voxels']
        subject_num = data_list['subj']
        
        samples = samples.to(device)
        subject_num = subject_num.type(torch.LongTensor)
        subject_num = subject_num.to(device)
        # img_features = img_features.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            optimizer.zero_grad()
            output = model(samples)
            loss = nn.CrossEntropyLoss()(output, subject_num.detach())
            loss.backward()
            optimizer.step()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # loss /= accum_iter
        # loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        # if (data_iter_step + 1) % accum_iter == 0:

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