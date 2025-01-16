import classifier.utils as ut
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
    
class SubjectClassifier(nn.Module):
    def __init__(self, in_chans, dim, depth, kernel_size, patch_size, n_classes):
        super().__init__()
        
        self.patchifier = nn.Sequential(
            nn.Conv1d(in_chans, dim, kernel_size = patch_size, stride = patch_size),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        
        self.convmixer = ConvMixer(dim, depth, kernel_size)
        
        self.adaptivepool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, n_classes),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        
        x = self.patchifier(x)
        x = self.convmixer(x)
        x = self.adaptivepool(x)
        x = self.classifier(x)
        
        return x
    
class ClassifierEncoder(nn.Module):
    def __init__(self, in_chans, dim, depth, kernel_size, patch_size, n_classes):
        super().__init__()
        
        self.patchifier = nn.Sequential(
            nn.Conv1d(in_chans, dim, kernel_size = patch_size, stride = patch_size),
            nn.GELU(),
            nn.BatchNorm1d(dim)
        )
        
        self.convmixer = ConvMixer(dim, depth, kernel_size)
        
        self.adaptivepool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        
        x = self.patchifier(x)
        x = self.convmixer(x)
        x = self.adaptivepool(x)
        
        return x