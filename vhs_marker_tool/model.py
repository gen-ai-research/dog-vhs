import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.io import loadmat, savemat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn import L1Loss, CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast

import torchvision
import torchvision.transforms as T
import torchvision.models as models
from collections import Counter
from torchvision import models
import os
import sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange

sys.path.append(os.path.join(os.path.dirname(__file__), 'mamba'))


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#################### START : MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# Mamba Custom Model
# =========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange

class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.d_inner = dim * expand

        self.in_proj = nn.Linear(dim, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, dim)

        self.selective_scan = selective_scan_fn
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # (B, HW, C)
        x = self.in_proj(x)
        x = x.transpose(1, 2).view(B, self.d_inner, H, W)
        x = self.out_proj(x.view(B, self.d_inner, -1).transpose(1, 2))
        x = x.transpose(1, 2).view(B, self.dim, H, W)
        return x

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.down(x)

class MambaStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

class MambaStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2):
        super().__init__()
        self.down = Downsample(in_ch, out_ch)
        self.blocks = nn.Sequential(*[nn.Sequential(ResidualBlock(out_ch), MambaBlock(out_ch)) for _ in range(depth)])
        self.se = SELayer(out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        x = self.se(x)
        return x

class MambaVHS(nn.Module):
    def __init__(self, in_ch=3, num_points=12):
        super().__init__()
        self.stem = MambaStem(in_ch, 64)
        self.stage1 = MambaStage(64, 128, depth=2)
        self.stage2 = MambaStage(128, 256, depth=3)
        self.stage3 = MambaStage(256, 512, depth=3)
        self.stage4 = MambaStage(512, 640, depth=3)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(640, 384),
            nn.ReLU(),
            nn.Linear(384, num_points)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.regressor(x).view(x.size(0), -1)
        

# Load EfficientNet model and modify its classifier
def get_model(device):
    model = MambaVHS()

    # 2. Load the checkpoint
    checkpoint_path = '20250506_204657/models/bm_43.pth'

    # 3. Load model state dict
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    return model.to(device)
   

# Function to calculate VHS from model outputs
def calc_vhs(x: torch.Tensor):
    A = x[..., 0:2]
    B = x[..., 2:4]
    C = x[..., 4:6]
    D = x[..., 6:8]
    E = x[..., 8:10]
    F = x[..., 10:12]
    
    AB = torch.norm(A - B, p=2, dim=-1)
    CD = torch.norm(C - D, p=2, dim=-1)
    EF = torch.norm(E - F, p=2, dim=-1)
    
    vhs = 6 * (AB + CD) / EF
    
    return vhs
