import os
import torch
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# Mamba Custom Model
# =========================

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer
    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the channel dimension. 
    """ 
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
    """
    Residual Block with Conv2d, BatchNorm2d, and SiLU activation.   
    Args:
        dim (int): Number of input channels.
    """
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
    """
    Mamba Block with selective scan and linear projections.
    Args:
        dim (int): Number of input channels.
        d_state (int): Dimension of the state.  
        expand (int): Expansion factor for the inner dimension.
    """ 
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
    """
    Downsample block with Conv2d, BatchNorm2d, and SiLU activation. 
    Args:   
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """
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
    """
    Stem block for the Mamba model.
    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """ 
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
    """ 
    Mamba Stage block with downsampling and multiple Mamba blocks.
    Args:   
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        depth (int): Number of Mamba blocks in the stage.
    """
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
    """
    MambaVHS model for predicting VHS scores.   
    Args:
        in_ch (int): Number of input channels.
        num_points (int): Number of points for the model.
    """ 
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
    

def get_model(device, num_points=12,checkpoint_path=None):
    """     
    Get MambaVHS model with specified number of points.
    Args:
        num_points (int): Number of points for the model.
        checkpoint_path (str): Path to the checkpoint file.
    Returns:
        model (MambaVHS): MambaVHS model instance.
    """
    model = MambaVHS(num_points=num_points)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("No checkpoint found, using random initialization.")  
    
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
