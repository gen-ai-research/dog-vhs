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

model = MambaVHS()

# 2. Load the checkpoint
checkpoint_path = '20250506_204657/models/bm_43.pth'

# 3. Load model state dict
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

model = model.to(device)

#################### END : MODEL
# Check the code carefully!
class DogHeartDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = [filepath for filepath in list(sorted(os.listdir(os.path.join(root, "Images")))) 
                     if filepath.endswith('png') or filepath.endswith('jpg')]
        self.points = list(sorted(os.listdir(os.path.join(root, "Labels"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        points_path = os.path.join(self.root, "Labels", self.points[idx])
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if self.transforms is not None:
            img = self.transforms(img)
        h_new, w_new = img.shape[1], img.shape[2]
        mat = loadmat(points_path)
        six_points = mat['six_points'].astype(float)
        six_points = torch.as_tensor(six_points, dtype=torch.float32)
        six_points[:,0] = w_new / w * six_points[:,0] # Resize image to any size and maintain original points
        six_points[:,1] = h_new / h * six_points[:,1]
        six_points = torch.reshape(six_points, (-1,))/h_new # Normlize the points
        # six_points = torch.reshape(six_points, (-1,))
        VHS = mat['VHS'].astype(float)
        VHS = torch.as_tensor(VHS, dtype=torch.float32).reshape([1,1])
        return idx, img, six_points, VHS

    def __len__(self):
        return len(self.imgs)
    
def get_transform(resized_image_size):
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Resize((resized_image_size, resized_image_size)))
    transforms.append(T.Normalize(mean = [0.485,0.456,0.406], std = [0.229, 0.224, 0.225]))
    return T.Compose(transforms)

resized_image_size = 512
true_batch_size = 256
accumulation_steps = 8
root_db_folder= "../data"

dataset_train = DogHeartDataset(f'{root_db_folder}/Train', get_transform(resized_image_size))
dataset_valid = DogHeartDataset(f'{root_db_folder}/Valid', get_transform(resized_image_size))
train_loader = DataLoader(dataset_train, batch_size=true_batch_size//accumulation_steps, shuffle=True, num_workers=8) 
valid_loader = DataLoader(dataset_valid, batch_size=true_batch_size//accumulation_steps, shuffle=False, num_workers=8)

class DogHeartTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = [filepath for filepath in list(sorted(os.listdir(os.path.join(root, "Images")))) 
                     if filepath.endswith('png') or filepath.endswith('jpg')]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

test_dataset = DogHeartDataset(f'{root_db_folder}/Test_Images', get_transform(resized_image_size))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

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

################################
## Utilites
################################
 

import os
from datetime import datetime

# Get the current timestamp
current_timestamp = datetime.now()

# Create folder name (YYYYMMDD)
folder_name = current_timestamp.strftime("%Y%m%d_%H%M%S")

# Ensure the folder exists
os.makedirs(folder_name, exist_ok=True)
os.makedirs(f"{folder_name}/predictions", exist_ok=True)
os.makedirs(f"{folder_name}/models", exist_ok=True)

predictions_folder = f"{folder_name}/predictions"
models_folder = f"{folder_name}/models"
#os.makedirs("models", exist_ok=True)  # Ensure models directory exists

# Define file names
log_file_name = os.path.join(folder_name, f"training_log.txt")
loss_file_name = os.path.join(folder_name, f"loss.csv")
vhs_score_file = os.path.join(folder_name, f"vhs_score.csv")
best_val_model = os.path.join(folder_name, f"best_model.pth")
last_model = os.path.join(folder_name, f"last_model.pth")

# Print paths for confirmation
print("current_folder",folder_name)

import logging
# logging.basicConfig(filename=log_file_name, level=logger.info, format="%(asctime)s - %(message)s")
# --- Create a logger ---
logger = logging.getLogger('dual_logger')
logger.setLevel(logging.DEBUG)  # Capture all levels

class InfoOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

# --- File handler for general info logs ---
info_handler = logging.FileHandler(f'{folder_name}/info_log.txt')
info_handler.setLevel(logging.INFO)
info_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(info_formatter)
info_handler.addFilter(InfoOnlyFilter())

# --- File handler for warnings/errors only ---
warn_handler = logging.FileHandler(f'{folder_name}/warning_log.txt')
warn_handler.setLevel(logging.WARNING)
warn_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
warn_handler.setFormatter(warn_formatter)

# --- Add handlers to the logger ---
logger.addHandler(info_handler)
logger.addHandler(warn_handler)

# Get script path and current date-time
script_path = os.path.abspath(__file__)
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pid = os.getpid()

# Log script path with date and time
logger.info(f"Script started: {script_path} at {current_time}, PID : {pid}")
logger.warning(f"Script started: {script_path} at {current_time}, PID : {pid}")

# Print number of samples in each dataset
logger.warning(f"Number of samples in the training dataset: {len(dataset_train)}")
logger.warning(f"Number of samples in the validation dataset: {len(dataset_valid)}")
logger.warning(f"Number of samples in the test dataset: {len(test_dataset)}")

# Print number of batches in each DataLoader
logger.warning(f"Number of batches in the training DataLoader: {len(train_loader)}")
logger.warning(f"Number of batches in the validation DataLoader: {len(valid_loader)}")
logger.warning(f"Number of batches in the test DataLoader: {len(test_loader)}")


num_epochs = 500

acc = 0
best_loss=0.0

# Training loop

def generate_predictions(model, test_loader, calc_vhs, device, folder_name, iter,epoch):
    model.eval()
    img_names = []
    vhs_pred_list = []
    
    # Create predictions directory if it doesn't exist
    os.makedirs(f'{folder_name}/predictions', exist_ok=True)
    
    with torch.no_grad():
        for inputs, img_name in test_loader:
            inputs = inputs.to(device)
            img_names += list(img_name)
            outputs = model(inputs)
            vhs_pred = calc_vhs(outputs)
            vhs_pred_list += list(vhs_pred.detach().cpu().numpy())
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({
            'ImageName': img_names,
            'VHS': vhs_pred_list
        })
        df.to_csv(f'{folder_name}/predictions/predictions_epoch_{iter+1}_{epoch+1}.csv', 
                 index=False, header=False)
    
    print(f"Predictions for epoch {epoch+1} saved to {folder_name}/predictions/")

from tqdm import tqdm

def get_labels(vhs):
    """
    Assigns class labels based on VHS values using a compact formula.

    Args:
        vhs (torch.Tensor): Tensor containing VHS scores.
        
    Returns:
        torch.Tensor: Tensor of class labels (0, 1, 2).
    """
    return ((vhs >= 10).long() - (vhs < 8.2).long() + 1).squeeze()

from sklearn.metrics import classification_report

#---------------
# Psuedo Labeling
#----------------

##---------------------- CLASS WEIGHTS ------------------
def compute_class_weights(loader, device, save_path="class_weights.pt"):
    if os.path.exists(save_path):
        logger.info(f"[INFO] Found existing class weights at {save_path}. Loading...")
        return torch.load(save_path,weights_only=True).to(device)
    
    logger.info(f"[INFO] Class weights not found. Computing from dataset...")
    label_counts = Counter()

    for _, _, _, vhs in tqdm(loader, desc="Counting class distribution"):
        vhs = vhs.to(device)
        labels = get_labels(vhs.squeeze()).cpu().tolist()
        label_counts.update(labels)

    total_classes = max(label_counts.keys()) + 1
    counts = torch.tensor([label_counts.get(i, 0) for i in range(total_classes)], dtype=torch.float)

    weights = 1.0 / counts
    weights = weights / weights.sum()  # Normalize

    torch.save(weights, save_path)
    logger.info(f"[INFO] Class weights computed and saved to {save_path}")
    return weights.to(device)

# Train loader example
train_class_weights = compute_class_weights(train_loader, device, "train_class_weights.pt")

# Validation loader (optional, for inspection only)
val_class_weights = compute_class_weights(valid_loader, device, "val_class_weights.pt")

##------------------- CLASS WEIGHTS --------------
#-------------------------- VHS LOSS ------------------
class VHSAwareLoss(nn.Module):
    def __init__(self, class_weights=None, margin=0.05, middle_class_multiplier=2.0):
        super().__init__()
        self.margin = margin
        self.middle_class_multiplier = middle_class_multiplier
        self.l1 = nn.L1Loss()
        self.class_weights = class_weights if class_weights is not None else torch.tensor([1.0, 1.0, 1.0])

    def forward(self, vhs_pred, vhs_true):
        # Get class labels
        true_class = get_labels(vhs_true)
        pred_class = get_labels(vhs_pred)

        # Base L1 loss
        l1_loss = self.l1(vhs_pred.squeeze(), vhs_true.squeeze())

        # Class mismatch penalty
        mismatch_penalty = (pred_class != true_class).float()

        # Distance from class boundary (for soft margin)
        soft_penalty = torch.zeros_like(vhs_pred)

        # Class 0: VHS < 8.2
        mask_0 = true_class == 0
        soft_penalty[mask_0] = F.relu(vhs_pred[mask_0] - (8.2 + self.margin))

        # Class 1: 8.2 ≤ VHS < 10 (apply tighter margin and boost)
        mask_1 = true_class == 1
        tighter_margin = self.margin / self.middle_class_multiplier
        soft_1 = F.relu(8.2 - vhs_pred[mask_1]) + F.relu(vhs_pred[mask_1] - (10 + tighter_margin))
        soft_penalty[mask_1] = self.middle_class_multiplier * soft_1

        # Class 2: VHS ≥ 10
        mask_2 = true_class == 2
        soft_penalty[mask_2] = F.relu((10 - self.margin) - vhs_pred[mask_2])

        # Weight by class imbalance
        weight = self.class_weights[true_class]
        total_penalty = weight * (mismatch_penalty + soft_penalty)

        return l1_loss + total_penalty.mean()

#------------------------- END : VHS LOSS

vhs_valid_loss = VHSAwareLoss(class_weights=val_class_weights)  
vhs_train_loss = VHSAwareLoss(class_weights=train_class_weights)

#------------------- VALIDATE ---------------------------

def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_kpt_loss = 0.0
    total_cls_loss = 0.0
    total_ef_loss = 0.0
    val_correct = 0

    #ce_loss = nn.CrossEntropyLoss()
    #ce_loss = FocalLoss()
    #ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    l1_loss = nn.L1Loss()

    # Collect all labels and predictions
    all_true_labels = []
    all_pred_labels = []

    val_progress = tqdm(loader, desc="Validating", leave=False, unit="batch")

    with torch.no_grad():
        for batch_idx, (idx, images, points, vhs) in enumerate(val_progress):
            images = images.to(device)
            points = points.to(device)
            vhs = vhs.to(device)

            outputs = model(images).squeeze()
            pred_vhs = calc_vhs(outputs).squeeze().to(device)

            # --- Loss 1: Keypoint L1 Loss ---
            loss_kpt = vhs_valid_loss(outputs, points)

            # --- Loss 2: VHS classification ---
            loss_cls = vhs_valid_loss(pred_vhs,vhs)

            # --- Loss 3: EF line L1 loss ---
            # EF_true = points[:, 10:12] - points[:, 8:10]
            # EF_pred = outputs[:, 10:12] - outputs[:, 8:10]
            # loss_ef = l1_loss(EF_pred, EF_true)

            # --- Total Combined Loss ---
            #loss = 10 * loss_kpt + 0.1 * loss_cls + 0.05 * loss_ef
            loss = 10 * loss_kpt + 0.1 * loss_cls 

            total_loss += loss.item() * images.size(0)
            total_kpt_loss += loss_kpt.item() * images.size(0)
            total_cls_loss += loss_cls.item() * images.size(0)
            
            # Accuracy
            pred_class = get_labels(pred_vhs).cpu().numpy()
            true_class = get_labels(vhs).cpu().numpy()

            all_true_labels.extend(true_class)
            all_pred_labels.extend(pred_class)

            val_correct += (pred_class == true_class).sum()

            val_progress.set_postfix({
                "Val Loss": f"{(total_loss / ((batch_idx + 1) * images.size(0))):.4f}"
            })

    val_progress.close()

    avg_loss = total_loss / len(loader.dataset)
    avg_kpt_loss = total_kpt_loss / len(loader.dataset)
    avg_cls_loss = total_cls_loss / len(loader.dataset)
    #avg_ef_loss = total_ef_loss / len(loader.dataset)
    val_acc = val_correct / len(loader.dataset)

    logger.info(f"Validation Loss: {avg_loss:.4f} | "
                 f"Keypoint Loss: {avg_kpt_loss:.4f} | "
                 f"Classification Loss: {avg_cls_loss:.4f} | "
                 #f"EF Loss: {avg_ef_loss:.4f} | "
                 f"Accuracy: {val_acc:.4f}")
    
    # At the end of validation
    class_names = ['< 8.2', '8.2 - 10', '>= 10']
    report = classification_report(all_true_labels, all_pred_labels, target_names=class_names, digits=4)
    logger.warning(f"\nClassification Report:\n{report}")

    return avg_loss

###################### TEST ACCURACY ########################
def test_accuracy(model, test_loader, device, calc_vhs, get_labels, logger=None):
    model.eval()
    total_correct = 0
    total_samples = 0

    all_true_labels = []
    all_pred_labels = []

    progress = tqdm(test_loader, desc="Testing", leave=False, unit="batch")

    with torch.no_grad():
        for batch_idx, (idx, images, _, vhs) in enumerate(progress):
            images = images.to(device)
            vhs = vhs.to(device)

            outputs = model(images).squeeze()
            pred_vhs = calc_vhs(outputs).squeeze()

            pred_classes = get_labels(pred_vhs).cpu().numpy()
            true_classes = get_labels(vhs).cpu().numpy()

            total_correct += (pred_classes == true_classes).sum()
            total_samples += len(true_classes)

            all_true_labels.extend(true_classes)
            all_pred_labels.extend(pred_classes)

            progress.set_postfix({
                "Accuracy": f"{(total_correct / total_samples):.4f}"
            })

    progress.close()
    accuracy = total_correct / total_samples

    class_names = ['< 8.2', '8.2 - 10', '>= 10']
    #report = classification_report(all_true_labels, all_pred_labels, target_names=class_names, digits=4)
    
    if logger:
        logger.info(f"Test Accuracy: {accuracy:.4f}")
     #   logger.warning(f"\nClassification Report:\n{report}")
    else:
        print(f"Test Accuracy: {accuracy:.4f}")
      #  print(f"\nClassification Report:\n{report}")

    return accuracy

################# TRAINING PROCESS ##############
def train_model(model, train_loader, valid_loader, test_loader, 
                folder_name, calc_vhs, device, num_epochs=100, accumulation_steps=4):
    
    
    # Main loss functions
    l1_loss = L1Loss()
    #ce_loss = CrossEntropyLoss()
    #ce_loss = FocalLoss()
    #ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_loss = float('inf')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-7)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,eta_min=1e-6)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)    


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_loss_cls = 0.0
        running_loss_kpt = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_progress):
            idx, images, points, vhs = batch

            images = images.to(device)
            points = points.to(device)
            vhs = vhs.to(device)

            # Forward pass
            outputs = model(images).squeeze()
            pred_vhs = calc_vhs(outputs).squeeze().to(device)

            # --- Loss 1: Keypoint L1 Loss ---
            loss_kpt = vhs_train_loss(outputs, points)

            # # --- Loss 2: VHS regression ---
            # loss_vhs = l1_loss(pred_vhs, vhs.squeeze())

            # --- Loss 3: Classification Loss from VHS ---
            loss_cls = vhs_train_loss(pred_vhs,vhs)

            # # --- Loss 4: EF line L1 loss ---
            # EF_true = points[:, 10:12] - points[:, 8:10]
            # EF_pred = outputs[:, 10:12] - outputs[:, 8:10]
            # loss_EF = l1_loss(EF_pred, EF_true)

            # --- Total Combined Loss ---
            #loss = 10 * loss_kpt + loss_cls * 0.1 + 0.05 * loss_EF
            loss =10 * loss_kpt + loss_cls * 0.1 
            loss.backward()

            running_loss_kpt+=loss_kpt
            running_loss_cls+=loss_cls

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            train_progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Avg Loss": f"{running_loss / (batch_idx + 1):.4f}"
            })

        train_progress.close()

        # --- Validation ---
        val_loss = validate(model, valid_loader, device)
        scheduler.step()

        # --- Save model only if validation loss improves ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{folder_name}/models/bm_{epoch+1}.pth')
            logger.info(f"✅ Saved new best model at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

        # Every 5 epochs: evaluate on test set
        if (epoch + 1) % 5 == 0 or (epoch+1)==1:
            logger.info(f"📊 Running test accuracy at epoch {epoch+1}")
            test_accuracy(model, test_loader, device, calc_vhs, get_labels, logger=logger)

        logger.info(f"Epoch {epoch+1}/{num_epochs} | KPT Loss: {running_loss_kpt/len(train_loader):.4f}| | CLS Loss: {running_loss_cls/len(train_loader):.4f}| Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    return model

############################## END : TRAIINING

# Modify the training function call to include folder_name
trained_model = train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    folder_name=folder_name,
    calc_vhs=calc_vhs,
    device=device,
    num_epochs=500,
    accumulation_steps=accumulation_steps
)

## Save the Last Model
torch.save(trained_model.state_dict(), f'{folder_name}/models/last.pth')
#20250506_204657
#20250506_230518 - Current