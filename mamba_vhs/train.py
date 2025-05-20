import os
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from PIL import Image
from scipy.io import loadmat, savemat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn import L1Loss, CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast

from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.transforms as T

# Mamba modules
from mamba_vhs.dataset import DogHeartDataset, get_transform
from mamba_vhs.helper import calc_vhs, get_labels
from mamba_vhs.mamba_vhs import get_mamba_vhs_model
from mamba_vhs.logger import setup_logger
from mamba_vhs.vhs_aware_loss import VHSAwareLoss

# State-space and sequence modeling
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange

# Metrics
from sklearn.metrics import classification_report
# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# 2. Load the checkpoint
checkpoint_path = '20250506_204657/models/bm_43.pth'
model = get_mamba_vhs_model(checkpoint_path=checkpoint_path)
model = model.to(device)

#################### END : MODEL

resized_image_size = 512
true_batch_size = 256
accumulation_steps = 8
root_db_folder= "../data"

dataset_train = DogHeartDataset(f'{root_db_folder}/Train', get_transform(resized_image_size))
dataset_valid = DogHeartDataset(f'{root_db_folder}/Valid', get_transform(resized_image_size))
train_loader = DataLoader(dataset_train, batch_size=true_batch_size//accumulation_steps, shuffle=True, num_workers=8) 
valid_loader = DataLoader(dataset_valid, batch_size=true_batch_size//accumulation_steps, shuffle=False, num_workers=8)
test_dataset = DogHeartDataset(f'{root_db_folder}/Test_Images', get_transform(resized_image_size))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)

################################
## Utilites
################################
 
logger, folder_name, log_file_name, loss_file_name, vhs_score_file, best_val_model, last_model = setup_logger()
logger.info("Logger initialized successfully.")

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

##------------------- LOSS FUNCTIONS ------------------  

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