import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_vhs.helper import get_labels

#-------------------------- VHS LOSS ------------------
class VHSAwareLoss(nn.Module):
    """
    VHS-aware loss function that combines L1 loss with a class-aware penalty.       
    The penalty is based on the distance from class boundaries and class imbalance.
    Args:
        class_weights (torch.Tensor): Weights for each class to handle class imbalance.
        margin (float): Margin for the VHS classes.
        middle_class_multiplier (float): Multiplier for the middle class penalty.
    """

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