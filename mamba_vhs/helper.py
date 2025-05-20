import torch

def calc_vhs(x: torch.Tensor):
    """
    Calculate VHS (Ventricular Heart Score) based on the coordinates of six points.
    The VHS is calculated using the formula:
    VHS = 6 * (AB + CD) / EF
    where AB, CD, and EF are the distances between specific pairs of points.
    Args:
        x (torch.Tensor): Tensor of shape (N, 12) containing the coordinates of six points.
    Returns:
        torch.Tensor: Tensor of shape (N,) containing the VHS scores.
    """
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

def get_labels(vhs):
    """
    Assigns class labels based on VHS values using a compact formula.

    Args:
        vhs (torch.Tensor): Tensor containing VHS scores.
        
    Returns:
        torch.Tensor: Tensor of class labels (0, 1, 2).
    """
    return ((vhs >= 10).long() - (vhs < 8.2).long() + 1).squeeze()