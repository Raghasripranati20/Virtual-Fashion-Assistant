import torch

def compute_uncertainty_map(mc_outputs):
    # Shape: [N, C, H, W]
    variance_map = torch.var(mc_outputs, dim=0)
    return variance_map.mean(dim=0)  # [H, W]
