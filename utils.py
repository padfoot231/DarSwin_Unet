import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio
import torch
import torch.nn.functional as F

# compute field of view from focal length and xi
def compute_fov(f, xi, width):
    return 2 * np.arccos((xi + np.sqrt(1 + (1 - xi**2) * (width/2/f)**2)) / ((width/2/f)**2 + 1) - xi)

# compute focal length from field of view and xi
def compute_focal(fov, xi, width):
    return width / 2 * (xi + np.cos(fov/2)) / np.sin(fov/2)

def mask_fish(im, f, xi, new_fov):
    """extract a fov from a fisheye image.

    Args:
        im (str or np.ndarray): image or path to image
        f (float): focal length of the camera in pixels
        xi (float): distortion parameter following the spherical distortion model
        new_fov: fov to extract

    Returns:
        np.ndarray: undistorted image
        """

    if isinstance(im, str):
        im = imageio.imread(im)

    im = torch.tensor(im.astype(float))

    height, width, _ = im.shape

    max_theta= np.deg2rad(new_fov/2) 

    u0 = width / 2
    v0 = height / 2

    grid_x, grid_y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    X_Cam = (grid_x - u0) / f
    Y_Cam = (grid_y - v0) / f
    omega = (xi + np.sqrt(1 + (1 - xi**2) * (X_Cam**2 + Y_Cam**2))) / (X_Cam**2 + Y_Cam**2 + 1)
    X_Sph = X_Cam * omega
    Y_Sph = Y_Cam * omega
    Z_Sph = omega - xi
    nthetax = np.arctan2(X_Sph, Z_Sph)
    nthetay = np.arctan2(Y_Sph, Z_Sph)
    valid = (np.abs(nthetay) <=max_theta) & (np.abs(nthetax) <= max_theta)
    nx= grid_x[~valid]
    ny=grid_y[~valid]
    new_im = im 
    new_im[ny,nx]=0
    return new_im.squeeze(2)

def eval_depth(pred, target, mask=None):
    assert pred.shape == target.shape
    valid_mask = (target > 0).detach()
    if mask is not None:
        valid_mask = (valid_mask * mask).detach()

    target= target[valid_mask]
    pred= pred[valid_mask]
    pred_fix = torch.where(pred == 0.0,
            pred + 1e-24,
            pred
        )

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred_fix) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)
    #abs_rel = torch.mean(torch.abs(diff) /pred)
    #sq_rel = torch.mean(torch.pow(diff, 2) /pred)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred_fix) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.85 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10':log10.item(), 'silog':silog.item()}


class MDELoss(nn.Module):
    def __init__(self, lambd=0.85):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, mask=None):
        pred_fix = torch.where(pred == 0.0,
            pred + 1e-24,
            pred
        )

        valid_mask = (target > 0).detach()
        if mask is not None:
            valid_mask = (valid_mask * mask).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred_fix[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


