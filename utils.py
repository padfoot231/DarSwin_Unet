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






"""
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                breakpoint()
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        breakpoint()
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + "_gt.nii.gz")
    return metric_list
"""