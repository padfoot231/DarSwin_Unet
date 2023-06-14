from PIL import Image
import torch
import torch.nn as nn
import pickle as pkl
import random
import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor
import numpy as np
use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"
# from utils import get_sample_params_from_subdiv, get_sample_locations
import numpy as np
from utils_rad import get_sample_params_from_subdiv, get_sample_locations, distort_image
from torchvision.transforms import transforms
t2pil = transforms.ToTensor()
pil = transforms.ToPILImage()

def KMeans(x, c, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    B, N, D = x.shape  # Number of samples, dimension of the ambient space

    x_i = LazyTensor(x.view(B, N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(B, 1, K, D))  # (1, K, D) centroids

    D_ij = ((x_i - c_j) ** 2).sum(B, -1)  # (N, K) symbolic squared distances
    cl = D_ij.argmin(dim=2).long().view(B, -1)  # Points -> Nearest cluster

    return cl, c

with open('/home-local2/akath.extra.nobkp/woodscapes/calib.pkl', 'rb') as f:
    data = pkl.load(f)

key = list(data.keys())


grid = torch.empty([len(key), 4096, 25, 2])
# sampling_loc = []
# # D = torch.tensor([339.749, -31.988, 48.275, -7.201]).reshape(1,4).transpose(1,0).cuda()
# D = torch.tensor([0, 0, 0, 0]).reshape(1,4).transpose(1,0).cuda("cuda:1")
# # a = torch.tensor([0.0, 0.375, 3.047911227757854]).reshape(1, 3).transpose(0,1).cuda("cuda:1") ### (16, 16)
# a = torch.tensor([0.0, 1.5, 3.047911227757854]).reshape(1, 3).transpose(0,1).cuda("cuda:1")
# # a = torch.tensor([1.0, 33.535136959282696, 3.047911227757854]).reshape(1, 3).transpose(0,1).cuda()

for i in range(len(key)):
    D = torch.tensor(data[key[i]].reshape(1,4).transpose(1,0)).cuda("cuda:0")
    
    azimuth_subdiv = 128
    radius_subdiv = 32
    subdiv = (radius_subdiv, azimuth_subdiv)
    # subdiv = 3
    n_radius = 5
    n_azimuth = 5
    img_size = (128, 128)
    radius_buffer, azimuth_buffer = 0, 0
    params, D_s = get_sample_params_from_subdiv(
        subdiv=subdiv,
        img_size=img_size,
        D = D, 
        n_radius=n_radius,
        n_azimuth=n_azimuth,
        radius_buffer=radius_buffer,
        azimuth_buffer=azimuth_buffer, 
        distortion_model = 'polynomial')

    sample_locations = get_sample_locations(**params)  ## B, azimuth_cuts*radius_cuts, n_radius*n_azimut
    B, n_p, n_s = sample_locations[0].shape
    x_ = sample_locations[0].reshape(1, n_p, n_s, 1).float()
    x_ = x_/ 64
    y_ = sample_locations[1].reshape(1, n_p, n_s, 1).float()
    y_ = y_/64
#     out = torch.cat((y_, x_), dim = 3)
    grid_ = torch.cat((x_, y_), dim=3)
    grid[i] = grid_[0].cpu()
    print(i)
#     import pdb;pdb.set_trace()

# with open('/home-local2/akath.extra.nobkp/woodscapes/grid_sample.pkl', 'rb') as f:
#     grid = pkl.load(f)
x = torch.linspace(0, 128, 129) - 64.5
y = torch.linspace(0, 128, 129) - 64.5
grid_x, grid_y = torch.meshgrid(x[1:], y[1:])
x_ = grid_x.reshape(128*128, 1)
y_ = grid_y.reshape(128*128, 1)
grid_pix = torch.cat((x_, y_), dim=1)
grid_pix = grid_pix.reshape(1, 128*128, 2)
grid_pix = torch.repeat_interleave(grid_pix, grid.shape[0], 0)


def resample(grid, grid_pix, H, B):
    B, N, D, K = grid.shape[0], grid.shape[1], 2, grid_pix.shape[1]
    cl, c = KMeans(grid/(H//2), grid_pix/(H//2), K)
#     import pdb;pdb.set_trace()
    ind = torch.arange(N).reshape(1, -1)
    ind = torch.repeat_interleave(ind, B, 0)
    mat = torch.zeros(B, K, N)
    mat[:, cl, ind] = 1
#     output = output.reshape(B, L, -1).transpose(1, 2)
#     pixel_out = torch.matmul(mat, output)
#     div = mat.sum(-1).unsqueeze(2)
#     div[div == 0] = 1
#     pixel_out = torch.div(pixel_out, div)
#     pixel_out = pixel_out.transpose(2, 1).reshape(B, 3, H, H)
    return mat

grid = grid.reshape(8234, -1, 2)
B, N, D = grid.shape
B, N_p, D = grid_pix.shape
dic = {}
for i in range(len(key)):
    g = grid[i].reshape(1, N, D)
    g_p = grid_pix[i].reshape(1, N_p, D)
    x = resample(g, g_p, 128, 2)
    x = np.array(x)
    np.save('/home-local2/akath.extra.nobkp/woodscape/mat/' + key[i][:-4], x)
    print(i)