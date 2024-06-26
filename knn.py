import torch
from utils_rad import get_sample_params_from_subdiv, get_sample_locations
import matplotlib.pyplot as plt 
import time 
from imageio.v2 import imread 
from skimage.transform import resize 
import numpy as np 
import torch.nn as nn
from pykeops.torch import LazyTensor 
from datasets.dataset_synapse import load_depth, warpToFisheye
import cv2 

cuda_id= "cuda:1"

def get_grid_pix(H):
  x = torch.linspace(0, H, H+1) - (H//2+0.5)
  y = torch.linspace(0,H,H+1) - (H//2 + 0.5)
  grid_x, grid_y = torch.meshgrid(x[1:], y[1:])
  x_ = grid_x.reshape(H*H, 1)
  y_ = grid_y.reshape(H*H, 1)
  grid_pix = torch.cat((x_, y_), dim=1)
  grid_pix = grid_pix.reshape(1, H*H, 2).cuda(cuda_id)
  return grid_pix


def KNN(x, c, P=10, k = 1, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    #     start = time.time()
    B, N, D = x.shape  # Number of samples, dimension of the ambient space
    x_i = LazyTensor(x.view(B, 1, N, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(B, P, 1, D))  # (1, K, D) centroids

    D_ij = ((x_i - c_j) ** 2).sum(B, -1)
    #     import pdb;pdb.set_trace()
    #     .sum(B, -1)  # (N, K) symbolic squared distances
      
    cl = D_ij.argKmin(k, dim=2)  # Points -> Nearest cluster
    return cl
    

def restruct(output, cls, embed_dim, H, W):
    B, P, K = cls.shape
    B, dim, patch, sample = output.shape
    out_1 = output.view(B, dim, -1).transpose(1, 2)
    # out_1 = output.transpose(1, 2)
    out = out_1[:, cls]
    out = out.view(-1, P, K, dim)[0::B]
    out = out.mean(2)
    out = out.transpose(1, 2)
    out = out.reshape(B, embed_dim, H, W)
    return out



if __name__ == "__main__":
    
    radius_subdiv = 16
    azimuth_subdiv = 64
    dist_model= "spherical"
    subdiv = (radius_subdiv, azimuth_subdiv)
    n_radius = 10
    n_azimuth = 10
    img_size = (64,64)
    H= img_size[0]
    radius_buffer, azimuth_buffer = 0, 0
    #D= np.array([341.725, -26.4448, 32.7864, 0.50499])
    #D= torch.tensor(D).reshape(4,1).cuda(cuda_id)

   
    #img_path ='/gel/usr/icshi/DATA_FOLDER/Synwoodscape/rgb_images/00000_FV.png'
    depth_path= '/home-local2/icshi.extra.nobkp/matterport/M3D_low/1LXtFkjw3qL/108_spherical_1_depth_center_0.png'
    max_depth= 8.0
    #img= imread(img_path)/255
    
    img = load_depth(depth_path, max_depth)['depth']
    img=img.permute(1,2,0)
    h= img.shape[0]
    fov=150
    xi= 0.7
    img, f = warpToFisheye(img.numpy(), outputdims=(h,h),xi=xi, fov=fov, order=0)
    D = np.array([xi, f/(h/H), np.deg2rad(fov)])
    D= torch.tensor(D).reshape(-1,1).cuda(cuda_id)
    #img=resize(img,(H,H), order=0)
    img = cv2.resize(img, (H,H), interpolation = cv2.INTER_NEAREST)
    img= torch.tensor(img).unsqueeze(2).permute(2,0,1).unsqueeze(0).cuda(cuda_id)
    img=img.float()

    
    

    params, D_s = get_sample_params_from_subdiv(
                    subdiv= subdiv,
                    img_size= img_size,
                    distortion_model = dist_model,
                    D = D,
                    n_radius= n_radius,
                    n_azimuth= n_azimuth,
                    radius_buffer=radius_buffer,
                    azimuth_buffer=azimuth_buffer)
    sample_locations = get_sample_locations(**params)
    B, n_p, n_s = sample_locations[0].shape
       
    x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
    y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()

    #used to get samples from labels (to use nn.functional.grid_sample grid should be btw -1,1)
    grid_out= torch.cat((y_/(H//2), x_/(H//2)), dim=3)

    output= nn.functional.grid_sample(img, grid_out, align_corners = True, mode='nearest')
   
    grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2
    grid= grid.view(B,-1,2)
    grid_pix= get_grid_pix(H)

    s= time.time()
    B, N, D, P, k = grid.shape[0], grid.shape[1], 2, grid_pix.shape[1], 4
    cl = KNN(grid/(H//2), grid_pix/(H//2), P, k)
    dim= output.shape[1]
    pixel_out = restruct(output, cl , dim, H, H)
    print(time.time()-s)

    pixel_out= torch.where(img==0, img, pixel_out)


    example= pixel_out[0,...].permute(1,2,0)
    example = example.squeeze(2)
    print(example.shape)
    plt.imsave('knn_180.png', example.cpu().numpy())


   
        