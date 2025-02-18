import torch
from utils_rad import get_sample_params_from_subdiv, get_sample_locations
import matplotlib.pyplot as plt 
import time 
from imageio.v2 import imread 
from skimage.transform import resize 
import numpy as np 
import torch.nn as nn 

cuda_id= "cuda:0"

def func(arr,Npix, W=4):
  mat= -np.ones((Npix,W))
  Ns= arr.shape[0]//2
  pixels, indices = arr[:Ns], arr[Ns:]
  uniq, index = np.unique(pixels, return_index=True)
  #print("Pixels ",pixels)
  #print("Indices ", indices)
  start= index
  stop= np.append(index[1:], [Ns])
  indices= np.append(indices, [-1])
  #print(start)
  #print(stop)
  output_indices= [np.pad(range(a,b), (0,W-b+a), 'constant', constant_values=[-1]) for (a,b) in zip(start, stop)]
  #print(output_indices)
  #print(np.take(indices, output_indices))
  mat[uniq,:] = np.take(indices, output_indices)
  #print(mat)
  return mat

#grid: fractional grid, output (from darswin or sampled label here), indices representing the matrix of indices (from 0 to H*H -1)
def round_sample(grid, output, indices):
  H= indices.shape[0]
  K=H*H
  B, L, Np, Ns = output.shape
  N=Np*Ns #nb samples

  grid = grid + (H//2 -0.5) #go back to coord [-0.5, 127,5] #127,5= H -0.5
  grid= torch.floor(torch.abs(grid)) #go back to coord [0,127]
  grid= grid.reshape(B,-1,2)
  coord= grid.view(-1,2)
  cl= indices[coord[:,0].tolist(),coord[:,1].tolist()].view(B,-1)

  #print (cl)
  sorted_pixels, sorted_indices= torch.sort(cl, dim=1)
  _, sample_counts = torch.unique_consecutive(sorted_pixels, return_counts=True, dim=1)
  W = sample_counts.max().cpu()
  #print(W)
  concat= torch.hstack([sorted_pixels, sorted_indices]).cpu().numpy()
  mat=  np.apply_along_axis(func,1,concat, Npix=K, W=W )
  mat = torch.tensor(mat).cuda(cuda_id)
  output = output.reshape(B, L, -1)
  dummy_samples = torch.zeros([B,L,1]).cuda(cuda_id)
  output = torch.cat((output, dummy_samples), 2) 
  #output = output.transpose(1,0)
  #print('mat',mat.shape)
  #print('out', output.shape)
  pixel_out = output[:,:,mat.long()]
  pixel_out = pixel_out[::B].squeeze(0)
  pixel_out= pixel_out.transpose(1,0)
  valid=torch.count_nonzero(pixel_out, dim=3)
  valid= torch.where(valid==0,1, valid)
  pixel_out = torch.sum(pixel_out, dim=3)/valid
  pixel_out = pixel_out.view(B, L, H, H)
  return pixel_out
 

  """
  cl= indices[coord[:,0].tolist(),coord[:,1].tolist()].view(B,-1)
  ind = torch.arange(N).reshape(1, -1)
  ind = torch.repeat_interleave(ind, B, 0)
  mat = torch.zeros(B, K, N).cuda(cuda_id) #the division matrix mapping samples to pixels
  mat[:, cl, ind] = 1
  output = output.reshape(B, L, -1).transpose(1, 2)
  pixel_out = torch.matmul(mat, output)
  div = mat.sum(-1).unsqueeze(2) #frequency of each sample to pixel (for averaging the values)
  div[div == 0] = 1
  pixel_out = torch.div(pixel_out, div)
  pixel_out = pixel_out.transpose(2, 1).reshape(B,L, H, H)

  return pixel_out
  """



if __name__ == "__main__":
    
    cuda_id= "cuda:0"

    radius_subdiv = 32
    azimuth_subdiv = 128
    dist_model= "polynomial"
    subdiv = (radius_subdiv, azimuth_subdiv)
    n_radius = 2
    n_azimuth = 2
    img_size = (128,128)
    H= img_size[0]
    radius_buffer, azimuth_buffer = 0, 0
    D= np.array([341.725, -26.4448, 32.7864, 0.50499])
    D= torch.tensor(D).reshape(4,1).cuda(cuda_id)

   
    img_path ='/gel/usr/icshi/DATA_FOLDER/Synwoodscape/rgb_images/00000_FV.png'
    img= imread(img_path)/255
    img=resize(img,(H,H), order=1)
    img= torch.tensor(img).permute(2,0,1).unsqueeze(0).cuda(cuda_id)
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


    s= time.time()
    indices= torch.arange(H*H).reshape(H,H).cuda(cuda_id)
    pixel_out=  round_sample(grid, output, indices)
    print(pixel_out.shape)

    print(time.time()-s)

    example= pixel_out[0,...].permute(1,2,0)
    print(example.shape)
    plt.imsave('opt.png', example.cpu().numpy())


   
        