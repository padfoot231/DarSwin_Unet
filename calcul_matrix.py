import torch
from torch.utils.data import DataLoader
from datasets.dataset_synapse import *
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from pykeops.torch import LazyTensor 

cuda_id= torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def get_inverse_distortion(num_points,D, fov,device=cuda_id):
    theta_d_max = fov/2
    m = 2.288135593220339
    n = 5.0
    a = theta_d_max
    b = 8.31578947368421
    c =  0.3333333333333333

  
    def f(x, n, a, b):
        return b*torch.pow(x/a, n)
    def h(x, m, a):
        return -torch.pow(-x/a + 1, m) + 1
    def g(x, m, n, a, b, c):
        return c*f(x, n, a, b) + (1-c)*h(x, m, a)
    def g_inv(y, m, n, a, b, c):
        test_x = torch.linspace(0, theta_d_max, 20000).to(device)
        test_y = g(test_x, m, n, a, b, c).to(device)
        x = torch.zeros(num_points).to(device)
        for i in range(num_points):
            lower_idx = test_y[test_y <= y[i]].argmax()
            x[i] = test_x[lower_idx]
        return x

    def rad(D, theta, theta_max):
        focal_func = lambda x: 1/(x.reshape(1, x.shape[0]).repeat_interleave(D.shape[1], 0).flatten() * (torch.outer(D[0], x**0).flatten() + torch.outer(D[1], x**1).flatten() + torch.outer(D[2], x**2).flatten() +torch.outer(D[3], x**3).flatten()))
        focal = focal_func(theta_max.reshape(1)).reshape(1, D.shape[1])
        funct = g_inv(theta, m = m, n = n, a = a, b = b, c = c).reshape(1,-1).repeat_interleave(D.shape[1], 0).transpose(1,0).to(device)
        radius= focal* funct * (D[0] * funct**0 + D[1] * funct**1 + D[2] * funct**2 + D[3] * funct**3)
        return radius
    
    theta_d = torch.linspace(0, g(theta_d_max, m = m, n = n, a = a, b = b, c = c), num_points + 1).to(device)
    r_list = rad(D, theta_d, theta_d_max).to(device)
    return r_list, theta_d_max

def get_inverse_dist_spherical(num_points, xi, fov, new_f, device=cuda_id):
    
    theta_d_max = fov/2

    m = 2.288135593220339
    n = 5.0
    a = theta_d_max
    b = 8.31578947368421
    c =  0.3333333333333333



    def f(x, n, a, b):
        return b*torch.pow(x/a, n)
    def h(x, m, a):
        return -torch.pow(-x/a + 1, m) + 1
    def g(x, m, n, a, b, c):
        return c*f(x, n, a, b) + (1-c)*h(x, m, a)
    def g_inv(y, m, n, a, b, c):
        test_x = torch.linspace(0, theta_d_max, 10000).to(device)
        test_y = g(test_x, m, n, a, b, c).to(device)
        x = torch.zeros(num_points).to(device)
        for i in range(num_points):
            lower_idx = test_y[test_y <= y[i]].argmax()
            x[i] = test_x[lower_idx]
        return x

    def rad(xi, theta):
        funct = g_inv(theta, m = m, n = n, a = a, b = b, c = c)
        funct = funct.reshape(num_points, 1)
        # breakpoint()
        radius = ((torch.cos(funct[-1]) + xi)/torch.sin(funct[-1]))*torch.sin(funct)/(torch.cos(funct) + xi)
        return radius
  
    theta_d = torch.linspace(0, g(theta_d_max, m = m, n = n, a = a, b = b, c = c), num_points + 1).to(device)

    r_list = rad(xi, theta_d)
   
    return r_list, theta_d_max

def get_sample_params_from_subdiv(subdiv, distortion_model, img_size, D, device=cuda_id):
    """Generate the required parameters to sample every patch based on the subdivison
    Args:
        subdiv (tuple[int, int]): the number of subdivisions for which we need to create the 
                                  samples. The format is (radius_subdiv, azimuth_subdiv)
        n_radius (int): number of radius samples
        n_azimuth (int): number of azimuth samples
        img_size (tuple): the size of the image
    Returns:
        list[dict]: the list of parameters to sample every patch
    """
    max_radius = min(img_size)/2
    width = img_size[1]
    # D_min = get_inverse_distortion(subdiv[0], D, max_radius)
    if distortion_model == 'spherical': # in case of spherical distortion pass the 
        fov = D[2][0]
        f  = D[1]
        xi = D[0]
        D_min, theta_max = get_inverse_dist_spherical(subdiv[0], xi, fov, f)
        D_min = D_min*max_radius
        # breakpoint()
    elif distortion_model == 'polynomial' or distortion_model == 'polynomial_woodsc':
        # 
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, 1.0)
        D_min = D_min*max_radius
    # breakpoint()
    alpha = 2*torch.tensor(np.pi).cuda() / subdiv[1]
    D_min = D_min.reshape(subdiv[0], 1, D.shape[1]).repeat_interleave(subdiv[1], 1).to(device)
    phi_start = 0
    phi_end = 2*torch.tensor(np.pi)
    phi_step = alpha
    phi_list = torch.arange(phi_start, phi_end, phi_step)
    phi_list = phi_list.reshape(1, subdiv[1]).repeat_interleave(subdiv[0], 0).reshape(subdiv[0], subdiv[1], 1).repeat_interleave(D.shape[1], 2).to(device)
   
    phi_list_cos  = torch.cos(phi_list) 
    phi_list_sine = torch.sin(phi_list) 
    x = D_min * phi_list_cos    # takes time the cosine and multiplication function 
    y = D_min * phi_list_sine
    return x.transpose(1, 2).transpose(0,1), y.transpose(1, 2).transpose(0,1), theta_max


def get_grid_pix(H,W):
  y = torch.linspace(0, W, W+1) - (W//2+0.5)
  x = torch.linspace(0,H,H+1) - (H//2 + 0.5)
  grid_x, grid_y = torch.meshgrid(x[1:], y[1:])
  x_ = grid_x.reshape(H*H, 1)
  y_ = grid_y.reshape(W*W, 1)
  grid_pix = torch.cat((x_, y_), dim=1)
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

def get_sample_loc(root_path, split='test', model='spherical', img_size=(64,64), subdiv=(16,64), n=(25,4), device=cuda_id):
    
    db= Synapse_dataset(base_dir=root_path, model=model, split=split)
    loader = DataLoader(db, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    dist_model= model
    subdiv = subdiv
    n_radius = n[0]
    n_azimuth = n[1]
    img_size = img_size
    H, W = img_size
    radius_buffer, azimuth_buffer = 0, 0

    grid_pix = get_grid_pix(H,W)
    grid_pix = grid_pix.reshape(1, H*W, 2).to(device)

    for i_batch, sampled_batch in enumerate(loader):
        #uncomment line in dataset_synapse
        label_batch, dist, path =  sampled_batch['label'], sampled_batch['dist'], sampled_batch['path']
        label_batch, dist = label_batch.to(device), dist.to(device)
        
        dist= dist.transpose(1,0)
        xc, yc, theta_max = get_sample_params_from_subdiv(
              subdiv = (subdiv[0]*n_radius,subdiv[1]*n_azimuth),
              img_size=img_size,
              distortion_model = model,
              D = dist,
              device=device
              )
        
        B, n_p, n_s = xc.shape
        x = xc.reshape(1, 1, -1).transpose(1,2).to(device)
        y = yc.reshape(1, 1, -1).transpose(1,2).to(device)
        grid_ = torch.cat((x, y), dim=2).type(torch.float32)
    
        B, N, D, P, k = grid_.shape[0], grid_.shape[1], 2, grid_pix.shape[1], 4
        cl = KNN(grid_/(H//2), grid_pix/(W//2), P, k)
        cl = cl[0].cpu() 

        cl_path = path[0] 
        if split =='test':
            cl_path = cl_path[:-4]+'_test.npy'

        with open(os.path.join(root_path, cl_path), 'wb') as f:
            np.save(f, cl)
        




if __name__ == "__main__":
    root_path= '/home-local2/icshi.extra.nobkp/matterport/M3D_low'
    split='test'
    get_sample_loc(root_path, split, model='spherical', img_size=(64,64), subdiv=(16,64), n=(25,4))
    