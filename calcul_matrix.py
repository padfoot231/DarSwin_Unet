import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.dataset_synapse import *
from utils_rad import get_sample_params_from_subdiv, get_sample_locations
import matplotlib.pyplot as plt 
import time 
from imageio.v2 import imread 
from skimage.transform import resize 
from knn import restruct, get_grid_pix, KNN
import numpy as np 
import os 

cuda_id= "cuda:1"

def calcul_cl(root_path):
    batch_size= 1
    
    db_train = Synapse_dataset(base_dir=root_path, model="spherical", split="test")
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    radius_subdiv = 16
    azimuth_subdiv = 64
    dist_model= "spherical"
    subdiv = (radius_subdiv, azimuth_subdiv)
    n_radius = 10
    n_azimuth = 10
    img_size = (64,64)
    H= img_size[0]
    radius_buffer, azimuth_buffer = 0, 0

    for i_batch, sampled_batch in enumerate(trainloader):
        #uncomment line in dataset_synapse
        label_batch, dist, path =  sampled_batch['label'], sampled_batch['dist'], sampled_batch['path']
        label_batch, dist = label_batch.cuda(cuda_id), dist.cuda(cuda_id)
        
        #torch.save(label_batch, 'label.pt')
        #torch.save(dist, 'distL.pt')
        
        dist= dist.transpose(1,0)
        #print("clacul cl ", dist)
        params, D_s = get_sample_params_from_subdiv(
                    subdiv= subdiv,
                    img_size= img_size,
                    distortion_model = dist_model,
                    D = dist,
                    n_radius= n_radius,
                    n_azimuth= n_azimuth,
                    radius_buffer=radius_buffer,
                    azimuth_buffer=azimuth_buffer)
        sample_locations = get_sample_locations(**params)
        B, n_p, n_s = sample_locations[0].shape
        print(sample_locations[0].shape)
       
        x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
        y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()

        grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2
        grid = grid.view(B,-1,2)
        grid_pix= get_grid_pix(H)
        grid_pix= torch.repeat_interleave(grid_pix, B, 0).cuda(cuda_id)
        B, N, D, P, k = grid.shape[0], grid.shape[1], 2, grid_pix.shape[1], 4
        cl = KNN(grid/(H//2), grid_pix/(H//2), P, k)
        cl = cl[0].cpu()
        with open(os.path.join(root_path, path[0]), 'wb') as f:
            np.save(f, cl)
        




if __name__ == "__main__":
    #root_path = '/gel/usr/icshi/DATA_FOLDER/Synwoodscape'
    root_path= '/home-local2/icshi.extra.nobkp/matterport/M3D_low'
    batch_size= 1
    
    db_train = Synapse_dataset(base_dir=root_path, model="spherical", split="val")
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    radius_subdiv = 16
    azimuth_subdiv = 64
    dist_model= "spherical"
    subdiv = (radius_subdiv, azimuth_subdiv)
    n_radius = 10
    n_azimuth = 10
    img_size = (64,64)
    H= img_size[0]
    radius_buffer, azimuth_buffer = 0, 0

    for i_batch, sampled_batch in enumerate(trainloader):
        #uncomment line in dataset_synapse
        label_batch, dist, path =  sampled_batch['label'], sampled_batch['dist'], sampled_batch['path']
        label_batch, dist = label_batch.cuda(cuda_id), dist.cuda(cuda_id)
        
        #torch.save(label_batch, 'label.pt')
        #torch.save(dist, 'distL.pt')
        #print(dist)
        dist= dist.transpose(1,0)
        params, D_s = get_sample_params_from_subdiv(
                    subdiv= subdiv,
                    img_size= img_size,
                    distortion_model = dist_model,
                    D = dist,
                    n_radius= n_radius,
                    n_azimuth= n_azimuth,
                    radius_buffer=radius_buffer,
                    azimuth_buffer=azimuth_buffer)
        sample_locations = get_sample_locations(**params)
        B, n_p, n_s = sample_locations[0].shape
        #print(sample_locations[0].shape)
       
        x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
        y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()

        grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2
        grid= grid.view(B,-1,2)
        grid_pix= get_grid_pix(H)
        grid_pix= torch.repeat_interleave(grid_pix, B, 0).cuda(cuda_id)
        B, N, D, P, k = grid.shape[0], grid.shape[1], 2, grid_pix.shape[1], 4
        cl = KNN(grid/(H//2), grid_pix/(H//2), P, k)
        cl = cl[0].cpu()
        #print(cl.shape)

        with open(os.path.join(root_path, path[0]), 'wb') as f:
            np.save(f, cl)
        

