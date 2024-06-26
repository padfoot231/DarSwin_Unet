import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt 
from sparseCnn import SparseConvNet, sparseLoss, round_sample_opt, round_sample
from utils_rad import get_sample_params_from_subdiv, get_sample_locations
from datasets.utils import get_mask_wood, get_mask_matterport


cuda_id= "cuda:0"

def get_sparse_label(label_batch, dist,subdiv, img_size, distortion_model,n_radius, n_azimuth, radius_buffer, azimuth_buffer):
    params, D_s = get_sample_params_from_subdiv(
                            subdiv= subdiv,
                            img_size= img_size,
                            distortion_model = distortion_model,
                            D = dist,
                            n_radius= n_radius,
                            n_azimuth= n_azimuth,
                            radius_buffer= radius_buffer,
                            azimuth_buffer= azimuth_buffer)
    sample_locations = get_sample_locations(**params)
    B, n_p, n_s = sample_locations[0].shape
    B, n_p, n_s = sample_locations[0].shape
    x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
    y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()

    #used to get samples from labels (to use nn.functional.grid_sample grid should be btw -1,1)
    H= img_size[0]
    grid_out= torch.cat((y_/(H//2), x_/(H//2)), dim=3)
    sampled_label= nn.functional.grid_sample(label_batch, grid_out, align_corners = True, mode='nearest')
    grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2
    #sparse_batch = round_sample(grid, sampled_label, indices)

    return sampled_label, grid 


def trainer_sparseCnn(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    
    batch_size = args['batch_size']
    db_val = Synapse_dataset(base_dir=args['root_path'], model=args['dist_model'], split="val")
    print("The length of validation set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args['seed'] + worker_id)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    save_path= snapshot_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    H = args['img_size']
    subdiv=(args['radius_subdiv'], args['azimuth_subdiv'])
    img_size= (args['img_size'], args['img_size'])
    distortion_model = args['dist_model']
    n_radius= args['n_radius']
    n_azimuth= args['n_azimuth']
    radius_buffer= args['radius_buffer']
    azimuth_buffer= args['azimuth_buffer']
    indices= torch.arange(H*H).reshape(H,H).cuda(cuda_id) #grid representing the index of each coord (values from 0 to 127)
    #to be updated wether add it in the dataloader or inside the loop (for matterport invalid is specific to each image)
    #add it in the dataloader with an attribute return_masks=True/False
    if distortion_model=="polynomial":
        mask_p= get_mask_wood()
    else:
        mask_p= get_mask_matterport()
    #mask_p = get_mask_matterport() #get_mask_wood() #1 for periphery otherwise 0 (array)
    mask_p= torch.tensor(mask_p).unsqueeze(0).unsqueeze(0)
    #model.eval()
    mae= nn.L1Loss()
    losses=[]
    times=[]
    with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    label_batch, dist = sampled_batch['label'], sampled_batch['dist']
                    label_batch, dist = label_batch.cuda(cuda_id), dist.cuda(cuda_id)
                    dist= dist.transpose(1,0)
                    sampled_label, grid = get_sparse_label(label_batch, dist,subdiv, img_size, distortion_model,n_radius, n_azimuth, radius_buffer, azimuth_buffer)
                    #plt.imsave(save_path+ '/sparse_label_{}.png'.format(epoch_num),sparse_batch[0,...].squeeze(0).cpu().numpy())
                    B= grid.shape[0]
                    
                    masks_p= torch.repeat_interleave(mask_p, B, 0).cuda(cuda_id)

                    s=time.time()
                    sparse_batch = round_sample(grid, sampled_label, indices)
                    #if label!=0 it will return 1 so the pixel is not missing  (periphery=0 => 0 so periphery will be considered missing)
                    masks= (sparse_batch!=0).long()
                    #add the periphery mask to exclude/include the periphery
                    masks= masks+ masks_p
                    outputs= model(sparse_batch.float(), masks.float())
                    times.append( time.time()-s)
                    valid = masks_p==0
                    loss= mae(outputs[valid],label_batch[valid])

                    #print('mae', loss.item())
                    losses.append(loss.item())

                    #loss_masks= torch.where(masks==0,1,0)
                    #loss_val= SLoss(label_batch, outputs, loss_masks)
                    #val_losses.append(loss_val.item())
                    if i_batch% 100==0:
                        plt.imsave(save_path+ '/val_pred_{}.png'.format(i_batch),outputs[0,...].squeeze(0).cpu().numpy())
                        plt.imsave(save_path+ '/val_label_{}.png'.format(i_batch),label_batch[0,...].squeeze(0).cpu().numpy())
                        plt.imsave(save_path+ '/val_sparse_label_{}.png'.format(i_batch),sparse_batch[0,...].squeeze(0).cpu().numpy())

                print("Avg loss ", torch.mean(torch.tensor(losses)))
                print("Avg time ", torch.sum(torch.tensor(times))/len(db_val))
            #logging.info('epoch  %d : val_loss : %f' % (epoch_num, torch.mean(torch.tensor(val_losses))))
        
       

   
    return "Evaluation Finished!"

if __name__ == "__main__":
    ckpt= '/gel/usr/icshi/Radial-transformer-Unet/SparseM/epoch_9.pth'
    dataset='Matterport'
    model= SparseConvNet(dataset=dataset).cuda(cuda_id)
    model.load_from(ckpt)
    for param in model.parameters():
        param.requires_grad=False

    dist_model="spherical"

    if dist_model== 'polynomial':
        root_path= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape'
    else:
        root_path= '/gel/usr/icshi/Swin-Unet/data/M3D_low'
    

    args={'root_path': root_path, 
    'batch_size':8,
    'radius_subdiv' : 32,
    'azimuth_subdiv' : 128,
    'dist_model': dist_model,
    'n_radius' : 2, 
    'n_azimuth' : 2,
    'img_size' : 128,
    'radius_buffer': 0, 
    'azimuth_buffer' : 0,
    'seed': 1234
    }
    
    snapshot_path="val_sparse_2_2"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    trainer_sparseCnn(args, model, snapshot_path)

    















