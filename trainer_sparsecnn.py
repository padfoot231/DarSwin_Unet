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
from sparseCnn import SparseConvNet, sparseLoss, round_sample
from utils_rad import get_sample_params_from_subdiv, get_sample_locations
from datasets.utils import get_mask_wood


cuda_id= "cuda:2"

def get_sparse_label(label_batch, dist,subdiv, img_size, distortion_model,n_radius, n_azimuth, radius_buffer, azimuth_buffer,indices):
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
    H= indices.shape[0]
    grid_out= torch.cat((y_/(H//2), x_/(H//2)), dim=3)
    sampled_label= nn.functional.grid_sample(label_batch, grid_out, align_corners = True, mode='nearest')
    grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2
    sparse_batch = round_sample(grid, sampled_label, indices)

    return sparse_batch


def trainer_sparseCnn(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    batch_size = args['batch_size']
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args['root_path'], split="train", transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args['img_size'], args['img_size']])]))

    db_val = Synapse_dataset(base_dir=args['root_path'], split="train")
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args['seed'] + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    
    SLoss= sparseLoss()
    optimizer= optim.Adam(params= model.parameters(), betas=(0.9,0.999), lr=1e-3, weight_decay=5*1e-4)
    writer = SummaryWriter(snapshot_path + '/log')
    save_path= snapshot_path + '/validation'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    iter_num = 0
    max_epoch = args['max_epochs']
    max_iterations = max_epoch * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

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
    mask_p = get_mask_wood() #1 for periphery otherwise 0 (array)
    mask_p= torch.tensor(mask_p).unsqueeze(0).unsqueeze(0)

    for epoch_num in iterator:
        model.train()
        with tqdm(total=len(trainloader)) as pbar:
            for i_batch, sampled_batch in enumerate(trainloader):
                label_batch, dist = sampled_batch['label'], sampled_batch['dist']
                label_batch, dist = label_batch.cuda(cuda_id), dist.cuda(cuda_id)
                dist= dist.transpose(1,0)
                sparse_batch= get_sparse_label(label_batch, dist,subdiv, img_size, distortion_model,n_radius, n_azimuth, radius_buffer, azimuth_buffer,indices)
                B= sparse_batch.shape[0]
                masks_p= torch.repeat_interleave(mask_p, B, 0).cuda(cuda_id)
                #if label!=0 it will return 1 so the pixel is not missing  (periphery=0 => 0 so periphery will be considered missing)
                masks= (sparse_batch!=0).long()
                #add the periphery mask to exclude/include the periphery
                masks= masks+ masks_p
                outputs= model(sparse_batch.float(), masks.float())

                loss_masks= torch.where(masks==0,1,0)
                loss= SLoss(label_batch, outputs,loss_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

                iter_num = iter_num + 1

                #tensorboard
                #writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                #here to add visualization !!

                pbar.set_description('iteration %d : loss : %f' % (iter_num, loss.item()))
                # pbar.set_description("loss_ce %f" % loss_ce.item())
                pbar.update(1)
        model.eval()
        val_losses=[]
        with torch.no_grad():
            with tqdm(total=len(valloader)) as pbar_v:
                for i_batch, sampled_batch in enumerate(valloader):
                    label_batch, dist = sampled_batch['label'], sampled_batch['dist']
                    label_batch, dist = label_batch.cuda(cuda_id), dist.cuda(cuda_id)

                    dist= dist.transpose(1,0)
                    sparse_batch= get_sparse_label(label_batch, dist,subdiv, img_size, distortion_model,n_radius, n_azimuth, radius_buffer, azimuth_buffer,indices)
                    B= sparse_batch.shape[0]
                    masks_p= torch.repeat_interleave(mask_p, B, 0).cuda(cuda_id)
                    #if label!=0 it will return 1 so the pixel is not missing  (periphery=0 => 0 so periphery will be considered missing)
                    masks= (sparse_batch!=0).long()
                    #add the periphery mask to exclude/include the periphery
                    masks= masks+ masks_p
                    outputs= model(sparse_batch.float(), masks.float())
                    loss_masks= torch.where(masks==0,1,0)
                    loss_val= SLoss(label_batch, outputs, loss_masks)
                    val_losses.append(loss_val.item())
                    pbar_v.update(1)
            
            plt.imsave(save_path+ '/pred_{}.jpg'.format(epoch_num),outputs[0,...].squeeze(0).cpu().numpy())
            plt.imsave(save_path+ '/label_{}.jpg'.format(epoch_num),label_batch[0,...].squeeze(0).cpu().numpy())

            """
            image= image_batch[0,...].permute(1,2,0)
            image*= torch.tensor(std).cuda()
            image+= torch.tensor(mean).cuda()
            plt.imsave(save_path+ '/img_{}.jpg'.format(epoch_num), np.clip(image.cpu().numpy(),0,1) )
            """
            

            writer.add_scalar('info/total_loss_val', torch.mean(torch.tensor(val_losses)), epoch_num)
            logging.info('epoch  %d : loss : %f' % (epoch_num, torch.mean(torch.tensor(val_losses))))

        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    model= SparseConvNet().cuda(cuda_id)
    args={'root_path': '/gel/usr/icshi/DATA_FOLDER/Synwoodscape', 
    'batch_size':24,
    'max_epochs':1,
    'radius_subdiv' : 32,
    'azimuth_subdiv' : 128,
    'dist_model': "polynomial",
    'n_radius' : 2, 
    'n_azimuth' : 2,
    'img_size' : 128,
    'radius_buffer': 0, 
    'azimuth_buffer' : 0,
    'seed': 1234
    }
    
    snapshot_path="Sparse1"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    trainer_sparseCnn(args, model, snapshot_path)

    

