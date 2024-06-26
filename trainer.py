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
from utils import MDELoss
from torchvision import transforms
from networks.visualize import sample2pixel 
import matplotlib.pyplot as plt 
from datasets.utils import get_mask_wood, get_mask_matterport
from sparseCnn import SparseConvNet, sparseLoss
from sparseCnn import round_sample_opt 
from pyinstrument import Profiler 
from knn import restruct, get_grid_pix, KNN


cuda_id="cuda:1"

#max_iter= 200*len(trainloader)

#p= Profiler()
def trainer_synapse(args, model, snapshot_path, config ):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    #num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    dist_model= config.MODEL.SWIN.DISTORTION
    db_train = Synapse_dataset(base_dir=args.root_path, split="train", model= dist_model, transform =# None)
                                   transforms.Compose(
                                  [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_val = Synapse_dataset(base_dir=args.root_path, model=dist_model, split="val")
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    #ce_loss = CrossEntropyLoss()
    #dice_loss = DiceLoss(num_classes)
    mde_loss= MDELoss()
    writer = SummaryWriter(snapshot_path + '/log')
    save_path= snapshot_path + '/validation'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    iter_num = 0
    init_epoch = 0
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    #2296= 9180(training img)/batch_size * (2->10)
    #scheduler= optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=base_lr, step_size_up= 1148*5, mode='triangular2')

    resume_path = config.MODEL.RESUME

    if resume_path is not None:
        print("resuming from ", resume_path)
        device = torch.device(cuda_id if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(pretrained_dict['model_state_dict'])
        init_epoch = pretrained_dict['epoch'] +1
        iter_num = pretrained_dict['iter'] +1
        optimizer.load_state_dict(pretrained_dict['optimizer_state_dict'])
        #scheduler= optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=base_lr, step_size_up= 2296, mode='triangular')
        #last_lr= pretrained_dict['lr']

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1

    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(init_epoch,max_epoch), ncols=70)

    
    #ckpt= config.MODEL.SPARSE_CNN_CKPT
    #sparseNet= SparseConvNet().cuda(cuda_id)
    #sparseNet.load_from(ckpt)
    #for param in sparseNet.parameters():
    #    param.requires_grad=False

    H = args.img_size 
    #indices= torch.arange(H*H).reshape(H,H).cuda(cuda_id)
    if dist_model=='spherical':
        #mask_p= get_mask_matterport()
        #Matterport
        mean= [0.2217, 0.1939, 0.1688]
        std= [0.1884, 0.1744, 0.1835]

    else:
        #mask_p = get_mask_wood() #get_mask_matterport() #1 for periphery otherwise 0 (array)
        #woodscape
        mean= [0.1867, 0.1694, 0.1573]
        std= [0.2172, 0.2022, 0.1884]

    #mask_p= torch.tensor(mask_p).unsqueeze(0).unsqueeze(0)
    #mask_p= torch.where(mask_p==1,0,1)
    #grid_pix= get_grid_pix(H)

    s=time.time()
    for epoch_num in iterator:
        model.train()
        with tqdm(total=len(trainloader)) as pbar:
            for i_batch, sampled_batch in enumerate(trainloader):
                image_batch, label_batch, mask_batch, dist, cl = sampled_batch['image'], sampled_batch['label'],sampled_batch['mask'], sampled_batch['dist'],sampled_batch['cl'] 
                if 0 in  torch.count_nonzero(mask_batch,dim=(-2,-1)):
                    continue 

                image_batch, label_batch, mask_batch, dist, cl = image_batch.cuda(cuda_id), label_batch.cuda(cuda_id),mask_batch.cuda(cuda_id), dist.cuda(cuda_id), cl.cuda(cuda_id)
                
                
                outputs = model(image_batch, dist, cl)
                #mask
                outputs= torch.where(label_batch==0, label_batch, outputs)

                #sparse_batch = round_sample_opt(grid, outputs, indices)
                #B, dim = outputs.shape[0], outputs.shape[1]
                #grid= grid.view(B,-1,2)

                #grid_pixel= torch.repeat_interleave(grid_pix, B, 0).cuda(cuda_id)
                #masks_p= torch.repeat_interleave(mask_p, B, 0).cuda(cuda_id)
                #B, N, D, P, k = grid.shape[0], grid.shape[1], 2, grid_pix.shape[1], 4
                #cl = KNN(grid/(H//2), grid_pixel/(H//2), P, k)
                #torch.save(cl, 'cl.pt')
                
                #outputs = restruct(outputs, cl , dim, H, H)
                #outputs = outputs* masks_p.float() 
            

            
                
                #if label!=0 it will return 1 so the pixel is not missing  (periphery=0 => 0 so periphery will be considered missing)
                #masks= (sparse_batch!=0).long()
                #add the periphery mask to exclude/include the periphery
                #masks= masks+ masks_p
                #masks[masks>1]=1

                #outputs= sparseNet(sparse_batch.float(), masks.float())

                #print(time.time()-s)
                loss= mde_loss(outputs,label_batch, mask_batch)


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1

                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss, iter_num)
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                #here to add visualization !!

                pbar.set_description('iteration %d : loss : %f' % (iter_num, loss.item()))
                # pbar.set_description("loss_ce %f" % loss_ce.item())
                pbar.update(1)
            
            if (epoch_num+1) % 100==0:
                masks= mask_batch[0,...].squeeze(0).cpu().numpy()
                plt.imsave(save_path +'/mask_{}.png'.format(epoch_num), masks)

                image= image_batch[0,...].permute(1,2,0)
                image*= torch.tensor(std).cuda(cuda_id)
                image+= torch.tensor(mean).cuda(cuda_id)
                plt.imsave(save_path+ '/img_{}.png'.format(epoch_num), np.clip(image.cpu().numpy(),0,1) )
                lab= label_batch[0,...].squeeze(0).cpu().numpy()
                plt.imsave(save_path+ '/label_{}.png'.format(epoch_num),lab)
                pred= outputs[0,...].squeeze(0).cpu().detach().numpy()
                plt.imsave(save_path+ '/pred_{}.png'.format(epoch_num),pred)
            
    
        model.eval()
        val_losses=[]
        with torch.no_grad():
            with tqdm(total=len(valloader)) as pbar_v:
                for i_batch, sampled_batch in enumerate(valloader):
                    image_batch, label_batch, mask_batch, dist, cl = sampled_batch['image'], sampled_batch['label'], sampled_batch['mask'], sampled_batch['dist'],sampled_batch['cl'] 
                    if 0 in  torch.count_nonzero(mask_batch,dim=(-2,-1)):
                        continue 
                    image_batch, label_batch, mask_batch, dist, cl = image_batch.cuda(cuda_id), label_batch.cuda(cuda_id), mask_batch.cuda(cuda_id), dist.cuda(cuda_id), cl.cuda(cuda_id)
                    outputs = model(image_batch, dist, cl)
                    outputs= torch.where(label_batch==0, label_batch, outputs)
                    loss_val= mde_loss(outputs,label_batch,mask_batch)
                    val_losses.append(loss_val.item())
                    pbar_v.update(1)
                
                
                if (epoch_num+1) % 100==0:
                    image= image_batch[0,...].permute(1,2,0)
                    image*= torch.tensor(std).cuda(cuda_id)
                    image+= torch.tensor(mean).cuda(cuda_id)
                    plt.imsave(save_path+ '/val_img_{}.png'.format(epoch_num), np.clip(image.cpu().numpy(),0,1) )
                    lab= label_batch[0,...].squeeze(0).cpu().numpy()
                    plt.imsave(save_path+ '/val_label_{}.png'.format(epoch_num),lab)
                    pred= outputs[0,...].squeeze(0).cpu().detach().numpy()
                    plt.imsave(save_path+ '/val_pred_{}.png'.format(epoch_num),pred)
                    masks= mask_batch[0,...].squeeze(0).cpu().numpy()
                    plt.imsave(save_path +'/val_mask_{}.png'.format(epoch_num), masks)
                
            writer.add_scalar('info/total_loss_val', torch.mean(torch.tensor(val_losses)), epoch_num)
            logging.info('epoch  %d : val_loss : %f' % (epoch_num, torch.mean(torch.tensor(val_losses))))
        
        save_interval = 50
        #if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if epoch_num > 0 and (epoch_num + 1) % save_interval == 0:

            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'lr': lr_,
            'iter': iter_num - 1
            }, save_mode_path)
            #torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'lr': lr_,
            'iter': iter_num - 1
            }, save_mode_path)
            #torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    
    print(time.time()-s)
    writer.close()
    #print(p.output_text(unicode=True, color=True))
    return "Training Finished!"

