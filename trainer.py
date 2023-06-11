import argparse
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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
import wandb
from utils import test_single_volume

wandb.init(project="Distortion", entity='padfoot')
def resample(samples, n_radius, n_azimuth, B):
    patch_azr = []
    patch_azg = []
    patch_azb = []
    # patch = torch.empty(n_radius, B, 1, 16384, 5)
    for i in range(n_radius):
        patch_azr.append(samples[:, :, :, i*n_radius:i*n_radius + n_radius])
    pr = []
    pg = []
    pb = []
    for patch in patch_azr:
        pr.append(torch.flip(patch, (3, )))
    ar = torch.cat((pr[0], pr[1]), 3)
    for i in range(len(pr)-2):
        ar = torch.cat((ar, pr[i+2]), 3)
    br = ar.reshape(B, 10, -1)
    imr = torch.zeros((B, 10, 32*n_radius, 128*n_azimuth))
    for i in range(128):
        k = n_azimuth*n_radius*32
        imr[:, :, :, i*n_azimuth:i*n_azimuth + n_azimuth] = br[:, :, i*k:i*k + k].reshape(B, 10, n_radius*32, n_azimuth)
    imr = imr.cuda("cuda:1")
    x = torch.linspace(-127, 127, 128)
    y = torch.linspace(-127, 127, 128)
    meshgrid = torch.meshgrid(x, y)
    x = meshgrid[0]
    y = meshgrid[1]
    r = torch.sqrt(x**2+y**2)
    t = torch.atan2(y,x)
    x = r/64 -1
    y = t/(np.pi) 
    x_ = torch.reshape(x, (128, 128, 1))
    y_ = torch.reshape(y, (128, 128, 1))
    grid = torch.stack((y, x), 2).reshape(1, 128, 128, 2).cuda("cuda:1")
    grid_ = torch.repeat_interleave(grid, B, 0).cuda("cuda:1")
    outr = torch.nn.functional.grid_sample(imr, grid_,  align_corners=True, mode='nearest')
    return outr

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, split="train")
    db_val = Synapse_dataset(base_dir=args.root_path, split="test")
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=2, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        with tqdm(total=len(trainloader)) as pbar:
            for i_batch, sampled_batch in enumerate(trainloader):
                image_batch, label_batch, dist = sampled_batch['image'], sampled_batch['label'], sampled_batch['dist']
                image_batch, label_batch, dist = image_batch.cuda("cuda:1"), label_batch.cuda("cuda:1"), dist.cuda("cuda:1")
                outputs, label_batch = model(image_batch, dist, label_batch)
                B, _, _, _ = image_batch.shape
                # breakpoint()
                # outputs = resample(outputs, 4, 4, B)
                # breakpoint()
                loss_ce = ce_loss(outputs, label_batch.long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.9 * loss_ce + 0.1 * loss_dice
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_
                # import pdb;pdb.set_trace()
                iter_num = iter_num + 1
                # writer.add_scalar('info/lr', lr_, iter_num)
                # writer.add_scalar('info/total_loss', loss, iter_num)
                # writer.add_scalar('info/loss_ce', loss_ce, iter_num)

                # logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
                pbar.set_description('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
                wandb.log({"loss_train" : loss.item(), "loss_ce":loss_ce.item(), "epoch" : epoch_num})
                # pbar.set_description("loss_ce %f" % loss_ce.item())
                pbar.update(1)
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(valloader)) as pbar_v:
                for i_batch, sampled_batch in enumerate(valloader):
                    image_batch, label_batch, dist = sampled_batch['image'], sampled_batch['label'], sampled_batch['dist']
                    image_batch, label_batch, dist = image_batch.cuda("cuda:1"), label_batch.cuda("cuda:1"), dist.cuda("cuda:1")
                    outputs, label_batch = model(image_batch, dist, label_batch)
                    B, _, _, _ = image_batch.shape
                    # breakpoint()
                    # outputs = resample(outputs, 4, 4, B)
                    loss_ce_val = ce_loss(outputs, label_batch.long())
                    loss_dice_val = dice_loss(outputs, label_batch, softmax=True)
                    loss_val = 0.9 * loss_ce_val + 0.1 * loss_dice_val

                    # writer.add_scalar('info/total_loss_val', loss, iter_num)
                    # writer.add_scalar('info/loss_ce_val', loss_ce, iter_num)
                    pbar_v.set_description('loss : %f, loss_ce: %f' % (loss_val.item(), loss_ce_val.item()))
                    wandb.log({"loss_val" : loss_val.item(), "loss_ce_val":loss_ce_val.item(), "epoch" : i_batch})
                    # pbar.set_description("loss_ce_val %f" % loss_ce_val.item())
                    pbar_v.update(1)

                # logging.info('val_iteration %d : val_loss : %f, val_loss_ce: %f' % (iter_num, loss_val.item(), loss_ce_val.item()))

                # if iter_num % 20 == 0:
                #     image = image_batch[1, 0:1, :, :]
                #     image = (image - image.min()) / (image.max() - image.min())
                #     # writer.add_image('train/Image', image, iter_num)
                #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                #     # writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                #     labs = label_batch[1, ...].unsqueeze(0) * 50
                    # writer.add_image('train/GroundTruth', labs, iter_num) 
        # breakpoint()
        save_interval = 10  # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        if (epoch_num + 1) % save_interval == 0:
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