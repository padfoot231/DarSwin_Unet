import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.utils import matterport_test
from utils import eval_depth, mask_fish
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import cv2
import matplotlib.pyplot as plt
import csv
from calcul_matrix import calcul_cl

parser = argparse.ArgumentParser()
#parser.add_argument('--volume_path', type=str,
#                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Matterport', help='experiment_name')

parser.add_argument('--root_path', type=str,
                    default='/home-local2/icshi.extra.nobkp/matterport/M3D_low', help='root dir for data')

#parser.add_argument('--num_classes', type=int,
#                   default=9, help='output channel of network')
#parser.add_argument('--list_dir', type=str,
#                   default='./lists/lists_Synapse', help='list dir')


parser.add_argument('--output_dir', type=str, default= "/gel/usr/icshi/Results", help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=64, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true",  help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.005, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

#parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')

parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

def plot_result(result_dict):
    line = "\n"
    line += "=" * 100 + '\n'
    for metric, value in result_dict.items():
        line += "{:>10} ".format(metric)
    line += "\n"
    for metric, value in result_dict.items():
        line += "{:10.4f} ".format(value)
    line += "\n"
    line += "=" * 100 + '\n'

    return line

def inference(args, model, test_save_path=None,save_metrics_path=None, xi_value=None, max_depth=8.0):
    metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']
    
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    db_test = args.Dataset(base_dir=args.root_path, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()


    mean= [0.2217, 0.1939, 0.1688]
    std= [0.1884, 0.1744, 0.1835]
    nb= 0 
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, dist, cl, mask   = sampled_batch["image"], sampled_batch["label"], sampled_batch["dist"], sampled_batch["cl"], sampled_batch['mask']
        
        if torch.count_nonzero(mask).item()==0:
            continue

        #image, label = image.cpu().detach(), label.cpu().detach()
        with torch.no_grad():
            image, dist, cl, label, mask = image.cuda(cuda_id), dist.cuda(cuda_id), cl.cuda(cuda_id), label.cuda(cuda_id), mask.cuda(cuda_id)
            pred= model(image, dist, cl)
            max_depth_tensor= torch.tensor(max_depth + 1e-6, dtype=torch.float32,device=torch.device(cuda_id))
            pred = torch.where(mask==0, max_depth_tensor, pred)
            pred= torch.where(label==0,label, pred)
        
        pred, label, image , dist, mask = pred.cpu().detach(), label.cpu().detach(), image.cpu().detach(), dist.cpu(), mask.cpu()
        pred, label, image, dist, mask  = pred.squeeze(), label.squeeze(), image.squeeze(), dist.squeeze(0), mask.squeeze()


        #new_fov=120
        #pred = mask_fish(pred.unsqueeze(2).numpy(), dist[1], dist[0], new_fov)
        #label = mask_fish(label.unsqueeze(2).numpy(), dist[1], dist[0], new_fov)

        computed_result = eval_depth(pred, label, mask)
        for metric in metric_name:
            result_metrics[metric] += computed_result[metric]

        nb = nb + 1 
        if test_save_path is not None and i_batch % 200==0:
            pred_path = os.path.join(test_save_path,'pred_{}_{}.png'.format(i_batch, xi_value))
            gt_path = os.path.join(test_save_path,'gt_{}_{}.png'.format(i_batch, xi_value))
            img_path = os.path.join(test_save_path,'img_{}_{}.png'.format(i_batch, xi_value))
            
            image= image.permute(1,2,0)
            image*= torch.tensor(std)
            image+= torch.tensor(mean)

            #print(torch.max(image),torch.min(image))

            plt.imsave(pred_path, pred.numpy())
            plt.imsave(img_path, np.clip(image.numpy(),0,1))
            plt.imsave(gt_path,label.numpy())


    for key in result_metrics.keys():
            #result_metrics[key] = result_metrics[key] / (i_batch+ 1)
            result_metrics[key] = result_metrics[key] / nb 
    display_result = plot_result(result_metrics)

    if save_metrics_path is not None:
        if not os.path.exists(save_metrics_path):
            headerList = ['xi'] + metric_name
            with open(save_metrics_path, 'a') as file:
                dw = csv.DictWriter(file, delimiter=',', fieldnames=headerList)
                dw.writeheader()

        infile = open(save_metrics_path, "a")
        writer = csv.writer(infile)
        data= [xi_value]+ list(result_metrics.values())
        writer.writerow(data)

    print(display_result)

    print("Done")

if __name__ == "__main__":

    cuda_id= "cuda:1"
    """
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    """

    dataset_config = {
        'Matterport': {
            'Dataset': Synapse_dataset,
            'root_path': args.root_path,
        },
    }
    dataset_name = args.dataset
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size).cuda(cuda_id)

    #snapshot = os.path.join(args.output_dir, 'best_model.pth')
    snapshot= config.TEST.CKPT 
    #if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    pretrained_dict= torch.load(snapshot, map_location = cuda_id)
    msg = net.load_state_dict(pretrained_dict['model_state_dict'])
    print("self trained swin unet",msg)


    args.is_savenii= False  
    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    save_metrics_path= os.path.join(args.output_dir,'dar_gp4_175.csv')
    for xi_value in np.arange(0,105,5)/100:
        matterport_test(args.root_path,xi_value, xi_value)
        calcul_cl(args.root_path)
        inference(args, net, test_save_path,save_metrics_path,xi_value)


