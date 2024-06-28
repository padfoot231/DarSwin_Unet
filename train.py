import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
from pyinstrument import Profiler




parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/gel/usr/icshi/DATA_FOLDER/Synwoodscape', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')

#parser.add_argument('--list_dir', type=str,
#                    default='./lists/lists_Synapse', help='list dir')
#parser.add_argument('--num_classes', type=int,
#                    default=9, help='output channel of network')
parser.add_argument('--output_dir', default='./gp3_64_10_pre', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=128, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
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
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path)


#args.resume = '/gel/usr/icshi/Radial-transformer-Unet/gp3_64_10_pre_001_500/epoch_399.pth'
config = get_config(args)


if __name__ == "__main__":
    
    device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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
    #use_it to get the correct sparseConv 
    dataset_name = args.dataset
    

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    #args.num_classes = dataset_config[dataset_name]['num_classes']
    #args.root_path = dataset_config[dataset_name]['root_path']

    """
    args.output_dir= '/home-local2/icshi.extra.nobkp/experiments/dar_gp4_175_aug_mask_2'
    args.base_lr = 0.01
    args.max_epochs =500
    args.batch_size = 16
    """
   

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    #net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda("cuda:0")
    net = ViT_seg(config, img_size=args.img_size).to(device)
    net.load_from(config)

    """
    dist_model= config.MODEL.SWIN.DISTORTION
    if dist_model== 'polynomial':
        args.root_path= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape'
    else:
        args.root_path= '/home-local2/icshi.extra.nobkp/matterport/M3D_low' #'/gel/usr/icshi/Swin-Unet/data/M3D_low'
    """
    
    
    trainer = {'Synapse': trainer_synapse,}

    #p= Profiler()
    #p.start()
    trainer[dataset_name](args, net, args.output_dir, config)
    #p.stop()
    #print(p.output_text(unicode=True, color=True))