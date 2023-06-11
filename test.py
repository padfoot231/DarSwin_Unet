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
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
import json
from torchvision.transforms import transforms
from torchmetrics import JaccardIndex
import torch.nn.functional as F
from PIL import Image
import pickle as pkl

trans = transforms.ToPILImage()

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
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
parser.add_argument('--type', type=str, default='swin', help='type of networks')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path)
config = get_config(args)
jaccard = JaccardIndex(num_classes=10, task = 'multiclass')
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
    br = ar.view(B, 1, -1)
    imr = torch.zeros((B, 1, 32*n_radius, 128*n_azimuth))
    for i in range(128):
        k = n_azimuth*n_radius*32
        imr[:, :, :, i*n_azimuth:i*n_azimuth + n_azimuth] = br[:, :, i*k:i*k + k].reshape(B, 1, n_radius*32, n_azimuth)
    imr = imr.cuda("cuda:0")
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
    grid = torch.stack((y, x), 2).reshape(1, 128, 128, 2).cuda("cuda:0")
    # grid_ = torch.repeat_interleave(grid, 1, 0).cuda("cuda:0")
    outr = torch.nn.functional.grid_sample(imr, grid,  align_corners=True, mode='nearest')
    return outr

def inference(args, model, test_save_path=None, seg_info=None):
    palette = seg_info['class_colors']
    pal = [0,
            0,
            0,
            255,
            0,
            255,
            255,
            0,
            0,
            0,
            255,
            0,
            0,
            0,
            255,
            255,
            255,
            255,
            255,
            255,
            0,
            0,
            255,
            255,
            128,
            128,
            255,
            0,
            128,
            128]
    pal = [[0, 0, 0],
            [255, 0, 255],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [255, 255, 0],
            [0, 255, 255],
            [128, 128, 255],
            [0, 128, 128]]
    db_test = args.Dataset(base_dir=args.volume_path, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        # breakpoint()
        image, label, dist = sampled_batch["image"], sampled_batch["label"], sampled_batch['dist']
        image, label, dist = image.cuda("cuda:0"), label.cuda("cuda:0"), dist.cuda("cuda:0")
        output, label = model(image, dist, label)
        if args.type == 'swin':
            pred = output.argmax(1)
            pred_on = pred.detach().cpu().type(torch.LongTensor)
            pred_on = F.one_hot(pred_on, num_classes=10) .transpose(1, 2).transpose(0, 1)
            lab_on = label.detach().cpu().type(torch.LongTensor) 
            lab_on = F.one_hot(lab_on, num_classes=10).transpose(1, 2).transpose(0, 1)
            lab = label
        elif args.type == 'darswin':
            pred = output.argmax(1).reshape(1, 1, 4096, 16)
            with open('sample.pkl', 'rb') as f:
                data = pkl.load(f)
            data = data.unsqueeze(0)
            B, _, _, _ = pred.shape
            import pdb;pdb.set_trace()
            pred = resample(pred, 4, 4, B)
            pred_on = pred.detach().cpu().type(torch.LongTensor)
            pred_on = F.one_hot(pred_on, num_classes=10) .transpose(1, 2).transpose(0, 1)
            lab = resample(label, 4, 4, B)
            lab_on = lab.detach().cpu().type(torch.LongTensor) 
            lab_on = F.one_hot(lab_on, num_classes=10).transpose(1, 2).transpose(0, 1)
        miou = jaccard(pred_on, lab_on)
        breakpoint()
        # prediction = Image.fromarray(pred[0].astype(np.uint8)).convert('P')
        prediction = trans(pred[0].type(torch.int32)).convert('P')
        # prediction.putpalette(pal)
        prediction.putpalette(np.array(pal, dtype=np.uint8))
        prediction.save('val/pred' + str(miou)+ str(i_batch) + '.png')
        lab = trans(lab[0].type(torch.int32)).convert('P')
        # lab = Image.fromarray(lab[0].astype(np.uint8)).convert('P')
        lab.putpalette(np.array(pal, dtype=np.uint8))
        lab.save('val/label'+ str(miou) + str(i_batch)+ '.png')


    #     metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
    #                                   test_save_path=test_save_path, z_spacing=args.z_spacing)
    #     metric_list += np.array(metric_i)
    #     logging.info('idx %d mean_dice %f mean_hd95 %f' % (i_batch, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    # metric_list = metric_list / len(db_test)
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

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

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 10,
            'z_spacing': 1,
        },
    }
    # dataset_config = {
    #     'Synapse': {
    #         'volume_path': args.volume_path,
    #         'num_classes': 10,
    #     },
    # }

    with open('/home-local2/akath.extra.nobkp/woodscapes/seg_annotation_info.json', 'r') as f:
        seg_info = json.load(f)


    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda("cuda:0")

    snapshot = os.path.join(args.output_dir, 'epoch_209.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path, seg_info)


