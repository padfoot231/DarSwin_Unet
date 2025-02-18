# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .radial_swin_transformer_unet import swin_transformer_angular as  SwinTransformerAng
from .radial_swin_transformer_unet_tan import swin_transformer_angular_tan as  SwinTransformerAng_tan
from .radial_swin_transformer_unet_theta import swin_transformer_angular_theta as  SwinTransformerAng_theta


logger = logging.getLogger(__name__)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cuda_id = "cuda:0"

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        if config.MODEL.TYPE == 'darswin_az_tan':
            self.swin_unet = SwinTransformerAng_tan(img_size=config.DATA.IMG_SIZE,
                        radius_cuts=config.MODEL.SWIN.RADIUS_CUTS, 
                        azimuth_cuts=config.MODEL.SWIN.AZIMUTH_CUTS,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        max_depth = config.DATA.MAX_DEPTH,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        distortion_model=config.MODEL.SWIN.DISTORTION, 
                        window_size=config.MODEL.SWINAZ.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        n_radius = config.MODEL.SWIN.N_RADIUS,
                        n_azimuth = config.MODEL.SWIN.N_AZIMUTH)
        elif config.MODEL.TYPE == 'darswin_az_theta':
            self.swin_unet = SwinTransformerAng_theta(img_size=config.DATA.IMG_SIZE,
                        radius_cuts=config.MODEL.SWIN.RADIUS_CUTS, 
                        azimuth_cuts=config.MODEL.SWIN.AZIMUTH_CUTS,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        max_depth = config.DATA.MAX_DEPTH,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        distortion_model=config.MODEL.SWIN.DISTORTION, 
                        window_size=config.MODEL.SWINAZ.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        n_radius = config.MODEL.SWIN.N_RADIUS,
                        n_azimuth = config.MODEL.SWIN.N_AZIMUTH)
        elif config.MODEL.TYPE == 'darswin_az':
            self.swin_unet = SwinTransformerAng(img_size=config.DATA.IMG_SIZE,
                        radius_cuts=config.MODEL.SWIN.RADIUS_CUTS, 
                        azimuth_cuts=config.MODEL.SWIN.AZIMUTH_CUTS,
                        in_chans=config.MODEL.SWIN.IN_CHANS,
                        max_depth = config.DATA.MAX_DEPTH,
                        embed_dim=config.MODEL.SWIN.EMBED_DIM,
                        depths=config.MODEL.SWIN.DEPTHS,
                        num_heads=config.MODEL.SWIN.NUM_HEADS,
                        distortion_model=config.MODEL.SWIN.DISTORTION, 
                        window_size=config.MODEL.SWINAZ.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                        qk_scale=config.MODEL.SWIN.QK_SCALE,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN.APE,
                        patch_norm=config.MODEL.SWIN.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                        n_radius = config.MODEL.SWIN.N_RADIUS,
                        n_azimuth = config.MODEL.SWIN.N_AZIMUTH)
    
    def forward(self, x, dist, cl=None):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        #change to swin !! 
        logits = self.swin_unet(x,dist,cl)
        return logits 

    def load_from(self, config):
        # breakpoint()
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cuda:0'))
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            #print(pretrained_dict.keys())
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")
            # breakpoint()
            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]
            # breakpoint()
            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
 