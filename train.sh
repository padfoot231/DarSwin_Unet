#!/bin/bash


python train.py --dataset Synapse --cfg configs/swin_tiny_patch2_window4_64.yaml --root_path /home-local2/akath.extra.nobkp/woodscapes --max_epochs 10000 --output_dir /home-local2/akath.extra.nobkp/Swin-Unet/KNN  --img_size 128 --base_lr 0.05 --batch_size 8