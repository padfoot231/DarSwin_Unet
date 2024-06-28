#!/bin/bash

export WANDB_MODE="disabled"



python train.py --dataset Synapse --cfg configs/exp1_64.yaml \
    --root_path /home-local2/icshi.extra.nobkp/matterport/M3D_low --max_epochs 500 \
    --output_dir  /home-local2/icshi.extra.nobkp/experiments/darswin_g_theta \
    --img_size 64 --base_lr 0.01 --batch_size 16