#!/bin/bash

export WANDB_MODE="disabled"



python train.py --dataset Synapse --cfg configs/exp1_64_1.yaml --grp gp1\
    --root_path $SLURM_TMPDIR/data/M3D_low --max_epochs 501 \
    --output_dir  /home/prongs/scratch/darswin_g_theta/grp1_resume \
    --img_size 64 --base_lr 0.01 --batch_size 16


#    --resume /home/prongs/scratch/darswin_g_theta/grp2_resume/epoch_499.pth \
