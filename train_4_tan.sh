#!/bin/bash

export WANDB_MODE="disabled"



python train.py --dataset Synapse --cfg configs/exp1_64_4_tan.yaml --grp gp4 --sample tan \
    --root_path $SLURM_TMPDIR/data/M3D_low --max_epochs 1000 \
    --output_dir  /home/prongs/scratch/darswin_g_theta/grp4_tan_resume \
    --resume /home/prongs/scratch/darswin_g_theta/grp4_tan_resume/epoch_499.pth \
    --img_size 64 --base_lr 0.01 --batch_size 16

        # --resume /home/prongs/scratch/darswin_g_theta/grp2_tan_resume/epoch_499.pth \
