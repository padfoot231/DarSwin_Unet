echo "start test model"


python test_fish.py --dataset Synapse --cfg configs/exp1_64_1_tan.yaml --grp gp1 --sample tan --xi_value 0.2 \
    --is_saveni False --root_path $SLURM_TMPDIR/data/M3D_low --max_epochs 500 --output_dir /home/prongs/scratch/darswin_g_theta/grp1_tan_resume \
    --img_size 64 --base_lr 0.01 --batch_size 16 



# python train.py --dataset Synapse --cfg configs/exp1_64_1_tan.yaml --grp gp1 --sample tan \
#     --root_path $SLURM_TMPDIR/data/M3D_low --max_epochs 1000 \
#     --output_dir  /home/prongs/scratch/darswin_g_theta/grp1_tan_resume \
#     --resume /home/prongs/scratch/darswin_g_theta/grp1_tan_resume/epoch_499.pth \
#     --img_size 64 --base_lr 0.01 --batch_size 16
