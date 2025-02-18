for i in $(seq 0.0 0.05 0.9)
do 
    echo $i
    python test_fish.py --dataset Synapse --cfg configs/exp1_64_2_theta.yaml --grp gp2 --sample theta --xi_value $i \
    --is_saveni False --root_path $SLURM_TMPDIR/data/M3D_low --max_epochs 500 --output_dir /home/prongs/scratch/darswin_g_theta/grp2_theta_resume \
    --img_size 64 --base_lr 0.01 --batch_size 16 
done