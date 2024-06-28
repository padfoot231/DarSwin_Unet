
echo "start test model"


python test_fish.py --dataset Synapse --cfg configs/exp1_64.yaml \
    --is_saveni False  --max_epochs 500 --output_dir /home-local2/icshi.extra.nobkp/experiments/darswin_g_theta \
    --img_size 64 --base_lr 0.01 --batch_size 16 