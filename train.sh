#!/bin/bash
if [$epoch_time]
then
    EPOCH_TIME = $epoch_time
else
    EPOCH_TIME = 150
fi

if [$out_dir]
then
    OUT_DIR = $out_dir
else
    OUT_DIR  = './model_out'
fi

if [$cfg]
then
    CFG = $cfg
else
    CFG = 'configs/swin_tiny_patch4_window7_224_lite.yaml'
fi

if [$data_dir]
then
    DATA_DIR = $data_dir
else
    DATA_DIR = 'datasets/Synapse'
fi

if [$learning_rate]
then
    LEARNING_RATE = $learning_rate
else
    LEARNING_RATE = 0.05
fi

if [$img_size]
then
    IMG_SIZE = $img_size
else
    IMG_SIZE = 224
fi

if [$batch_size]
then
    BATCH_SIZE = $batch_size
else
    BATCH_SIZE = 24
fi

echo "start train model"
pyhton train.py --dataset Synapse --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE

/home-local2/akath.extra.nobkp/woodscapes
/gel/usr/akath/Swin-Unet/output


python train.py --dataset Synapse --cfg configs/swin_tiny_patch2_window4_64.yaml --root_path /home-local2/akath.extra.nobkp/woodscapes --max_epochs 10000 --output_dir /home-local2/akath.extra.nobkp/Swin-Unet/KNN  --img_size 128 --base_lr 0.05 --batch_size 8