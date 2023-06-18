import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import json
from imageio.v2 import imread, imsave
import glob
import random

#to 1350,1350
def convert_img(path):
    im= imread(path)
    im=np.pad(im,((192,192),(35,35),(0,0)))
    x,y= np.meshgrid(np.arange(1350), np.arange(1350), indexing='ij')
    x-= 675
    y-=675
    r= np.sqrt(x**2 + y**2)
    im[r>675,:]=0
    imsave(path, im)

def convert_label(path):
    depth= np.load(path)
    depth=np.pad(depth,((192,192),(35,35)))
    x,y= np.meshgrid(np.arange(1350), np.arange(1350), indexing='ij')
    x-= 675
    y-=675
    r= np.sqrt(x**2 + y**2)
    depth[r>675]=0
    cv2.imwrite(path[:-3]+'exr',depth)


def convert_data(root_imgs, root_labels):
    for f in os.listdir(root_imgs):
        path= os.path.join(root_imgs,f )
        if 'BEV' in f :
            os.remove(path)
        else:
            convert_img(path)

    for f in os.listdir(root_labels):
        path= os.path.join(root_labels,f )
        if 'BEV' in f :
            pass
        else:
            convert_label(path)
        os.remove(path)

def woodscape_paths_to_json(root_path):
    dir_images = sorted(list(glob.glob(root_path +'/rgb_images/'+"*.png")))
    paths = [os.path.basename(paths) for paths in dir_images]
    train_file = os.path.join(root_path, "train.json")
    val_file = os.path.join(root_path, "val.json")

    tf_total = len(paths)
    percent = lambda part, whole: float(whole) / 100 * float(part)
    test_count = percent(10, tf_total)  # 2.5% of training data is allocated for validation as there is less data :D

    random.seed(777)
    test_frames = random.sample(paths, int(test_count))
    frames = set(paths) - set(test_frames)
    print(f'=> Total number of training frames: {len(frames)} and validation frames: {len(test_frames)}')

    with open(train_file, 'w') as tf:
        json.dump(sorted(frames), tf)

    with open(val_file, 'w') as vf:
        json.dump(sorted(test_frames), vf)

    print(f'Wrote {tf.name} and {vf.name} into disk')
    

if __name__ == "__main__":
    root_path= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape'
    root_imgs= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape/rgb_images'
    root_labels= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape/depth_maps'

    #convert_data(root_imgs, root_labels)
    woodscape_paths_to_json(root_path)