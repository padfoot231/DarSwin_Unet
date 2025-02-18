import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import json
from imageio.v2 import imread, imsave
import glob
import random
from skimage.transform import resize 
import random 
import pickle as pkl 
try:
    from .dataset_synapse import warpToFisheye
except:
    pass


#to be updated
def get_mask_matterport():
    img= np.ones((256,512,1))
    fov=150
    xi=0.5
    h=img.shape[0]
    img,_ = warpToFisheye(img, outputdims=(h,h),xi=xi, fov=fov, order=0)
    img= resize(img,(128,128), order=0)
    img= img.reshape(128,128)

    img= np.where(img< 0.5, 1, 0)
    plt.imsave("maskM.png", img , cmap="gray")
    return img

def get_mask_wood():
    mask= np.ones((966,1280))
    mask=np.pad(mask,((192,192),(35,35)))
    x,y= np.meshgrid(np.arange(1350), np.arange(1350), indexing='ij')
    x-= 675
    y-=675
    r= np.sqrt(x**2 + y**2)
    mask[r>675]=0
    mask= resize(mask,(128,128), order=0)
    mask= np.where(mask< 0.5, 1, 0)
    plt.imsave("mask.png", mask, cmap="gray")
    return mask 
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


def matterport_paths_to_json(root_path, low_train=0.5, high_train=0.7, low_val=0.5, high_val=0.7 , p=15):

    dir_images = sorted(list(glob.glob(root_path +'/**/*.png', recursive=True)))
    paths = [os.path.basename(os.path.dirname(paths))+'/'+os.path.basename(paths) for paths in dir_images]
    train_file = os.path.join(root_path, "train_gp3.json")
    val_file = os.path.join(root_path, "val_gp3.json")
    calib={}

    tf_total = len(paths)
    percent = lambda part, whole: float(whole) / 100 * float(part)
    test_count = percent(p, tf_total)  # 2.5% of training data is allocated for validation as there is less data :D

    random.seed(777)
    test_frames = random.sample(paths, int(test_count))
    frames = set(paths) - set(test_frames)
    print(f'=> Total number of training frames: {len(frames)} and validation frames: {len(test_frames)}')

    with open(train_file, 'w') as tf:
        json.dump(sorted(frames), tf)

    with open(val_file, 'w') as vf:
        json.dump(sorted(test_frames), vf)

    for i, f in enumerate(frames):
        xi = random.uniform(low_train,high_train)
        calib[f]= np.array([xi])
    for i, f in enumerate(test_frames):
        xi = random.uniform(low_train,high_train)
        calib[f]= np.array([xi])

    with open(root_path +'/calib_gp3.pkl', 'wb') as f:
        pkl.dump(calib, f)
    
    print(f'Wrote {tf.name} and {vf.name} into disk')

def matterport_test(root_path,low, high):
    
    with open(root_path + '/val_gp4.json', 'r') as f:
        data = json.load(f)
    
    calib={}
    for i, f in enumerate(data):
        #low, high = test_range[test_gp]
        xi = random.uniform(low,high)
        calib[f]= np.array([xi])

    with open(root_path +'/test_calib.pkl', 'wb') as f:
        pkl.dump(calib, f)
    
    print("len test data {}".format(len(data)))
    

if __name__ == "__main__":
    root_path= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape'
    root_imgs= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape/rgb_images'
    root_labels= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape/depth_maps'

    #convert_data(root_imgs, root_labels)
    #woodscape_paths_to_json(root_path)

    #for matterport, you should find the mask with invalid regions+periphery in the dataloader
    #mask= get_mask_wood()

    root_path= '/gel/usr/icshi/Swin-Unet/data/M3D_low'
    matterport_paths_to_json(root_path)
    #get_mask_matterport()
    