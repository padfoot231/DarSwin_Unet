import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import random
import numpy as np
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from skimage.transform import resize
import torchvision
import json


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

#'png'
def load_color(filename: str) -> torch.Tensor:    
    return {'color': torchvision.io.read_image(filename) / 255.0 }    

#'depth' and 'exr'
def load_depth(filename: str, max_depth: float=1000.0) -> torch.Tensor:
    depth_filename = filename.replace('.png', '.exr')
    depth = torch.from_numpy(
        cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    ).unsqueeze(0)
    #NOTE: add a micro meter to allow for thresholding to extact the valid mask
    depth[depth > max_depth] = max_depth + 1e-6
    return {
        'depth': depth
    }


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, c  = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label}
        return sample

#distortion params: polynomial K1,k2,k3,k4
"""
FV: Front CAM
RV: Rear CAM
MVL: Mirror Left CAM
MVR: Mirror Right CAM
"""
Distortion= {
'MVL': np.array([342.234, -18.6659, 23.1572, 4.28064]),
'FV' : np.array([341.725, -26.4448, 32.7864, 0.50499]),
'MVR' : np.array([340.749, -16.9704, 20.9909, 4.60924]),
'RV' : np.array([342.457, -22.4772, 28.5462, 1.78203])
}


mean= [0.1867, 0.1694, 0.1573]
std= [0.2172, 0.2022, 0.1884]

#normalize = None
normalize= transforms.Normalize(
            mean= mean ,
            std= std )
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        
        if split == 'train':
            with open(base_dir + '/train.json', 'r') as f:
                data = json.load(f)
        elif split == 'val':
            with open(base_dir + '/val.json', 'r') as f:
                data = json.load(f)
        elif split == 'test':
            with open(base_dir + '/test.json', 'r') as f:
                data = json.load(f)

        self.data = data[:10]
        self.data_dir = base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        b_path= self.data[idx]
        img_path = self.data_dir + '/rgb_images/' + b_path
        depth_path = self.data_dir + '/depth_maps/' + b_path.replace('png','exr')

        image= load_color(img_path)['color']
        depth= load_depth(depth_path)['depth']

        image=image.permute(1,2,0)
        depth=depth.permute(1,2,0)

        dist= Distortion[(b_path.split('.')[0]).split('_')[1]]

        #resizing to image_size
        image = resize(image,(128,128), order=1)
        label= resize(depth,(128,128), order=0)
        
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['image']= torch.from_numpy(image.astype(np.float32)) 
            sample['label']= torch.from_numpy(label.astype(np.float32))
        
        
        sample['image']= sample['image'].squeeze(0).permute(2,0,1)
        sample['label']= sample['label'].squeeze(0).permute(2,0,1)

        sample['dist'] = dist
        #print(sample['dist'])
        

        #sample['label']= sample['label'].squeeze(0)

        if normalize is not None:
            sample['image']= normalize(sample['image'])
        return sample

def get_mean_std(base_dir ):
    db= Synapse_dataset(base_dir, split="train", transform=None)
    print(len(db))
    #sample= db.__getitem__(0)
    #print(sample['image'].shape)
    #print(sample['label'].shape)
    #print(sample['dist'].shape)
    loader = DataLoader(db, batch_size=len(db), shuffle=False,num_workers=0)
    im_lab_dict = next(iter(loader))
    images, labels = im_lab_dict['image'], im_lab_dict['label']
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    print("mean",mean)
    print("std", std)
    return mean , std


if __name__ == "__main__":
    root_path= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape'
    mean,std= get_mean_std(root_path)

    """
    #verify max_depth
    root_labels= '/gel/usr/icshi/DATA_FOLDER/Synwoodscape/depth_maps'
    max_d=0
    for f in os.listdir(root_labels):
        path= os.path.join(root_labels,f)
        lab= cv2.imread(path,cv2.IMREAD_ANYDEPTH)
        if lab.max()> max_d:
                max_d=lab.max()
    print("max depth ", max_d)
    """


    


"""
import os
import random
import h5py
import numpy as np
import torch
from torchvision import transforms
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import json
from PIL import Image
import pickle as pkl

normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    img = normalize(torch.from_numpy(img.copy()))
    return img

def segm_transform(segm):
    # to tensor, -1 to 149
    segm = torch.from_numpy(np.array(segm)).long()
    return segm



class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        
        if split == 'train':
            with open(base_dir + '/train.json', 'r') as f:
                data = json.load(f)
        elif split == 'val':
            with open(base_dir + '/val.json', 'r') as f:
                data = json.load(f)
        elif split == 'test':
            with open(base_dir + '/test.json', 'r') as f:
                data = json.load(f)
        with open(base_dir + '/calib.pkl', 'rb') as f:
            calib = pkl.load(f)
        self.calib = calib
        self.data = data
        self.data_dir = base_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data_dir + '/rgb_images/' + self.data[idx]
        lbl_path = self.data_dir + '/gtLabels/' + self.data[idx]
        img = Image.open(img_path).convert('RGB')
        segm = Image.open(lbl_path).convert('L')
        key = self.data[idx][:-4] + '_img' + '.png'
        dist = torch.tensor(self.calib[key])
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

            # random_flip
        if np.random.choice([0, 1]):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

        # import pdb;pdb.set_trace()
        # note that each sample within a mini batch has different scale param
        img = imresize(img, (128, 128), interp='bilinear')
        segm = imresize(segm, (128,128), interp='nearest')

        # image transform, to torch float tensor 3xHxW
        img = img_transform(img)

        # segm transform, to torch long tensor HxW
        segm = segm_transform(segm)

        sample = {'image': img, 'label': segm, 'dist':dist}
        return sample
"""
