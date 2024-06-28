from envmap import EnvironmentMap
from envmap import rotation_matrix
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
import pickle as pkl 



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
    image = ndimage.rotate(image, angle, order=1, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

#'png'
def load_color(filename: str) -> torch.Tensor:    
    return {'color': torchvision.io.read_image(filename) / 255.0 }    

#'depth' and 'exr'
def load_depth(filename: str, max_depth: float=8.0) -> torch.Tensor:
    depth_filename = filename.replace('.png', '.exr')
    depth = torch.from_numpy(
        cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
    ).unsqueeze(0)
    #NOTE: add a micro meter to allow for thresholding to extact the valid mask
    depth[depth > max_depth] = max_depth + 1e-6 #replace inf value by max_depth #without any impact on wood 
    return {
        'depth': depth
    }

def sph2cart(az, el, r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

#rad
def compute_focal(fov, xi, width):
    return width / 2 * (xi + np.cos(fov/2)) / np.sin(fov/2)

#order 0:nearest 1:bilinear
def warpToFisheye(pano,outputdims,viewingAnglesPYR=[np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)],xi=0.9, fov=150, order=1):

    outputdims1=outputdims[0]
    outputdims2=outputdims[1]
   
    pitch, yaw, roll = np.array(viewingAnglesPYR)
    #print(pano.shape)

    e = EnvironmentMap(pano, format_='latlong')
    e = e.rotate(rotation_matrix(yaw, -pitch, -roll).T)

    r_max = max(outputdims1/2,outputdims2/2)

    h= min(outputdims1,outputdims2)
    f = compute_focal(np.deg2rad(fov),xi,h)
    
    t = np.linspace(0,fov/2, 100)
   

    #test spherical
    print('xi  {}, f {}'.format(xi,f))
    theta= np.deg2rad(t)
    funT = (f* np.sin(theta))/(np.cos(theta)+xi)
    funT= funT/r_max


    #creates the empty image
    [u, v] = np.meshgrid(np.linspace(-1, 1, outputdims1), np.linspace(-1, 1, outputdims2))
    r = np.sqrt(u ** 2 + v ** 2)
    phi = np.arctan2(v, u)
    validOut = r <= 1
    # interpolate the _inverse_ function!
    fovWorld = np.deg2rad(np.interp(x=r, xp=funT, fp=t))
    # fovWorld = np.pi / 2 - np.arccos(r)
    FOV = np.rad2deg((fovWorld))

    el = fovWorld + np.pi / 2

    # convert to XYZ
    #ref
    x, y, z = sph2cart(phi, fovWorld + np.pi / 2, 1)

    x = -x
    z = -z

    #return values in [0,1]
    #the source pixel from panorama 
    [u1, v1] = e.world2image(x, y, z)

    # Interpolate
    #validout to set the background to black (the circle part)
    eOut= e.interpolate(u1, v1, validOut, order)
    #eOut= e.interpolate(u1, v1)
    return eOut.data, f
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        #wether add the mask in the dataloader or no aug
        if random.random() > 0.5: #corrupt the mask (rotation 90)
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5: #corrupt the mask (random rotation)
            image, label = random_rotate(image, label)
        
        """
        x, y, c  = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        """
        
        image = torch.from_numpy(image.astype(np.float32)) #.unsqueeze(0)
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


#woodscape
#mean= [0.1867, 0.1694, 0.1573]
#std= [0.2172, 0.2022, 0.1884]

#Matterport
mean= [0.2217, 0.1939, 0.1688]
std= [0.1884, 0.1744, 0.1835]

#normalize = None
normalize= transforms.Normalize(
            mean= mean ,
            std= std )
#normalize = None
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, model= "spherical", transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.model= model
        self.img_size= 64
        self.data_dir = base_dir
        self.calib = None
        
        if split == 'train':
            with open(base_dir + '/train_gp4.json', 'r') as f:
                data = json.load(f)
        elif split == 'val':
            with open(base_dir + '/val_gp4.json', 'r') as f:
                data = json.load(f)
        elif split == 'test':
            with open(base_dir + '/val_gp4.json', 'r') as f:
                data = json.load(f)

            with open(self.data_dir + '/test_calib.pkl', 'rb') as f:
                self.calib = pkl.load(f)

        self.data = data #['1LXtFkjw3qL/85_spherical_1_emission_center_0.png'] #data[:5]

        if self.calib is None and os.path.exists(self.data_dir+ '/calib_gp4.pkl') :
            with open(self.data_dir + '/calib_gp4.pkl', 'rb') as f:
                self.calib = pkl.load(f)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        b_path= self.data[idx]
        if self.model =="polynomial":
            img_path = self.data_dir + '/rgb_images/' + b_path
            depth_path = self.data_dir + '/depth_maps/' + b_path.replace('png','exr')
            max_depth=1000.0
            dist= Distortion[(b_path.split('.')[0]).split('_')[1]]
        elif self.model== "spherical":
            img_path = self.data_dir + '/' +b_path
            depth_path = img_path.replace('emission','depth').replace('png','exr')
            max_depth=8.0
            
        image= load_color(img_path)['color']
        depth= load_depth(depth_path, max_depth)['depth']

        mat_path= img_path.replace('png','npy')
        if self.split=='test':
           mat_path =  mat_path[:-4]+'_test.npy'
        cl= np.load(mat_path)

        image=image.permute(1,2,0)
        depth=depth.permute(1,2,0)

        if self.model == "spherical":
            h= image.shape[0]
            fov=175
            xi= (self.calib[b_path])[0]

            if self.split == 'train':
                ang = random.uniform(0,360)
            else:
                ang= 0
            
            angles = [np.deg2rad(0), np.deg2rad(ang), np.deg2rad(0)]
            image, f = warpToFisheye(image.numpy(), outputdims=(h,h), viewingAnglesPYR= angles, xi=xi, fov=fov, order=1)
            depth,_= warpToFisheye(depth.numpy(), outputdims=(h,h), viewingAnglesPYR= angles, xi=xi, fov=fov, order=0)
            dist= np.array([xi, f/(h/self.img_size), np.deg2rad(fov)])

       

        image = cv2.resize(image, (self.img_size,self.img_size),interpolation = cv2.INTER_LINEAR)
        label= cv2.resize(depth, (self.img_size,self.img_size), interpolation = cv2.INTER_NEAREST)
        
        sample = {'image': image, 'label': label, 'path':b_path.replace('png','npy')}
        #sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['image']= torch.from_numpy(image.astype(np.float32)) 
            sample['label']= torch.from_numpy(label.astype(np.float32))
        

        sample['image']= sample['image'].permute(2,0,1)
        sample['label']= sample['label'].unsqueeze(0)

        sample['dist'] = dist
        sample['cl'] = cl 

        mask = ((sample['label']> 0) & (sample['label'] <= max_depth)
                                & ~torch.isnan(sample['label']))
        sample['mask'] = mask 

        if normalize is not None:
            sample['image']= normalize(sample['image'])

        #print(sample.keys())
        return sample

def get_mean_std(base_dir ):
    db= Synapse_dataset(base_dir, split="train", transform=None)
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


