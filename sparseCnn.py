import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.dataset_synapse import *
from utils_rad import get_sample_params_from_subdiv, get_sample_locations
import matplotlib.pyplot as plt 
import time 
from imageio.v2 import imread 
from skimage.transform import resize 

cuda_id= "cuda:0"

class SparseConv(nn.Module):
  def __init__(self, in_channels, out_channels,kernel):
    super().__init__()
    self.conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=1,padding= kernel//2, bias=False)
    #used apart, otherwise it will compute x*mask*w +b (or b is added after normalization)
    self.bias= nn.Parameter(data=torch.zeros(out_channels), requires_grad=True)
    #conv for the mask for the normalization (non learnable)
    self.sparsity= nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=1,padding= kernel//2, bias=False)
    #(cin=1,cout=1,k,k)
    w= torch.FloatTensor(torch.ones([kernel, kernel])).unsqueeze(0).unsqueeze(0)
    self.sparsity.weight= nn.Parameter(data=w, requires_grad=False)
    self.relu= nn.ReLU()
    self.max_pool= nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel//2)

  def forward(self,x, mask, eps=1e-8):
    x= x*mask
    x= self.conv(x)
    #normalization depending on the nb of valid pixels in the kernel
    x= x*(1/(self.sparsity(mask)+ eps))
    #check wo unsqueeze
    x= x+ self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    x= self.relu(x)
    mask= self.max_pool(mask)

    return x, mask

  

#check the output it should preserve existing pixels 
#check each mask 
class SparseConvNet(nn.Module):

    def __init__(self,out_channels=32, dataset="Matterport"):
        super().__init__()

        #self.SparseLayer1 = SparseConv(1, 16, 11)
        #self.SparseLayer2 = SparseConv(16, 16, 7)
        #self.SparseLayer3 = SparseConv(16, 16, 5)
        #self.SparseLayer4 = SparseConv(16, 16, 3)
        if dataset=='Matterport':
          self.SparseLayer1 = SparseConv(1, out_channels, 3)
        else:
          self.SparseLayer1 = SparseConv(1, out_channels, 5)
        self.SparseLayer5 = SparseConv(out_channels, out_channels, 3)
        self.SparseLayer6 = SparseConv(out_channels, 1, 1)

    def forward(self, x, mask):

        x, mask = self.SparseLayer1(x, mask)
        #x, mask = self.SparseLayer2(x, mask)
        #x, mask = self.SparseLayer3(x, mask)
        #x, mask = self.SparseLayer4(x, mask)
        x, mask = self.SparseLayer5(x, mask)
        x, mask = self.SparseLayer6(x, mask)

        return x
    def load_from(self, CKPT):
        pretrained_path = CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device(cuda_id if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            print(pretrained_dict.keys())
            self.load_state_dict(pretrained_dict)
        else:
            print("NO PRETRAINED ")
 

#relative absolute error (to get values btw 0 and 1)
class sparseLoss(nn.Module):
    def __init__(self):
      super().__init__()
    def forward(self,target,pred, mask):
      target= target*mask
      pred= pred*mask
      valid= target>0
      #print(torch.where(((target>0)==(mask==1))==False)) 0
      target= target[valid]
      pred = pred[valid]

      diff= target - pred 
      loss = torch.mean(torch.abs(diff) / target)

      #loss= torch.pow((target - pred),2)
      #loss = torch.sum(loss, dim=(2,3))
      #loss = loss/torch.sum(mask, dim=(2,3))
      #print("valid pixels ", torch.sum(mask, dim=(2,3))/(128*128))
      return loss
#grid: fractional grid, output (from darswin or sampled label here), indices representing the matrix of indices (from 0 to H-1)
def round_sample(grid, output, indices):
  H= indices.shape[0]
  K=H*H
  B, L, Np, Ns = output.shape
  N=Np*Ns #nb samples

  grid = grid + (H//2 -0.5) #go back to coord [-0.5, 127,5] #127,5= H -0.5
  grid= torch.floor(torch.abs(grid)) #go back to coord [0,127]
  grid= grid.reshape(B,-1,2)
  coord= grid.view(-1,2)
  cl= indices[coord[:,0].tolist(),coord[:,1].tolist()].view(B,-1)

  ind = torch.arange(N).reshape(1, -1)
  ind = torch.repeat_interleave(ind, B, 0)
  mat = torch.zeros(B, K, N).cuda(cuda_id) #the division matrix mapping samples to pixels
  mat[:, cl, ind] = 1
  output = output.reshape(B, L, -1).transpose(1, 2)
  pixel_out = torch.matmul(mat, output)
  div = mat.sum(-1).unsqueeze(2) #frequency of each sample to pixel (for averaging the values)
  div[div == 0] = 1
  pixel_out = torch.div(pixel_out, div)
  pixel_out = pixel_out.transpose(2, 1).reshape(B,L, H, H)

  return pixel_out 

#output a matrix (#pixels, W) where W is the max number of samples associated to a pixel (in the batch)
def func(arr,Npix, W=4):
  mat= -np.ones((Npix,W))
  Ns= arr.shape[0]//2
  pixels, indices = arr[:Ns], arr[Ns:]
  uniq, index = np.unique(pixels, return_index=True)
  #print("Pixels ",pixels)
  #print("Indices ", indices)
  start= index
  stop= np.append(index[1:], [Ns])
  indices= np.append(indices, [-1])
  #print(start)
  #print(stop)

  #if this can be optimized more, it will be super fast: used to generate arrays from start to stop and pad the arrays to have same dim
  #output_indices= [np.pad(range(a,b), (0,W-b+a), 'constant', constant_values=[-1]) for (a,b) in zip(start, stop)]
  output_indices= [np.hstack([range(a,b), np.array([0 for i in range(W-b+a)])]) for (a,b) in zip(start, stop)]
  #print(output_indices)
  #print(np.take(indices, output_indices))
  mat[uniq,:] = np.take(indices, output_indices)
  #print(mat)
  return mat


def round_sample_opt(grid, output, indices):
  H= indices.shape[0]
  K=H*H
  B, L, Np, Ns = output.shape
  N=Np*Ns #nb samples

  grid = grid + (H//2 -0.5) #go back to coord [-0.5, 127,5] #127,5= H -0.5
  grid= torch.floor(torch.abs(grid)) #go back to coord [0,127]
  grid= grid.reshape(B,-1,2)
  coord= grid.view(-1,2)
  cl= indices[coord[:,0].tolist(),coord[:,1].tolist()].view(B,-1)

  #print (cl)
  sorted_pixels, sorted_indices= torch.sort(cl, dim=1)
  _, sample_counts = torch.unique_consecutive(sorted_pixels, return_counts=True, dim=1)
  W = sample_counts.max().cpu()
  #print(W)
  concat= torch.hstack([sorted_pixels, sorted_indices]).cpu().numpy()
  mat=  np.apply_along_axis(func,1,concat, Npix=K, W=W )
  mat = torch.tensor(mat).cuda(cuda_id)
  output = output.reshape(B, L, -1)
  dummy_samples = torch.zeros([B,L,1]).cuda(cuda_id)
  output = torch.cat((output, dummy_samples), 2) 
  #print('mat',mat.shape)
  #print('out', output.shape)
  pixel_out = output[:,:,mat.long()]
  #print(pixel_out.min(), pixel_out.max())
  pixel_out = pixel_out[::B].squeeze(0)
  pixel_out= pixel_out.transpose(1,0)
  valid=torch.count_nonzero(pixel_out, dim=3)
  valid= torch.where(valid==0,1, valid)
  pixel_out = torch.sum(pixel_out, dim=3)/valid
  #print(pixel_out.min(), pixel_out.max())
  pixel_out = pixel_out.view(B, L, H, H)
  return pixel_out

"""
if __name__ == "__main__":
  ckpt= '/gel/usr/icshi/Radial-transformer-Unet/SparseM/epoch_9.pth'
  net = SparseConvNet().cuda(cuda_id)
  net.load_from(ckpt)
"""

if __name__ == "__main__":
    #root_path = '/gel/usr/icshi/DATA_FOLDER/Synwoodscape'
    root_path= '/gel/usr/icshi/Swin-Unet/data/M3D_low'
    batch_size= 8
    
    db_train = Synapse_dataset(base_dir=root_path, model="spherical", split="train")
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    radius_subdiv = 32
    azimuth_subdiv = 128
    dist_model= "spherical"
    subdiv = (radius_subdiv, azimuth_subdiv)
    n_radius = 2
    n_azimuth = 2
    img_size = (128,128)
    H= img_size[0]
    radius_buffer, azimuth_buffer = 0, 0

    for i_batch, sampled_batch in enumerate(trainloader):
        label_batch, dist =  sampled_batch['label'], sampled_batch['dist']
        label_batch, dist = label_batch.cuda(cuda_id), dist.cuda(cuda_id)
        
        #torch.save(label_batch, 'label.pt')
        #torch.save(dist, 'distL.pt')
        
        dist= dist.transpose(1,0)
        params, D_s = get_sample_params_from_subdiv(
                    subdiv= subdiv,
                    img_size= img_size,
                    distortion_model = dist_model,
                    D = dist,
                    n_radius= n_radius,
                    n_azimuth= n_azimuth,
                    radius_buffer=radius_buffer,
                    azimuth_buffer=azimuth_buffer)
        sample_locations = get_sample_locations(**params)
        B, n_p, n_s = sample_locations[0].shape
       
        x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
        y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()

        #used to get samples from labels (to use nn.functional.grid_sample grid should be btw -1,1)
        grid_out= torch.cat((y_/(H//2), x_/(H//2)), dim=3)

        sampled_label= nn.functional.grid_sample(label_batch, grid_out, align_corners = True, mode='nearest')

        grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2
        indices= torch.arange(H*H).reshape(H,H).cuda(cuda_id)
        s= time.time()
        pixel_out=  round_sample(grid,sampled_label, indices)

        print(time.time()-s)

        example= pixel_out[0,...].squeeze(0)
        print(example.shape)
        plt.imsave('optL.png', example.cpu().numpy())




        """
        K= H*H #nb pixels
        B, L, Np, Ns = sampled_label.shape
        N=Np*Ns #nb samples

        grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2

        indices= torch.arange(H*H).reshape(H,H).cuda(cuda_id)

        pixel_out= round_sample(grid, sampled_label, indices)
        """


        #print(time.time()-s)


        """"
        ###################################################################################
        grid = grid + (H//2 -0.5) #go back to coord [-0.5, 127,5] #127,5= H -0.5
        grid= torch.floor(torch.abs(grid)) #go back to coord [0,127]
        grid= grid.reshape(B,-1,2)

        ind= torch.arange(K).reshape(H,H).cuda(cuda_id) #grid representing the index of each coord (values from 0 to 127)
        coord= grid.view(-1,2)
        cl= ind[coord[:,0].tolist(),coord[:,1].tolist()].view(B,-1) #similar to cl in knn v1 (for each sample the associated pixel)

        ind = torch.arange(N).reshape(1, -1)
        ind = torch.repeat_interleave(ind, B, 0)
        mat = torch.zeros(B, K, N).cuda(cuda_id) #the division matrix mapping samples to pixels
        mat[:, cl, ind] = 1
        sampled_label = sampled_label.reshape(B, L, -1).transpose(1, 2)
        pixel_out = torch.matmul(mat, sampled_label)
        div = mat.sum(-1).unsqueeze(2) #frequency of each sample to pixel (for averaging the values)
        div[div == 0] = 1
        pixel_out = torch.div(pixel_out, div)
        pixel_out = pixel_out.transpose(2, 1).reshape(B,L, H, H)
        torch.save(pixel_out, 'sparse_img.pt')

        print(time.time()-s)
        ####################################################################################
      




        #torch.save(pixel_out, 'sparse_label.pt')
        #print(pixel_out.shape)


        #example to be ploted
        #for depth estimation the output is [B,1,128,128] change with .permute(1,2,0)
        #example= pixel_out[0,...].squeeze(0)
        example= pixel_out[0,...].permute(1,2,0)
        plt.imsave('saprse.png', example.cpu().numpy())
        print("done")
        '''


        #break
"""





