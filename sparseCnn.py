import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.dataset_synapse import *
from utils_rad import get_sample_params_from_subdiv, get_sample_locations
import matplotlib.pyplot as plt 

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

if __name__ == "__main__":
    cuda_id= "cuda:1"
    root_path = '/gel/usr/icshi/DATA_FOLDER/Synwoodscape'
    batch_size= 1

    db_train = Synapse_dataset(base_dir=root_path, split="train")
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    radius_subdiv = 32
    azimuth_subdiv = 128
    dist_model= "polynomial"
    subdiv = (radius_subdiv, azimuth_subdiv)
    n_radius = 5
    n_azimuth = 5
    img_size = (128,128)
    H= img_size[0]
    radius_buffer, azimuth_buffer = 0, 0

    for i_batch, sampled_batch in enumerate(trainloader):
        label_batch, dist =  sampled_batch['label'], sampled_batch['dist']
        label_batch, dist = label_batch.cuda(cuda_id), dist.cuda(cuda_id)
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

        K= H*H #nb pixels
        B, L, Np, Ns = sampled_label.shape
        N=Np*Ns #nb samples

        grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2
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
        print(pixel_out.shape)

        #example to be ploted
        #for depth estimation the output is [B,1,128,128] change with .permute(1,2,0)
        example= pixel_out[0,...].squeeze(0)
        plt.imsave('saprse.png', example.cpu().numpy())
        print("done")
        break







