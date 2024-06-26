import torch 
from pykeops.torch import LazyTensor
from networks.Swin_transformer_az import *
import matplotlib.pyplot as plt 


def KMeans(x, c, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    B, Np, Ns, D = x.shape  # Number of samples, dimension of the ambient space
    x = x.view(B, -1, D)

    x_i = LazyTensor(x.view(B, Np*Ns, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(B, 1, K, D))  # (1, K, D) centroids

    D_ij = ((x_i - c_j) ** 2).sum(B, -1)  # (N, K) symbolic squared distances
    cl = D_ij.argmin(dim=2).long().view(B, -1)  # Points -> Nearest cluster

    return cl, c

def resample(grid, grid_pix, H, B, output, embed_dim):
    N, D, K = grid.shape[1]*grid.shape[2], 2, grid_pix.shape[1]  #grid of samples and grid_pix of pixels
    B, L, Np, Ns = output.shape
    cl, c = KMeans(grid, grid_pix/(H//2), K) #cl : value of cluster centers (pixels)
    ind = torch.arange(N).reshape(1, -1) #ind represents index of each sample ranging from 0 to num_pixels
    ind = torch.repeat_interleave(ind, B, 0)
    mat = torch.zeros(B, K, N).cuda("cuda:0") #the division matrix mapping samples to pixels
    mat[:, cl, ind] = 1
    output = output.reshape(B, L, -1).transpose(1, 2)
    pixel_out = torch.matmul(mat, output)
    div = mat.sum(-1).unsqueeze(2) #frequency of each sample to pixel (for averaging the values)
    div[div == 0] = 1
    pixel_out = torch.div(pixel_out, div)
    pixel_out = pixel_out.transpose(2, 1).reshape(B, embed_dim, H, H)
    return pixel_out

def sample2pixel(dist, output, dist_model="polynomial"):
    #dist= torch.tensor(np.array(dist).reshape(4,1)).cuda()
    dist= dist.reshape(-1,1)
    radius_buffer, azimuth_buffer = 0, 0
    params, D_s = get_sample_params_from_subdiv(
            #subdiv(self.radius_cuts, self.azimuth_cuts)
            subdiv=(32,128),
            img_size=(128,128),
            distortion_model = dist_model,
            D = dist, 
            n_radius=5,
            n_azimuth=5,
            radius_buffer=radius_buffer, 
            azimuth_buffer=azimuth_buffer)

    sample_locations = get_sample_locations(**params)
    B, n_p, n_s = sample_locations[0].shape
    x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
    x_ = x_/ 64
    y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()
    y_ = y_/64    
    grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2

    ################################################
    x_p = torch.linspace(0, 128, 129) - 64.5
    y_p = torch.linspace(0, 128, 129) - 64.5
    grid_x, grid_y = torch.meshgrid(x_p[1:], y_p[1:], indexing='ij')
    x_ = grid_x.reshape(128*128, 1)
    y_ = grid_y.reshape(128*128, 1)
    grid_pix = torch.cat((x_, y_), dim=1).cuda()
    #print(grid_pix.shape)
    grid_pix = grid_pix.reshape(1, 128*128, 2)
    grid_pix = torch.repeat_interleave(grid_pix, B, 0)
    #####################################################
    
    H=128
    embed_dim= 1
    output= output.unsqueeze(0)
    x = resample(grid, grid_pix, H, B, output, embed_dim)
    x= x.squeeze(0).permute(1,2,0).squeeze(2)
    try:
        x= x.cpu().numpy()
    except:
        x= x.cpu().detach().numpy()
    return x 



if __name__ == "__main__":
    
    dist= torch.tensor(np.array([342.234, -18.6659, 23.1572, 4.28064]).reshape(4,1)).cuda()
    radius_buffer, azimuth_buffer = 0, 0
    params, D_s = get_sample_params_from_subdiv(
            #subdiv(self.radius_cuts, self.azimuth_cuts)
            subdiv=(32,128),
            img_size=(128,128),
            distortion_model = "polynomial",
            D = dist, 
            n_radius=5,
            n_azimuth=5,
            radius_buffer=radius_buffer, 
            azimuth_buffer=azimuth_buffer)

    sample_locations = get_sample_locations(**params)
    B, n_p, n_s = sample_locations[0].shape
    x_ = sample_locations[0].reshape(B, n_p, n_s, 1).float()
    x_ = x_/ 64
    y_ = sample_locations[1].reshape(B, n_p, n_s, 1).float()
    y_ = y_/64    
    grid = torch.cat((x_, y_), dim=3) #B, n_p,n_s,2

    ################################################
    x_p = torch.linspace(0, 128, 129) - 64.5
    y_p = torch.linspace(0, 128, 129) - 64.5
    grid_x, grid_y = torch.meshgrid(x_p[1:], y_p[1:], indexing='ij')
    x_ = grid_x.reshape(128*128, 1)
    y_ = grid_y.reshape(128*128, 1)
    grid_pix = torch.cat((x_, y_), dim=1).cuda()
    #print(grid_pix.shape)
    grid_pix = grid_pix.reshape(1, 128*128, 2)
    grid_pix = torch.repeat_interleave(grid_pix, B, 0)
    #####################################################
    x= torch.load('vis.pt')[0,...].unsqueeze(0)
    H=128
    embed_dim= 1
    x = resample(grid, grid_pix, H, B, x, embed_dim)
    x= x.squeeze(0).permute(1,2,0).squeeze(2)
    x= x.cpu().numpy()
    plt.imsave('test_resample.png', x)
    #print(x.shape)


    
    

