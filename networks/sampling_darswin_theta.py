
import torch
import numpy as  np 

# cuda_id= torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
cuda_id = "cuda:0"

def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out
    
def get_inverse_distortion(num_points,D, fov):
    theta_d_max = fov/2
    m = 2.288135593220339
    n = 5.0
    a = theta_d_max
    b = 8.31578947368421
    c =  0.3333333333333333

  
    def f(x, n, a, b):
        return b*torch.pow(x/a, n)
    def h(x, m, a):
        return -torch.pow(-x/a + 1, m) + 1
    def g(x, m, n, a, b, c):
        return c*f(x, n, a, b) + (1-c)*h(x, m, a)
    def g_inv(y, m, n, a, b, c):
        test_x = torch.linspace(0, theta_d_max, 20000).cuda(cuda_id)
        test_y = g(test_x, m, n, a, b, c).cuda(cuda_id)
        x = torch.zeros(num_points).cuda(cuda_id)
        for i in range(num_points):
            lower_idx = test_y[test_y <= y[i]].argmax()
            x[i] = test_x[lower_idx]
        return x

    def rad(D, theta, theta_max):
        focal_func = lambda x: 1/(x.reshape(1, x.shape[0]).repeat_interleave(D.shape[1], 0).flatten() * (torch.outer(D[0], x**0).flatten() + torch.outer(D[1], x**1).flatten() + torch.outer(D[2], x**2).flatten() +torch.outer(D[3], x**3).flatten()))
        focal = focal_func(theta_max.reshape(1)).reshape(1, D.shape[1])
        funct = g_inv(theta, m = m, n = n, a = a, b = b, c = c).reshape(1,-1).repeat_interleave(D.shape[1], 0).transpose(1,0).cuda(cuda_id)
        radius= focal* funct * (D[0] * funct**0 + D[1] * funct**1 + D[2] * funct**2 + D[3] * funct**3)
        return radius
    
    theta_d = torch.linspace(0, g(theta_d_max, m = m, n = n, a = a, b = b, c = c), num_points + 1).cuda(cuda_id)
    r_list = rad(D, theta_d, theta_d_max).cuda(cuda_id)
    return r_list, theta_d_max




def get_inverse_dist_spherical_theta(num_points, xi, fov, new_f):

    p = 1
    rad = lambda x: ((xi + torch.cos(torch.tensor(fov/2)))/torch.sin(torch.tensor(fov/2)))*torch.sin(x)/(xi + torch.cos(x)) 

    theta_d_max = torch.tensor(fov/2)
    # theta_d_max = torch.tensor(fov/2)
    theta_d = linspace(torch.tensor([0]).cuda(), theta_d_max, num_points)
    delta = float(torch.diff(theta_d, axis=0)[0][0])
    a = np.random.uniform(0, 1)
    err = np.random.uniform(0, delta/2)
    a = 0.5
    if  a > 0.5:
        err = np.random.uniform(0, delta/2)
        theta_d = theta_d + err

    elif a < 0.5:
    # import pdb;pdb.set_trace()
        theta_d = theta_d - err
        theta_d[0] = 0.0
  
    r_list = rad(theta_d)
    return r_list , theta_d_max

def get_sample_params_from_subdiv(subdiv, distortion_model, img_size, D):
    """Generate the required parameters to sample every patch based on the subdivison
    Args:
        subdiv (tuple[int, int]): the number of subdivisions for which we need to create the 
                                  samples. The format is (radius_subdiv, azimuth_subdiv)
        n_radius (int): number of radius samples
        n_azimuth (int): number of azimuth samples
        img_size (tuple): the size of the image
    Returns:
        list[dict]: the list of parameters to sample every patch
    """
    max_radius = min(img_size)/2
    width = img_size[1]
    # D_min = get_inverse_distortion(subdiv[0], D, max_radius)
    if distortion_model == 'spherical': # in case of spherical distortion pass the 
        fov = D[2][0]
        f  = D[1]
        xi = D[0]
        D_min, theta_max = get_inverse_dist_spherical_theta(subdiv[0], xi, fov, f)
        D_min = D_min*max_radius
        # breakpoint()
    elif distortion_model == 'polynomial' or distortion_model == 'polynomial_woodsc':
        # 
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, 1.0)
        D_min = D_min*max_radius
    # breakpoint()
    alpha = 2*torch.tensor(np.pi).cuda() / subdiv[1]
    D_min = D_min.reshape(subdiv[0], 1, D.shape[1]).repeat_interleave(subdiv[1], 1).cuda(cuda_id)
    phi_start = 0
    phi_end = 2*torch.tensor(np.pi)
    phi_step = alpha
    phi_list = torch.arange(phi_start, phi_end, phi_step)
    phi_list = phi_list.reshape(1, subdiv[1]).repeat_interleave(subdiv[0], 0).reshape(subdiv[0], subdiv[1], 1).repeat_interleave(D.shape[1], 2).cuda(cuda_id)
   
    phi_list_cos  = torch.cos(phi_list) 
    phi_list_sine = torch.sin(phi_list) 
    x = D_min * phi_list_cos    # takes time the cosine and multiplication function 
    y = D_min * phi_list_sine
    return x.transpose(1, 2).transpose(0,1), y.transpose(1, 2).transpose(0,1), theta_max