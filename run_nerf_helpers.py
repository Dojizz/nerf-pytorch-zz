import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
    
    # create encoding function according to the input kwargs
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                """ why there is no pi???"""
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# multires is l in paper, so the highest frequency is 2^(l-1), that is log2
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # no positional encoding??
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        # use view infomation
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # output alpha
            self.alpha_linear = nn.Linear(W, 1)
            # output rgb
            self.rgb_linear = nn.Linear(W//2, 3)
        # otherwise directly output
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        # x.shape=[N, 6] -> input_pts.shape=[N, 3], input_views.shape=[N, 3]
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            # i: index, l: module itself
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        # here, why not activate outputs??
        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

# SiNeRF
class Sine(nn.Module):
    def __init__(self, alpha=1., bias=0.):
        super(Sine, self).__init__()
        self.alpha = alpha
        self.bias = bias

    def forward(self, x):
        return torch.sin(self.alpha * x + self.bias)

class SirenLinear(nn.Module):
    def __init__(self, input_dim=256, output_dim=256, use_bias=True, w=1., is_first=False):
        super(SirenLinear, self).__init__()
        self.fc_layer = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.use_bias = use_bias
        self.activation = Sine(w)
        # is_first indicates whether this layer is the first layer
        self.is_first = is_first
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = w
        # and what is this?
        self.c = 6.0
        self.init_params()

    def init_params(self):
        with torch.no_grad():
            dim = self.input_dim
            if self.is_first:
                w_std = (1/dim)
            else:
                w_std = math.sqrt(self.c / dim)
            # tensor.uniform_(from, to)
            self.fc_layer.weight.uniform_(-w_std, w_std)
            if self.use_bias and self.fc_layer.bias is not None:
                self.fc_layer.bias.unifome(-w_std, w_std)
    
    def forward(self, x):
        out = self.fc_layer(x)
        out = self.activation(out)

# unlike the original version, just brutely substitute relu with sine
class SiNeRF(nn.Module):
    # here by default, bias = 0, no skip, structure is exactly the described sinerf 
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4]):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        # like the paper, final result just multiply the mag, no activation
        self.rgb_mag=1.0
        self.den_mag=25.0

        self.sine_w_layers = iter([30, 1, 1, 1, 1, 1, 1, 1])
        
        # positional encoding is considered before create nerf
        self.pts_sine_linears = nn.ModuleList(
            [SirenLinear(input_ch, W, True, next(self.sine_w_layers), is_first=True)])
        for i in range(D-1):
            # no skip
            self.pts_sine_linears.append(SirenLinear(W, W, True, next(self.sine_w_layers)))
            
        
        self.linear_layer = nn.Linear(W, W)
        self.sigma_layer1 = SirenLinear(W, W//2, True, 1)
        self.sigma_layer2 = nn.Linear(W//2, 1)
        self.rgb_layer1 = SirenLinear(W + input_ch_views, W//2, True, 1)
        self.rgb_layer2 = nn.Linear(W//2, 3)
        

    def forward(self, x):
        # x.shape=[N, 6] -> input_pts.shape=[N, 3], input_views.shape=[N, 3]
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_sine_linears):
            # i: index, l: module itself
            # no skip
            h = self.pts_sine_linears[i](h)
        sigma = self.sigma_layer1(h)
        sigma = self.sigma_layer2(sigma)
        rgb = self.linear_layer(h)
        rgb = self.rgb_layer1(torch.cat([rgb, input_views], -1))
        rgb = self.rgb_layer2(rgb)

        ### magnifying color and density. No output regulating here.
        rgb = self.rgb_mag * rgb
        sigma = self.den_mag * sigma

        return torch.cat([rgb, sigma], -1)
        

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

# generate rays_o, rays_d for ONE camera
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

# get coords for MRS, image: [H, W, 3]
def get_interest_coords(image):
    img = np.copy(image.numpy())
    img *= 255.
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    H = img_gray.shape[0]
    W = img_gray.shape[1]
    # Find key points
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    coords_list = [keypoint.pt for keypoint in keypoints]
    coords_list = np.array(coords_list).astype(int)
    # Remove duplicate points, xy: [N, 2]
    xy_set = set(tuple(point) for point in coords_list)
    coords_list = np.array([list(point) for point in xy_set]).astype(int)
    # Include neighbour points
    interest_regions = np.zeros((H, W), dtype=np.uint8)
    # Consider opencv coords are different
    interest_regions[coords_list[:,1], coords_list[:,0]] = 1
    interest_regions = cv2.dilate(interest_regions, np.ones((5, 5), np.uint8), iterations=1)
    interest_regions = np.array(interest_regions, dtype=bool)
    coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H)), -1), dtype=int)
    interest_indices = coords[interest_regions]
    
    return interest_indices # (N, 2) np.ndarray