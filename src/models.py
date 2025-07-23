import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
 
import numpy as np

from typing import Optional, Union, Callable
from tqdm.auto import trange
import os
import math
import functools
import string

from ml_collections import ConfigDict

########################
def DefineMLP(layers):
    
    model = []
    for ins, outs in zip(layers[:-1],
                         layers[1:]):
        model.append(torch.nn.Linear(ins,outs, bias=True))
        model.append(torch.nn.ReLU())
    model.pop()
    return torch.nn.Sequential(*model)
########################


########################
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    
    half = dim // 2
    
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    
    args = timesteps[:, None].float() * freqs[None]
    
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
   
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
########################


########################
class ToyNet(torch.nn.Module):

    def __init__(self, model_x,  model, config):
        super().__init__()

        self.embed_dim = config.model.embed_dim
         
        self.model_x = model_x
        self.model = model
        self.act = torch.nn.ReLU()

    def forward(self, x):
        """
        x - torch.Size([B,3]) 
        """
     
        z = timestep_embedding(x[:,0], self.embed_dim) #[B,128]
        x = self.model_x(x[:,1:]) #[B,128]
        x = self.act(x)
        x = x + z #[B,128]
        x = self.model(x) #[B,3]
        return x 
########################





 


class ExponentialMovingAverage:

    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
        parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]


    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))


    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

            
def get_act(config=None, act_str=None):
    """Get activation functions from the config file."""
    assert (config or act_str) is not None
    if config is not None:
        act_str = config.model.nonlinearity.lower()
    if act_str == 'elu':
        return nn.ELU()
    elif act_str == 'relu':
        return nn.ReLU()
    elif act_str == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif act_str == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')
        
def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init

def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg' , 'uniform' )


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    nn.init.zeros_(conv.bias)
    return conv

def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)

class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)
    
    
class AttnBlock_(nn.Module):
    """Channel-wise self-attention block."""
    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        return x + h
    

class Downsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels, stride=2, padding=0)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, H, W = x.shape
        # Emulate 'SAME' padding
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
            x = self.Conv_0(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

        assert x.shape == (B, C, H // 2, W // 2)
        return x
    
class Upsample(nn.Module):
    def __init__(self, channels, with_conv=False):
        super().__init__()
        if with_conv:
            self.Conv_0 = ddpm_conv3x3(channels, channels)
        self.with_conv = with_conv

    def forward(self, x):
        B, C, H, W = x.shape
        h = F.interpolate(x, (H * 2, W * 2), mode='nearest')
        if self.with_conv:
            h = self.Conv_0(h)
        return h
    
class ResnetBlockDDPM(nn.Module):
    """The ResNet Blocks used in DDPM."""
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
        self.act = act
        self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)
        self.out_ch = out_ch
        self.in_ch = in_ch
        self.conv_shortcut = conv_shortcut

    """
    def forward(self, x, temb=None):
        return checkpoint(self._forward, (x,) ,self.parameters(), use_checkpoint)
    """
    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        assert C == self.in_ch
        out_ch = self.out_ch if self.out_ch else self.in_ch
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        # Add bias to each feature map conditioned on the time embedding
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if C != out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        return x + h
    
    
    
class DDPM(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)

        self.nf = nf = config.model.nf # dimensionality of time embedding
        self.conditional = conditional = config.model.conditional  # time conditional for diffusion  process

        modules = [] # This list is composed of nets
        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_init()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_init()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.data.centered
        channels = config.data.num_channels


        # downsampling block #
        modules.append(ddpm_conv3x3(channels, nf))
        ch_mult = config.model.ch_mult
        dropout = config.model.dropout
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.num_resolutions = num_resolutions = len(ch_mult) # downsamples
        # all_resolutions: [16,8,4,2]
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        resamp_with_conv = config.model.resamp_with_conv


        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
        
        AttnBlock = functools.partial(AttnBlock_)

        #####################
        # Downsampling block#
        #####################
        hs_c = [nf]
        in_ch = nf

        for i_level in range(num_resolutions):

            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        #####################


        #####################
        # Upsampling block#
        #####################
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))
        #####################


        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(ddpm_conv3x3(in_ch, channels+1, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)


    def forward(self, x, labels):

        """
        Running Diffusion model


        inputs:
        - x: [B,C,H,W] = [B,3,16,16] images
        - labels: [B]                time

        returns:
        """

        modules = self.all_modules
        m_idx = 0


        ##################
        #Time embeddings #
        ##################
        if self.conditional:
            # timestep/scale embedding
            timesteps = labels # torch.Size([B])
            temb = get_timestep_embedding(timesteps, self.nf) # torch.Size([B, nf])
            # modules[0] = torch.nn.Linear(nf, nf*4)
            temb = modules[m_idx](temb) # torch.Size([B, nf*4])
            m_idx += 1
            # modules[1] = torch.nn.Linear(nf*4,nf*4) = torch.nn.Linear(512,512)
            temb = modules[m_idx](self.act(temb)) # torch.Size([B, nf*4])
            m_idx += 1
        else:
            temb = None
        #################



        ######################
        #Centering of Images #
        ######################
        if self.centered:
            # Input is in [-1, 1]
            h = x

            #assert torch.min(x).item() >= -1.1
            #assert torch.max(x).item() <=  1.1
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.
            #assert torch.min(x).item() >= 0.1
            #assert torch.max(x).item() <= 1.1
        #################




        #####################
        # Downsampling block#
        #####################

        # torch.nn.Conv2D(3, nf, kernel_size=(3, 3), stride=1, padding=1)
        hs = [modules[m_idx](h)] # torch.Size([B,nf,16,16])
        m_idx += 1

        for i_level in range(self.num_resolutions): #self.num_resolutions

            # Residual blocks for this resolution

            #################################
            ##### while resolution = 16 #####
            ##################################


            # 1. ResNetBlock(in_ch=nf,out_ch = nf*1 = nf*ch_mult[0]) -> [torch.Size([B,nf,16,16])]
            # 2. AttnBlock -> [torch.Size([B,nf,16,16])]
            # 3. ResNetBlock(in_ch=nf,out_ch = nf*1 = nf*ch_mult[0]) -> [torch.Size([B,nf,16,16])]
            # 4. AttnBlock -> [torch.Size([B,nf,16,16])]
            # 5. ResNetBlock(in_ch=nf,out_ch = nf*1 = nf*ch_mult[0]) -> [torch.Size([B,nf,16,16])]
            # 6. AttnBlock -> [torch.Size([B,nf,16,16])]
            # 7. ResNetBlock(in_ch=nf,out_ch = nf*1 = nf*ch_mult[0]) -> [torch.Size([B,nf,16,16])]
            # 8. AttnBlock -> [torch.Size([B,nf,16,16])]
            # 9. DownSampling -> [torch.Size([B,nf,8,8])]
            # len(hs) = 6 : hs = [before, after_attn_1, ...., after_attn_4, after_downsampling]

            ##################################


            #################################
            ##### while resolution = 8 #####
            ##################################



            #################################


            #4 times ResNet(in_ch=nf,out_ch=nf), 4 times hs.append(torch.Size([B,nf,16,16]))
            # Downsample step: hs.append([torch.Size([B,nf,8,8])])

            # while resolution = 8: ResNet(in_ch=nf,out_ch=nf*2) and Attn(nf*2) , 4 times hs.append(torch.Size([B,nf*2,8,8]))
            # Downsample step: hs.append([torch.Size([B,nf*2,4,4])])

            # while resolution = 4: 4 times ResNet(in_ch=nf*2,out_ch=nf*2), 4 times hs.append(torch.Size([B,nf*2,4,4]))
            # Downsample step: hs.append([torch.Size([B,nf*2,2,2])])

            # while resolution = 2: 4 times ResNet(in_ch=nf*2,out_ch=nf*2), 4 times hs.append(torch.Size([B,nf*2,2,2]))
            # without last Downsample step

            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)

                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    # Application of Attention block
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                # Application of DownSample block
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1

        # hs[-1] = torch.Size([B,nf*2,2,2])
        # temb  = torch.Size([B,nf*4])


        h = hs[-1]
        h = modules[m_idx](h, temb) # torch.Size([B,nf*2,2,2])
        m_idx += 1
        h = modules[m_idx](h) # torch.Size([B,nf*2,2,2])
        m_idx += 1
        h = modules[m_idx](h, temb)# torch.Size([B,nf*2,2,2])
        m_idx += 1
        #####################




        #####################
        # Upsampling block#
        #####################
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)
        
        if self.config.training.sde == 'poisson':
            # Predict the direction on the extra z dimension
            scalar = F.adaptive_avg_pool2d(h[:, -1], (1, 1))
            return h[:, :-1], scalar.reshape(len(scalar))

        return h

        