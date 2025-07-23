import torch
import numpy as np 
import os
import pickle
from typing import Dict, Any
import wandb

 

    
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    
def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

def random_color(im):
    hue = 360*np.random.rand()
    d = (im *(hue%60)/60)
    im_min, im_inc, im_dec = torch.zeros_like(im), d, im - d
    c_im = torch.zeros((3, im.shape[1], im.shape[2]))
    H = round(hue/60) % 6    
    cmap = [[0, 3, 2], [2, 0, 3], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]]
    return torch.cat((im, im_min, im_dec, im_inc), dim=0)[cmap[H]]
    
###########################
def get_mesh(config):
    mesh = []
    if len(config.DICT_MESH) == 2:
        y = torch.linspace(config.DICT_MESH["y"][0],config.DICT_MESH["y"][1], config.NUM_MESH)
        x = torch.linspace(config.DICT_MESH["x"][0], config.DICT_MESH["x"][1], config.NUM_MESH)
        for i in x:
            for j in y:
                mesh.append(torch.tensor([i, j]))
        
        
    elif len(config.DICT_MESH) == 3:
        z = torch.linspace(config.DICT_MESH["z"][0],config.DICT_MESH["z"][1], config.NUM_MESH)
        x = torch.linspace(config.DICT_MESH["x"][0], config.DICT_MESH["x"][1], config.NUM_MESH)
        y = torch.linspace(config.DICT_MESH["y"][0], config.DICT_MESH["y"][1], config.NUM_MESH)
        
        for i in x:
            for j in y:
                for k in z:
                    mesh.append(torch.tensor([i,j,k]))
    
        

    else:
        raise ValueError("The length of DICT MESH should be 2 or 3!")
    return torch.stack(mesh, dim=0)
##########################





##########################
def forward_interpolation(x, y, config):

    if config.training.interpolation == 'uniform':
        
        high = config.q.x_loc - config.training.epsilon
        den  = config.L                  
        
        t = torch.distributions.Uniform(low=config.p.x_loc + config.training.epsilon,
            high=high).sample(torch.Size([config.training.small_batch_size  ])).to(config.device)[:, None]#[b,D+1]
        
        perturbed_samples_vec = (t-config.p.x_loc)/den*y[:config.training.small_batch_size ]\
                  + (1 - (t - config.p.x_loc)/den)*x[:config.training.small_batch_size ] #[b, D+1]

    elif config.training.interpolation == 'both_side':
        
        m = torch.rand(config.training.small_batch_size//2, device=config.device)  * config.training.M
        multiplier = (1+config.training.tau) ** m # 1.04-1.12
        left_z = torch.randn((config.training.small_batch_size//2, 1)).to(config.device) * config.training.sigma_end
        left_z = left_z.abs() 
        perturbed_left_z = config.training.epsilon + left_z.squeeze() * multiplier

        m = torch.rand(config.training.small_batch_size//2, device=config.device) * config.training.M
        multiplier = (1+config.training.tau) ** m
        right_z = torch.randn((config.training.small_batch_size//2, 1)).to(config.device) * config.training.sigma_end
        right_z = right_z.abs() 
        perturbed_right_z = config.L - config.training.epsilon - right_z.squeeze() * multiplier

        perturbed_z = torch.cat([perturbed_left_z, perturbed_right_z],dim=0)
       
         
        mask_right = torch.nonzero(perturbed_z > config.q.x_loc - config.training.epsilon)
        perturbed_z[mask_right.view(-1)] = torch.distributions.Uniform(low=config.p.x_loc + config.training.epsilon,
                high=config.q.x_loc - config.training.epsilon).sample(torch.Size([len(mask_right)])).to(config.device)

        mask_left = torch.nonzero(perturbed_z < config.p.x_loc + config.training.epsilon)
        perturbed_z[mask_left.view(-1)] = torch.distributions.Uniform(low=config.p.x_loc + config.training.epsilon,
                high=config.q.x_loc - config.training.epsilon).sample(torch.Size([len(mask_left)])).to(config.device)

         
        perturbed_x = y[:config.training.small_batch_size,1:]*((perturbed_z[:,None]-config.training.epsilon)/config.L)\
                      + (1 - (perturbed_z[:,None]-config.training.epsilon)/config.L)*x[:config.training.small_batch_size,1:]

        perturbed_samples_vec = torch.cat([perturbed_z[:, None],
                                           perturbed_x.reshape(config.training.small_batch_size, 
                                           config.DIM-1)], dim=1)

    elif config.traaining.interpolation == 'pfgm':
        
        
        
        m = torch.rand((x.shape[0],), device=x.device) * config.training.M
        data_dim = config.data.channels * config.data.image_size * config.data.image_size # N
        tau = config.training.tau
        z = torch.randn((len(x), 1, 1, 1)).to(x.device) * config.model.sigma_end  # [B,1,1,1]
        z = z.abs() # [B,1,1,1]


        # Confine the norms of perturbed data.
        # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
        if config.training.restrict_M:
            idx = (z < 0.005).squeeze()
            num = int(idx.int().sum())
            restrict_m = int(sde.M * 0.7)
            m[idx] = torch.rand((num,), device=samples_batch_x.device) * restrict_m


        multiplier = (1+tau) ** m # torch.Size([B])
        perturbed_z = z.squeeze() * multiplier # torch.Size([B])* torch.Size([B]) = torch.Size([B])


        ####### perturbation for x component #######

        # Sample uniform angle
        gaussian = torch.randn(len(x), data_dim).to(x.device) # torch.Size([B, C*H*W])
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True) #  torch.Size([B, C*H*W])

        # injected noise amount
        noise = torch.randn_like(x).reshape(len(x), -1) * config.model.sigma_end #torch.Size([B, C*H*W])
        norm_m = torch.norm(noise, p=2, dim=1) * multiplier # torch.Size([B])*torch.Size([B]) = torch.Size([B])


        # Construct the perturbation for x
        perturbation_x = unit_gaussian * norm_m[:, None] # torch.Size([B,C*H*W])* torch.Size([B,1])=  torch.Size([B,C*H*W])
        perturbation_x = perturbation_x.view_as(x) # torch.size([B,C,H,W])

        # Perturb x
        perturbed_x = samples_batch_x + perturbation_x # torch.size([B,C,H,W])

        # Augment the data with extra dimension z
        perturbed_samples_vec = torch.cat((perturbed_x.reshape(len(x), -1),
                                           perturbed_z[:, None]), dim=1)

        # concatenate: torch.Size([B,C*H*W], torch.Size([[B,1]]) = torch.Size([B,C*H*W + 1]
         

        
        
        
        
        
        
    elif config.training.interpolation == 'gaussian_tube':


        noise = torch.randn(config.model.training_batch_small//2).to(config.device)[:,None] * config.model.sigma_end #[B/2,DIM]
        z_left = config.p.x_loc + config.model.epsilon + noise.abs()
        noise = torch.randn(config.model.training_batch_small//2).to(config.device)[:,None] * config.model.sigma_end #[B/2,DIM]
        z_right = config.L - config.model.epsilon  - noise.abs()
        perturbed_z = torch.cat([z_left,z_right],dim=0)
        perturbed_z = torch.clamp(perturbed_z, min=config.p.x_loc + config.model.epsilon,
                                     max=config.L - config.model.epsilon)
        
        perturbed_x = ((perturbed_z - config.p.x_loc)/config.L)*y[:,1:] +\
                      (1-((perturbed_z - config.p.x_loc)/config.L))*x[:,1:] #[B,DIM]

        gaussian = torch.randn(config.model.training_batch_small ,
                               config.DIM-1).to(config.device)  
        unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)


        noise = torch.randn_like(gaussian).reshape(config.model.training_batch_small ,
                                -1) * config.model.sigma_end
        norm_m = torch.norm(noise, p=2, dim=1, keepdim=True)* config.model.norm_scaler*(- torch.abs(perturbed_z - config.L/2) + config.L/2)
     
        perturbed_x = perturbed_x + unit_gaussian * norm_m
        perturbed_samples_vec = torch.cat((perturbed_z , perturbed_x), dim=1)
    
    elif config.model.interpolation == 'efm':
        
        mesh = get_mesh(config).to(config.device)
        idxs = torch.randperm(mesh.shape[0])
        perturbed_samples_vec = mesh[idxs][:config.model.training_batch_small]
    else:
        raise ValueError(f"There is no such interpolation as {config.model.interpolation}")
    
    return perturbed_samples_vec 
##########################



##########################
def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, net, step, config, lr=config.optim.lr,
                      warmup=config.optim.warmup,
                      grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        
        
        
        
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
         
        
        total_norm = 0
        for p in  net.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        wandb.log({"Gradient's norm before clip" : total_norm },step=step)
         
       
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        
      
        total_norm = 0
        for p in net.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        wandb.log({"Gradient's norm after clip" : total_norm },step=step)
        
        
        optimizer.step()
        wandb.log({"Learning rate" : optimizer.param_groups[0]['lr']},step=step)
        #scheduler.step()

    return optimize_fn
##########################


##########################
class Config():

    @staticmethod
    def fromdict(config_dict):
        config = Config()
        for name, val in config_dict.items():
            setattr(config, name, val)
        return config
    
    @staticmethod
    def load(path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'rb') as handle:
            config_dict = pickle.load(handle)
        return Config.fromdict(config_dict)

    def store(self, path):
        os.makedirs(os.path.join(*("#" + path).split('/')[:-1])[1:], exist_ok=True)
        with open(path, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def set_attributes(
            self, 
            attributes_dict: Dict[str, Any], 
            require_presence : bool = True,
            keys_upper: bool = True
        ) -> int:
        _n_set = 0
        for attr, val in attributes_dict.items():
            if keys_upper:
                attr = attr.upper()
            set_this_attribute = True
            if require_presence:
                if not attr in self.__dict__.keys():
                    set_this_attribute = False
            if set_this_attribute:
                if isinstance(val, list):
                    val = tuple(val)
                setattr(self, attr, val)
                _n_set += 1
        return _n_set
##########################