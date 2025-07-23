import torch
import numpy as np

import math
from tqdm import tqdm
import typing as tp

from scipy import integrate, linalg
import sys
sys.path.append("/trinity/home/a.kolesov/EFM/")
from src.utils import from_flattened_numpy,  to_flattened_numpy


#############################
class DippoleGroundTrurthEFMODESolver:
    
    def __init__(self, config) -> None:
        self._config = config # private attribute
        
    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self,config) -> None:
        print("You modify configuration for running code")
        self._config = config
    
    def __call__(self, efm ,
                       perturbed_samples_vec: torch.Tensor ,
                       p_samples: torch.Tensor,
                       q_samples: torch.Tensor) -> tp.Sequence[torch.Tensor]: ## ???
        
        trajectory = [perturbed_samples_vec.clone().detach().cpu()] 
        #### uniform motion along axis z ####
        for step in tqdm(range(math.ceil(self._config.L//self._config.ode.step))):  
            
            field =  efm.GroundTruth(perturbed_samples_vec=perturbed_samples_vec,
                                     p_samples=p_samples,
                                     q_samples=q_samples)
            
            perturbed_samples_vec = perturbed_samples_vec +\
                                    (self._config.ode.step/field[:,0].view(-1,1) + self._config.ode.gamma)*field 
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
        #### uniform motion along axis z ####
        
         
        #### movement behind the second plate ####
        field = efm.GroundTruth(perturbed_samples_vec=perturbed_samples_vec,
                                     p_samples=p_samples,
                                     q_samples=q_samples)
        mask_start = field[:,0] > 0 # E_z > 0
        mask = mask_start 
        
        #while  torch.nonzero(mask).__len__() != 0:
        for _ in tqdm(range(self._config.ode.behind_num_steps)):
            
             
            field[mask] = efm.GroundTruth(perturbed_samples_vec=perturbed_samples_vec[mask],
                                     p_samples=p_samples,
                                     q_samples=q_samples)
            perturbed_samples_vec[mask] = perturbed_samples_vec[mask] +\
                                          (self._config.ode.behind_step/torch.norm(field[mask],keepdim=True))*field[mask]
            
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
            mask = torch.logical_and(mask_start, (perturbed_samples_vec[:,0] >= self._config.q.x_loc + 0.05).view(-1))  
        #### movement behind the second plate ####
        return perturbed_samples_vec, trajectory
#############################






#############################
class LearnDippoleEFMODESolver:
    
    def __init__(self, net, config):
        
        self._config = config
        self.net = net
     
    @property
    def config(self):
        return slef._config
    
    @config.setter
    def config(self,config):
        self._config = config
    
    def __call__(self, perturbed_samples_vec: torch.Tensor ,
                        p_samples: torch.Tensor,
                        q_samples: torch.Tensor) -> tp.Sequence[torch.Tensor]:
        
        trajectory = [perturbed_samples_vec.clone().detach().cpu()] 
        #### uniform motion along axis z ####
        for step in tqdm(range(math.ceil(self._config.L//self._config.ode.step))):  
            
            field =  self.net(perturbed_samples_vec)
            perturbed_samples_vec = perturbed_samples_vec +\
                                    (self._config.ode.step/field[:,0].view(-1,1) + self._config.ode.gamma)*field 
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
        #### uniform motion along axis z ####
        
         
        #### movement behind the second plate ####
        field =  self.net(perturbed_samples_vec)
        mask_start = field[:,0] > 0 # E_z > 0
        mask = mask_start 
        
        #while  torch.nonzero(mask).__len__() != 0:
        for _ in tqdm(range(self._config.ode.behind_num_steps)):
            
             
            field[mask] = self.net(perturbed_samples_vec[mask])
            perturbed_samples_vec[mask] = perturbed_samples_vec[mask] +\
                                          (self._config.ode.behind_step/torch.norm(field[mask],keepdim=True))*field[mask]
            
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
            mask = torch.logical_and(mask_start, (perturbed_samples_vec[:,0] >=  self._config.q.x_loc + 0.05).view(-1))
             
        #### movement behind the second plate ####
        
        
        return perturbed_samples_vec, trajectory 
#############################







#############################
def get_rk45_sampler_pfgm( y, config, shape,   rtol=1e-4, atol=1e-4,
                    method='RK45', eps=1e-3, device='cuda'):

    """RK45 ODE sampler for PFGM.

    Args:
    sde: An `methods.SDE` object that represents PFGM.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

    Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
    """

    
    def ode_sampler(model, y):

        x = y

        z = torch.ones((len(x), 1, 1, 1)).to(x.device)
        z = z.repeat((1, 1, config.data.image_size, config.data.image_size)) * config.L
        x = x.view(shape)
        # Augment the samples with extra dimension z
        # We concatenate the extra dimension z as an addition channel to accomondate this solver
        x = torch.cat((z, x), dim=1)
        x = x.float()
        new_shape = (len(x), config.data.num_channels + 1, config.data.image_size, config.data.image_size)
        
       
         

        

        def ode_func(t, x):

           


            x = from_flattened_numpy(x, new_shape).to(device).type(torch.float32)

            # Change-of-variable z=exp(t)
            z = np.exp(t)
            #net_fn = get_predict_fn(sde, model, train=False)

            x_drift, z_drift = model(x[:, 1:], torch.ones((len(x))).to(device) * z)
            x_drift = x_drift.view(len(x_drift), -1)

            # Substitute the predicted z with the ground-truth
            # Please see Appendix B.2.3 in PFGM paper (https://arxiv.org/abs/2209.11178) for details
            z_exp = config.sampling.z_exp

 

            if z < z_exp and config.training.gamma > 0:
                data_dim = config.data.image_size * config.data.image_size * config.data.num_channels
                sqrt_dim = np.sqrt(data_dim)
                norm_1 = x_drift.norm(p=2, dim=1) / sqrt_dim
                x_norm = config.training.gamma * norm_1 / (1 - norm_1)
                x_norm = torch.sqrt(x_norm ** 2 + z ** 2)
                z_drift = -sqrt_dim * torch.ones_like(z_drift) * z / (x_norm + config.training.gamma)

                
                
            # Predicted normalized Poisson field
            v = torch.cat([ z_drift[:, None], x_drift], dim=1)
            dt_dz = 1 / (v[:, 0] + 1e-5)
            dx_dt = v[:, 1:].view(shape)

            # Get dx/dz
            dx_dz = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
            # drift = z * (dx/dz, dz/dz) = z * (dx/dz, 1)
            drift = torch.cat([torch.ones((len(dx_dz), 1, config.data.image_size,
                                           config.data.image_size)).to(dx_dz.device) * z, z * dx_dz], dim=1)
            return to_flattened_numpy(drift)

 
        
        

        
        # Black-box ODE solver for the probability flow ODE.
        # Note that we use z = exp(t) for change-of-variable to accelearte the ODE simulation
        solution = integrate.solve_ivp(ode_func,
                                       (np.log(config.L),
                                                  np.log(config.training.epsilon)), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)

        nfe = solution.nfev
        num_itrs = len(solution.y[0])
        x = torch.tensor(solution.y[:, -1]).reshape(new_shape).to(device).type(torch.float32)
        
        trajectory = []
        visual_iters = np.linspace(int(num_itrs//8), num_itrs,  10)
       
        for itr in visual_iters:
            traj = torch.tensor(solution.y[:,int(itr)-1]).reshape(new_shape).to(device).type(torch.float32)
            trajectory.append(traj[:,1:].detach().cpu())
            
            
        # Detach augmented z dimension
        x = x[:, 1:]
        #x = inverse_scaler(x)
        return x, nfe, torch.stack(trajectory,dim=0)
 
    return ode_sampler


############################

  
    

    
#############################
class LearnedImageODESolver:

    def __init__(self, net, config):
        self.config = config
        self.net = net

    def __call__(self, x_init ):
        trajectory = [x_init[:,1:].view(-1,self.config.data.num_channels,
                                               self.config.data.image_size,
                                               self.config.data.image_size).clone().detach().cpu()]
        mask = torch.tensor(x_init.shape[0]*[True]).to(self.config.device)
        
        while mask.any():
 
                
            field_x, field_z = self.net(x_init[:,1:].view(-1,self.config.data.num_channels,
                                                             self.config.data.image_size,
                                                             self.config.data.image_size) , x_init[:,0]   )
            
            field = torch.cat([field_z.view(-1,1),
                               field_x.view(-1, self.config.data.num_channels*\
                                                self.config.data.image_size*\
                                                self.config.data.image_size)], dim=1) # [B, 1+C*H*W]
            
            # backward
            x_init  = x_init  - (self.config.ode.step/ ( field_z.view(-1,1)  + self.config.ode.gamma ))*field # [B, C*H*W+1]
            trajectory.append(x_init[:,1:].view(-1,self.config.data.num_channels,
                                               self.config.data.image_size,
                                               self.config.data.image_size).clone().detach().cpu())
            t = x_init[:,0]
            mask = t[0] > self.config.training.epsilon
            #mask = t[0] < self.config.L - self.config.training.epsilon
          
            
            
            
        return x_init, trajectory
#############################   
    
    
    
    
    
    
    



#############################

class BaseODESolver:

    def __init__(self, config):
        self.config = config

    def __call__(self, func, perturbed_samples_vec , p_samples, q_samples):
        
        trajectory = [perturbed_samples_vec.clone().detach().cpu()]
        mask = torch.tensor(perturbed_samples_vec.shape[0]*[True]).to(self.config.device)
        
        while mask.any():
            field = func( perturbed_samples_vec,
                          p_samples, q_samples, self.config)
           
            perturbed_samples_vec  = perturbed_samples_vec  +\
            (self.config.ode.step/ (field[:,0][:,None] + self.config.ode.gamma ))*field

            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
            mask = perturbed_samples_vec[:,0] < self.config.L 
            #print(torch.min(perturbed_samples_vec[:,0]))
            
        return perturbed_samples_vec, trajectory
#############################











#############################
class LearnedODESolver:

    def __init__(self, net, config):
        self.config = config
        self.net = net

    def __call__(self, perturbed_samples_vec, p_samples, q_samples):
        
        trajectory = [perturbed_samples_vec.clone().detach().cpu()]
        mask = torch.tensor(perturbed_samples_vec.shape[0]*[True])
        
        while mask.any():
 
                
            field = self.net(perturbed_samples_vec)
            perturbed_samples_vec  = perturbed_samples_vec  +\
                                     (self.config.ode.step/ (field[:,0][:,None] + self.config.ode.gamma ))*field
            trajectory.append(perturbed_samples_vec.clone().detach().cpu())
            mask = perturbed_samples_vec[:,0] < self.config.L 
 
        return perturbed_samples_vec, trajectory
#############################



 