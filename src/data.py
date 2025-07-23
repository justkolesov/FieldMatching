import torch
import numpy as np
import scipy
from sklearn.datasets import make_blobs, make_swiss_roll


###############################
class MultiDimStackedGauss:
    
    def __init__(self, config):
             
        self.config = config
        self.mu = config.p.mean
        self.std = config.p.std
        self.cov = config.p.cov
        self.x_loc = config.p.x_loc
        self.dist = torch.distributions.MultivariateNormal(loc = torch.tensor(self.mu), 
                    covariance_matrix = self.std*torch.eye(config.p.dim) if self.cov==None else self.cov)
        
    def sample(self, num_samples):
         
        samples = self.dist.sample(torch.tensor([num_samples])) 
        stacked = self.config.p.x_loc*torch.ones(num_samples) 
        samples = torch.cat([stacked[:,None], samples], dim=1)
        return samples
################################


 
################################    
class MultiDimStackedMixtureGauss:
    
    def __init__(self, config):
        
        self.config = config
        self.num_components = config.q.num_components
        self.mix_probs = config.q.mix_probs
        self.stds = config.q.stds
        self.cov = config.q.cov
        self.x_loc = config.q.x_loc
        self.dists = [torch.distributions.MultivariateNormal(loc = torch.tensor(mu), 
                      covariance_matrix = std*torch.eye(config.q.dim) if self.cov==None else self.cov)
                      for mu, std in zip(config.q.means, config.q.stds) ]
    
    def sample(self, num_samples):
        
        idxs = np.random.choice(range(self.num_components),
                                num_samples, p=self.mix_probs)
        
        samples = [self.dists[num].sample(torch.tensor([  np.bincount(idxs)[num] ])) 
                   for num in range(self.num_components)]
        
        samples = torch.cat(samples, dim=0)
        stacked = self.config.q.x_loc*torch.ones(num_samples)
        samples = torch.cat([stacked[:,None], samples], dim=1)
        
        idxs = torch.randperm(samples.shape[0])
        samples = samples[idxs]
        
        return samples
################################  



##########################
class Sampler:
    def __init__(
        self, device='cpu',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
##########################
    
    
    
###########################
class SwissRollSampler(Sampler):
    def __init__(
        self, config,  device='cpu'
    ):
        super(SwissRollSampler, self).__init__(device=device)
        assert  config.DIM  - 1 == 2
        self.config = config
        self.dim = config.DIM - 1
        self.x_loc = config.q.x_loc
   
        
    def sample(self, num_samples):
        
        batch =  make_swiss_roll(
            n_samples=num_samples,
            noise=0.8 if self.config.q.swiss_noise==None else self.config.q.swiss_noise
        )[0].astype('float32')[:, [0, 2]] / 5.
        
        batch = torch.from_numpy(batch)
        stacked =  self.config.q.x_loc*torch.ones(num_samples) 
        return torch.cat([stacked[:,None], batch], dim=1) 
###########################   
    
    