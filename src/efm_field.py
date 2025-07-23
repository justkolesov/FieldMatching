import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import wandb
import math
import typing as tp
from tqdm import tqdm
from IPython.display import clear_output

import sys
sys.path.append("/trinity/home/a.kolesov/EFM/")
from src.ode import get_rk45_sampler_pfgm, LearnedImageODESolver



class EFM:
    
    def __init__(self, config):
        self._config = config # private attribute
         
    @property
    def config(self):
        return self._config
    
    
    @config.setter
    def config(self, config):
        print("You modify the configuration for code running")
        self._config = config
    
    
    def __str__(self):
        return "Electrostatic field matching"
    
    
    def __repr__(self):
        return f"EFM({self._config})"
 


    ######################################
    @staticmethod
    def plotVectorField(*,mesh: torch.Tensor,
                         field:torch.Tensor, 
                         p_samples:torch.Tensor,
                         q_samples:torch.Tensor, **kwargs: tp.Any) -> matplotlib.figure.Figure:
        
        
        
        fig = plt.figure()
        fig.set_figheight(kwargs.get('figheight', 7))
        fig.set_figwidth( kwargs.get('figwidth', 7))

        ax = fig.add_subplot(1,1,1, projection='3d' )
        ax.scatter(p_samples[:,0].cpu(),p_samples[:,1].cpu(),p_samples[:,2].cpu(),
                   color='blue',edgecolor='black',s=80,label=r'$x_{+}  \sim \mathbb{P}(x_{+})$')
        ax.scatter(q_samples[:,0].cpu(),q_samples[:,1].cpu(),q_samples[:,2].cpu(),
                   color='red',edgecolor='black',s=80,label=r'$x_{-}  \sim \mathbb{Q}(x_{-})$')
        ax.quiver(mesh[:, 0].cpu(), mesh[:, 1].cpu(), mesh[:,2].cpu(),
                  field[:, 0].cpu(), field[:, 1].cpu(), field[:,2].cpu(),
                  color='black',length=1, normalize=True)
        ax.set_title(kwargs.get("title", "EFM Ground Truth"))
        ax.legend()
        return fig
    ######################################## 
    
    
    
    
    ######################################## 
    @staticmethod
    def plotTrajectories(*,traj: torch.Tensor,  
                         p_samples: torch.Tensor,
                         q_samples: torch.Tensor,**kwargs: tp.Any) -> matplotlib.figure.Figure:
        
        fig = plt.figure()
        fig.set_figheight(kwargs.get('figheight', 7))
        fig.set_figwidth( kwargs.get('figwidth', 7))
        
        ax = fig.add_subplot(1,1,1,  projection='3d' )
        ax.scatter(p_samples[:,0].cpu(),p_samples[:,1].cpu(),p_samples[:,2].cpu(),
                   color='blue',edgecolor='black',s=80, label=r'$x_{+}  \sim \mathbb{P}(x_{+})$')
        
         
        
        traj[-1][:,0] = 6.5
        for jdx in range(len(traj[-1])):
            
            ax.scatter(traj[-1][jdx,0].cpu(),traj[-1][jdx,1].cpu(),traj[-1][jdx,2].cpu(),
            color='lightgreen',edgecolor='black',zorder=20,label=r'$y \sim  T(x_{+})$' if jdx==0 else None,s=80)
            
            ax.scatter(q_samples[jdx,0].cpu(),q_samples[jdx,1].cpu(),q_samples[jdx,2].cpu(),
                   color='red',edgecolor='black',s=80, label=r'$x_{-}  \sim \mathbb{Q}(x_{-})$' if jdx==0 else None)
            
        for idx in range(200,230):
            ax.plot(traj[: ,idx,0].cpu(), 
                traj[:, idx,1].cpu(),
                traj[:, idx,2].cpu(),
                color='black',linewidth=0.5, zorder=3);
        ax.set_title(kwargs.get("title", "EFM Ground Truth"))
        ax.legend() 
        return fig
    ######################################## 
    
    
    
    ######################################
    @staticmethod
    def plot(x: torch.Tensor, **kwargs: tp.Any ) -> matplotlib.figure.Figure:
        
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              kwargs.get('figsize',5),
                              figsize=(kwargs.get('figsize',5),
                                       kwargs.get('figsize',5)))
        
        for idx in range(kwargs.get('figsize',5)):
            for jdx in range(kwargs.get('figsize',5)):
                ax[idx,jdx].imshow(x[idx,jdx])
                ax[idx,jdx].set_yticks([])
                ax[idx,jdx].set_xticks([])
        fig.tight_layout(pad=kwargs.get('pad',0.00001))       
        return fig
    ######################################


    ######################################
    @staticmethod
    def plota(x: torch.Tensor, **kwargs: tp.Any ) -> matplotlib.figure.Figure:
        
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              kwargs.get('figsize',5),
                              figsize=(kwargs.get('figsize',5),
                                       kwargs.get('figsize',5)))
        
        for idx in range(kwargs.get('figsize',5)):
            for jdx in range(kwargs.get('figsize',5)):
                ax[idx,jdx].imshow(x[idx,jdx].permute(1,2,0).cpu())
                ax[idx,jdx].set_yticks([])
                ax[idx,jdx].set_xticks([])
        fig.tight_layout(pad=kwargs.get('pad',0.00001))       
        return fig
    ######################################
    
    
    
    ######################################
    @staticmethod
    def plot_trajectory(traj: torch.Tensor, **kwargs: tp.Any) -> matplotlib.figure.Figure:
    
        fig,ax = plt.subplots(kwargs.get('figsize',5),
                              len(traj),
                              figsize=(len(traj),kwargs.get('figsize',5)),
                              sharex=True,sharey=True)
        
        for time in range(len(traj)):
            for idx in range(kwargs.get('figsize',5)):
                ax[idx,time].imshow(np.clip(traj[time,idx].permute(1,2,0).cpu().numpy()*255,0,255).astype(np.uint8))
                ax[idx,time].set_xticks([])
                ax[idx,time].set_yticks([])

        fig.tight_layout(pad=kwargs.get('pad',0.00001))
        return fig
    ######################################
    
    
    
    
    ########################################
    @staticmethod
    def get_mesh( kv : tp.Dict[str, float], 
                  mesh_num_points: tp.Optional[int] = None) -> torch.Tensor:
  
        num_points = 10 if mesh_num_points is None else mesh_num_points
    
        if len(kv) == 3:
            mesh = []
            z = torch.linspace(kv["z"][0],kv["z"][1], mesh_num_points)
            x = torch.linspace(kv["x"][0], kv["x"][1], mesh_num_points)
            y = torch.linspace(kv["y"][0], kv["y"][1], mesh_num_points)

            for i in x:
                for j in y:
                    for k in z:
                        mesh.append(torch.tensor([i,j,k]))
        else:
            raise ValueError("The length of DICT MESH should be 3!")
        return torch.stack(mesh, dim=0)
    ########################################
    
   
    
    
    ######################################## 
    def Toytrain(self,  p_dist ,
                    q_dist ,
                    net: tp.Callable[[torch.Tensor], torch.Tensor],
                    optimizer, **kwargs: tp.Any ): #-> tp.Sequence[tp.Callable[[torch.Tensor], torch.Tensor], tp.Sequence[int]]:
        
        
        losses = []
        for step in tqdm(range(self._config.training.training_steps)):

            optimizer.zero_grad()
            p_samples =  p_dist.sample(self._config.training.batch_size).to(self._config.device) #[B,D]
            q_samples =  q_dist.sample(self._config.training.batch_size).to(self._config.device) #[B,D]

            perturbed_samples_vec = self.forward_interpolation(p_samples, q_samples)

            field = self.GroundTruth(perturbed_samples_vec, p_samples.clone(),
                                                            q_samples.clone() ) 
            
            #field = math.sqrt(self._config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5)
            pred  = net(perturbed_samples_vec)
            loss = torch.mean((field - pred)**2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if kwargs.get("verbose",False):
                clear_output(wait=True)
                plt.plot(losses)
                plt.show()
            
        return net, losses
    ######################################## 
    
    
    
    #######################################
    def train(self, train_loader, eval_loader, net, optimizer, optimize_fn,
                    state,  **kwargs: tp.Any):
        
       
        train_iter = iter(train_loader)
        eval_iter = iter(eval_loader)
        
        for step in tqdm(range(self._config.training.n_iters  + 1)):
            
            
            ###############################
            try:
                batch_x,_ =  next(train_iter)
            except StopIteration:
                print('stop')
            else:
                train_iter = iter(train_loader)
                batch_x,_ =  next(train_iter)
            batch_x = batch_x.to(self._config.device)
            batch_y = torch.randn_like(batch_x).to(self._config.device)
            ###############################
            
            
            optimizer = state['optimizer']
            optimizer.zero_grad()
            
            
            ###############################
            perturbed_samples_vec = self.forward_interpolation(batch_x[:self._config.training.small_batch_size],
                                                               batch_y[:self._config.training.small_batch_size])
            
            try:
                assert torch.isnan(perturbed_samples_vec).any().item() == False
            except AssertionError:
                print('None values in perturbed samples between plates')
            else:
                perturbed_samples_vec = self.forward_interpolation(batch_x[:self._config.training.small_batch_size],
                                                               batch_y[:self._config.training.small_batch_size]) 
            ###############################
            
            
            
            ###############################
            field = self.GroundTruth(perturbed_samples_vec,
                                     torch.cat([self._config.p.x_loc*torch.ones(len(batch_x))[:,None].to(self._config.device),
                                                  batch_x.view(-1,self._config.DIM-1)], dim=1),
                                     torch.cat([self._config.q.x_loc*torch.ones(len(batch_y))[:,None].to(self._config.device),
                                                  batch_y.view(-1,self._config.DIM-1)], dim=1))
            
            try:
                assert torch.isnan(perturbed_samples_vec).any().item() == False
            except AssertionError:
                print('None values in Ground Truth field')
            else:
                field = self.GroundTruth(perturbed_samples_vec,
                                     torch.cat([self._config.p.x_loc*torch.ones(len(batch_x))[:,None].to(self._config.device),
                                                  batch_x.view(-1,self._config.DIM-1)], dim=1),
                                     torch.cat([self._config.q.x_loc*torch.ones(len(batch_y))[:,None].to(self._config.device),
                                                  batch_y.view(-1,self._config.DIM-1)], dim=1))
            ###############################
            
            
            #field = math.sqrt(self._config.DIM)*field/( torch.norm(field, dim=1, keepdim=True) + 1e-5)
            
            
            perturbed_samples_x = perturbed_samples_vec[:, 1:].view(-1,self._config.data.num_channels,
                                                                    self._config.data.image_size,
                                                                    self._config.data.image_size)
            perturbed_samples_z = perturbed_samples_vec[:, 0]
            net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
            
            
            ###############################
            try:
                assert torch.isnan(net_x).any().item() == False
                assert torch.isnan(net_z).any().item() == False
            except AssertionError:
                print('None values in network prediction')
            else:
                net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
            ###############################    
                
            net_x = net_x.view(net_x.shape[0], -1)
            # Predicted N+1-dimensional Poisson field
            pred = torch.cat([net_z[:, None], net_x], dim=1)
             
            loss = torch.mean((field - pred)**2)
 
            loss.backward()
            optimize_fn(optimizer, net , step=state['step'], config=self._config)
            state['step'] += 1
            state['ema'].update(net.parameters())
            wandb.log({"loss train":loss.item()},step=step)
            
            
            if step % self._config.training.eval_freq == 0:
        
                try:
                    batch_x,_ =  next(eval_iter)
                except StopIteration:
                    print('stop')
                else:
                    eval_iter = iter(eval_loader)
                    batch_x,_ =  next(eval_iter)
                batch_x = batch_x.to(self._config.device) 
                batch_y = torch.randn_like(batch_x).to(self._config.device)


                with torch.no_grad():
                    ema = state['ema']
                    ema.store(net.parameters())
                    ema.copy_to(net.parameters())
                    
                    perturbed_samples_vec = self.forward_interpolation(batch_x[:self._config.training.small_batch_size],
                                                                       batch_y[:self._config.training.small_batch_size])
            
                    field = self.GroundTruth(perturbed_samples_vec,
                                             torch.cat([self._config.p.x_loc*\
                                                          torch.ones(len(batch_x))[:,None].to(self._config.device),
                                                          batch_x.view(-1,self._config.DIM-1)], dim=1),
                                             torch.cat([self._config.q.x_loc*\
                                                          torch.ones(len(batch_y))[:,None].to(self._config.device),
                                                          batch_y.view(-1,self._config.DIM-1)], dim=1))
                    
                    #field = field/( torch.norm(field, dim=1, keepdim=True) + 1e-5) 
                    perturbed_samples_x = perturbed_samples_vec[:, 1:].view(-1,self._config.data.num_channels,
                                                                    self._config.data.image_size,
                                                                    self._config.data.image_size)
                    perturbed_samples_z = perturbed_samples_vec[:, 0]
                    net_x, net_z = net(perturbed_samples_x, perturbed_samples_z)
                    net_x = net_x.view(net_x.shape[0], -1)
                    # Predicted N+1-dimensional Poisson field
                    pred = torch.cat([net_z[:, None], net_x], dim=1)

                    eval_loss = torch.mean((field - pred)**2)
   
                    ema.restore(net.parameters())
                    wandb.log({"loss eval":eval_loss.item()},step=step)
                    
            
            # sampling #
            
            if step % self._config.training.snapshot_freq == 0:
                
                with torch.no_grad():
                    ema.store(net.parameters())
                    ema.copy_to(net.parameters())

                    shape = (25, self._config.data.num_channels,
                                 self._config.data.image_size, self._config.data.image_size)

                    batch_y = torch.randn(*shape)
                    
                    # first sampling procedure #
                    sampling_fn = get_rk45_sampler_pfgm(y=batch_y , config=self._config,
                                                       shape=shape,
                                                       eps=self._config.training.epsilon,
                                                       device=self._config.device)
                    sample, n, traj = sampling_fn(net, batch_y)
                    # first sampling procedure #
                    
                    sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    batch_y = np.clip(batch_y.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    fig_1 = self.plot(sample.reshape(5,5,32,32,3) )
                    fig_2 = self.plot(batch_y.reshape(5,5,32,32,3) )
                    fig_3 = self.plot_trajectory(traj)
                    wandb.log({"Generated Images RK45":fig_1},step=step)
                    wandb.log({"Init Images":fig_2},step=step)
                    wandb.log({"Trajectories RK45":fig_3},step=step)


                    
                    # second sampling procedure #
                    ode_solver = LearnedImageODESolver(net , self._config)
                    batch_y = torch.randn(*shape).to(self._config.device)
                    sample, traj = ode_solver(torch.cat([(self._config.L)*torch.ones(batch_y.shape[0],
                                                                                    device=batch_y.device)[:,None],
                                                         batch_y.view(-1, self._config.DIM-1)],dim=1).to(self._config.device))
                    # second sampling procedure #
                    
                    #sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    #batch_y = np.clip(batch_y.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                     
                    fig_1 = self.plota(sample[:,1:].reshape(5,5,3,32,32).detach().cpu() )
                    #fig_3 = self.plot_trajectory(traj)
                    wandb.log({"Generated Images Euler":fig_1},step=step)
                    #wandb.log({"Trajectories Euler":fig_3},step=step)
    
        return net, state
    #######################################
    



    
    ######################################## 
    def forward_interpolation(self,
                              p_samples: torch.Tensor,
                              q_samples: torch.Tensor) -> torch.Tensor:
        
        """
        The definition Inter-plate points between plates.
        
        Input:
        p_samples - torch.Size([b,D+1]) or torch.Size([b,C,H,W])
        q_samples - torch.size([b,D+1]) or torch.Size([b,C,H,W])
        
        Return:
        perturbed_vec_samples - torch.Size([B,D+1])
        """
        
        

        #################################################
        ######       Mesh Interpolation (Toy)      ######
        #################################################
        if self._config.training.interpolation == 'mesh':
            
            mesh = self.get_mesh(self._config.KV, self._config.mesh_num_points).to(self._config.device)
            idxs = torch.randperm(mesh.shape[0])
            perturbed_samples_vec = mesh[idxs][:self._config.training.small_batch_size]
        #################################################
        ######   Mesh Interpolation (Toy)          ######
        #################################################   
        

        
        
        #################################################
        ######   Uniform Interpolation             ######
        ################################################# 
        elif self._config.training.interpolation == 'Uniform':

            t = torch.distributions.Uniform(low=self._config.p.x_loc + self._config.training.epsilon,
                high=self._config.q.x_loc - self._config.training.epsilon).sample(\
                torch.Size([self._config.training.small_batch_size])).to(self._config.device)[:, None] #[b,1]
            
            den = self._config.L - 2*self._config.training.epsilon
            
            """
            perturbed_x = (t-self._config.training.epsilon)/den*\
                          q_samples[:self._config.training.small_batch_size,1:]\
               +(1-(t-self._config.training.epsilon)/den)*\
               p_samples[:self._config.training.small_batch_size,1:] #[b, D+1]
            """
            
            perturbed_x = q_samples*(t[:,None,None]/self._config.L) + (1 - t[:,None,None]/self._config.L)*p_samples
            perturbed_samples_vec = torch.cat([t,
                                               perturbed_x.reshape(len(p_samples), self._config.DIM-1)], dim=1)
            
            #perturbed_samples_vec = torch.cat([t, perturbed_x], dim=1) 
        #################################################
        ######   Uniform Interpolation             ######
        ################################################# 



        
        #################################################
        ######           PFGM Interpolation        ######
        #################################################
        elif self._config.training.interpolation == 'Gaussian_mixing':

            assert p_samples.shape == torch.Size([self._config.training.small_batch_size, self._config.data.num_channels,
                                                  self._config.data.image_size, self._config.data.image_size]) 
            assert p_samples.shape == q_samples.shape
            
            ####### perturbation for z component #######
            m = torch.rand((p_samples.shape[0],), device=p_samples.device) * self._config.training.M  #[b] : m ~ U[0,M]
            tau = self._config.training.tau 
            z = torch.randn( p_samples.shape[0], 1).to(p_samples.device)*self._config.training.sigma_end #[b, 1] : influence on the performance ??
            z = z.abs()  # [b,1]  : z = epsilon + N(0,I)
            assert z.shape==torch.Size([self._config.training.small_batch_size, 1])
                                      
            # confine norms
            # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
            """
            if config.training.restrict_M:
                idx = (z < self._config.training.epsilon + 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            """
            if self._config.training.restrict_M:
                idx = (z < 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            
 
            multiplier = (1+tau) ** m # torch.Size([b]) : the essence of this form??
            perturbed_z = z.squeeze() * multiplier # torch.Size([b])* torch.Size([b]) = torch.Size([b])
            """
            perturbed_z = torch.clamp(perturbed_z, min=self._config.training.epsilon ,
                                                   max=self._config.L - self._config.training.epsilon) #torch.Size([b])
            """
            ####### perturbation for z component #######
            

            ####### perturbation for x component #######
            # Sample uniform angle
            gaussian = torch.randn(p_samples.shape[0], self._config.DIM-1).to(p_samples.device) # torch.Size([b, C*H*W=D])
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True) #  torch.Size([b, C*H*W=D])
            noise = torch.randn_like(p_samples).reshape(p_samples.shape[0] , 
                                                        -1) * self._config.training.sigma_end #torch.Size([b, C*H*W=D])
            norm_m = torch.norm(noise, p=2, dim=1) * multiplier # torch.Size([b])*torch.Size([b]) = torch.Size([b])

            # Construct the perturbation for x
            perturbation_x = unit_gaussian * norm_m[:, None] # torch.Size([b,C*H*W])* torch.Size([b,1])=  torch.Size([b,C*H*W=D])
            perturbation_x = perturbation_x.view_as(p_samples) # torch.size([b,C,H,W])
            perturbed_x = p_samples + perturbation_x  # torch.size([b,C,H,W])  + torch.size([b,C,H,W])
            ####### perturbation for x component #######
            
            perturbed_samples_vec = torch.cat((perturbed_z[:, None],
                                               perturbed_x.reshape(p_samples.shape[0], self._config.DIM - 1),
                                               ), dim=1) #[b, D+1]
        #################################################
        ######           PFGM Interpolation        ######
        #################################################

        elif self._config.training.interpolation == 'Uniform_mixing':
            
            
            assert p_samples.shape == torch.Size([self._config.training.small_batch_size, self._config.data.num_channels,
                                                  self._config.data.image_size, self._config.data.image_size]) 
            assert p_samples.shape == q_samples.shape
            
            ####### perturbation for z component #######
            m = torch.rand((p_samples.shape[0],), device=p_samples.device) * self._config.training.M  #[b] : m ~ U[0,M]
            tau = self._config.training.tau 
            z = torch.randn( p_samples.shape[0], 1).to(p_samples.device)*self._config.training.sigma_end #[b, 1] : influence on the performance ??
            z = z.abs()  # [b,1]  : z = epsilon + N(0,I)
            assert z.shape==torch.Size([self._config.training.small_batch_size, 1])
                                      
            # confine norms
            # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
            """
            if config.training.restrict_M:
                idx = (z < self._config.training.epsilon + 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            """
            if self._config.training.restrict_M:
                idx = (z < 0.005).squeeze()
                num = int(idx.int().sum())
                restrict_m = int(self._config.training.M * 0.7)
                m[idx] = torch.rand((num,), device=p_samples.device) * restrict_m
            
 
            multiplier = (1+tau) ** m # torch.Size([b]) : the essence of this form??
            perturbed_z = z.squeeze() * multiplier # torch.Size([b])* torch.Size([b]) = torch.Size([b])
            
            
            perturbed_x = q_samples*(perturbed_z[:,None,None,None]/self._config.L) + (1 - perturbed_z[:,None,None,None]/self._config.L)*p_samples
            perturbed_samples_vec = torch.cat([perturbed_z[:, None],
                                               perturbed_x.reshape(len(p_samples), self._config.DIM-1)], dim=1)
            
        elif self._config.training.interpolation == 'both_side':
            pass
    
        return perturbed_samples_vec
    ########################################        
                     





        
        
    ########################################                                   
    def GroundTruth(self, 
                    perturbed_samples_vec: torch.Tensor,
                    p_samples: torch.Tensor,
                    q_samples: torch.Tensor,
                    **kwargs: tp.Any) -> torch.Tensor:
        """
        input:

        perturbed_samples_vec - torch.Size([b,D+1]) 
        p_samples - torch.Size([B,D+1])
        q_samples - torch.Size([B,D+1])
        config

        output:

        Superposition field - torch.Size([B, D+1])

        source: https://github.com/Newbeeer/Poisson_flow/blob/main/losses.py
        """
                                      
        gt_distance_x = torch.norm((perturbed_samples_vec.unsqueeze(1) - p_samples),dim=-1) # [b,B]
        gt_distance_y = torch.norm((perturbed_samples_vec.unsqueeze(1) - q_samples),dim=-1) # [b,B]
         
        #assert gt_distance_x.shape==torch.Size([self._config.training.small_batch_size,self._config.training.batch_size])
    

        # For numerical stability, timing each row by its minimum value
        if self._config.training.stability:
            distance_x = torch.min(gt_distance_x, dim=1, keepdim=True)[0] / (gt_distance_x + 1e-7) #[b,1]/[b,B] = [b,B]
            distance_y = torch.min(gt_distance_y, dim=1, keepdim=True)[0] / (gt_distance_y + 1e-7) #[b,1]/[b,B] = [b,B]
        else:
            distance_x = 1./ (gt_distance_x + 1e-7) # [b,B]
            distance_y = 1./ (gt_distance_y + 1e-7) # [b,B]

        data_dim = self._config.DIM # N+1
        distance_x = distance_x ** data_dim # [b,B]
        distance_y = distance_y ** data_dim # [b,B]


        distance_x = distance_x[:, :, None] # [b,B,1]
        distance_y = distance_y[:, :, None] # [b,B,1]

        # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)

        coeff_x = distance_x / (torch.sum(distance_x, dim=1, keepdim=True)  ) # [b,B,1]
        coeff_y = distance_y / (torch.sum(distance_y, dim=1, keepdim=True)  ) # [b,B,1]

        diff_x = - (perturbed_samples_vec.unsqueeze(1) - p_samples) # [b,B,D+1]
        diff_y = - (perturbed_samples_vec.unsqueeze(1) - q_samples) # [b,B,D+1]

        # Calculate empirical Poisson field (N+1 dimension in the augmented space)
        gt_direction_x = torch.sum(coeff_x * diff_x, dim=1) #[b,D+1]
        gt_direction_y = torch.sum(coeff_y * diff_y, dim=1) #[b,D+1]
        assert len(gt_direction_x.shape)==2
        assert gt_direction_x.shape[1]==self._config.DIM 


        gt_direction_x = gt_direction_x.view(gt_direction_x.size(0), -1)#[b,D+1]
        gt_direction_y = gt_direction_y.view(gt_direction_y.size(0), -1)#[b,D+1]

        
        # Normalizing the N+1-dimensional Poisson field
        gt_norm_x = gt_direction_x.norm(p=2, dim=-1)
        gt_norm_y = gt_direction_y.norm(p=2, dim=-1)

        #if kwargs.get('Normalized',True):
        if True:
            gt_direction_x /= (gt_norm_x.view(-1, 1) + self._config.training.gamma)
            gt_direction_y /= (gt_norm_y.view(-1, 1) + self._config.training.gamma)

        
        gt_direction_x *= np.sqrt(self._config.DIM)
        gt_direction_y *= np.sqrt(self._config.DIM)
        
        
        return  - gt_direction_x +  gt_direction_y                             
    ######################################## 
    