import numpy as np
import os
import torch

from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PickDomains(nn.Module):
    def __init__(self, L):
        super().__init__()

        self.layer1 = nn.Linear(  3,  L ) #sin and cos for 3 inputs gives 6 dimensions
        self.layer2 = nn.Linear(  L,  L ) 
        self.layer3 = nn.Linear(  L,  1 ) #single output is streamfunction
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights and biases of layer3 with zeros
        nn.init.zeros_(self.layer3.weight)
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):

        y = self.layer1(x - np.pi)
        y = torch.cos( y )

        y = self.layer2(y)
        y = torch.cos( y )

        y = self.layer3(y)

        #so that y is a perturbation of the input
        y = x + y
        return y





class PeriodicLayer(nn.Module):
    def __init__(self):
        #This will have no internal parameters
        super().__init__()
    
    def forward(self, x):
        #take sin and cos and stack them
        #I am having a bizarre bug where stack isn't working.

        s = torch.sin(x)
        c = torch.cos(x)

        y = torch.cat( (s, c), dim=1  )
        return y



class StreamfunctionNetwork(nn.Module):
    def __init__(self, L):
        super().__init__()

        self.per    = PeriodicLayer()

        self.layer1 = nn.Linear(  6,  L ) #sin and cos for 3 inputs gives 6 dimensions
        self.layer2 = nn.Linear(  L,  L ) 
        self.layer3 = nn.Linear(  L,  1 ) #single output is streamfunction
        
        self.conv1 = nn.Conv1d( in_channels=1, out_channels=L, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #force network to learn periodic functions of the input
        y = self.per(x)

        #linear + gaussian activation
        y = self.layer1(y)
        #y = torch.exp( -torch.square(y) )
        y = torch.sin(y)

        #linear + gaussian activation        
        y = self.layer2(y)
        #y = torch.exp( -torch.square(y) )
        y = torch.sin(y)

        #linear to get streamfunction
        y = self.layer3(y)

        return y



class HydroNetwork(nn.Module):
    def __init__(self, streamnetwork):
        super().__init__()
        self.streamnetwork = streamnetwork        
        
        #internal parameters we can train for ECS
        self.T = nn.Parameter(torch.tensor([20.0]), requires_grad=True)  # period
        self.a = nn.Parameter(torch.tensor([0.0]),  requires_grad=True)  # drift

    def forward(self, x):
        psi = self.streamnetwork(x) #evaluate the streamfunction
        
        #autodiff the streamfunction
        dpsi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        # flow is u \hat{i} + v \hat{j}
        u =  dpsi[:,1]
        v = -dpsi[:,0]

        # Compute the Laplacian (divergence of the gradient)
        # w = -lap(psi)
        w =-torch.autograd.grad(dpsi[:, 0], x, grad_outputs=torch.ones_like(dpsi[:, 0]), create_graph=True, retain_graph=True)[0][:, 0] \
           -torch.autograd.grad(dpsi[:, 1], x, grad_outputs=torch.ones_like(dpsi[:, 1]), create_graph=True, retain_graph=True )[0][:, 1]

        #stack flow and interal parameters
        T = self.T * torch.ones( w.size(), requires_grad=True ).to(device)
        a = self.a * torch.ones( w.size(), requires_grad=True ).to(device)
        
        f = torch.stack( (w,u,v,T,a), dim=1 )

        return f
    
class WeakPINN(nn.Module):
    def __init__( self, hydro_model, nu, p ):
        super().__init__()
        self.model = hydro_model
        self.nu = nu
        
        #Legendre-Gauss quadrature information
        self.p = p #number of roots to use
        self.points, self.weights = np.polynomial.legendre.leggauss(self.p)
        self.points  = torch.tensor(self.points,  dtype=torch.float32, requires_grad=False)
        self.weights = torch.tensor(self.weights, dtype=torch.float32, requires_grad=False)

        D = 2*np.pi #domain size
        Hx = D/16
        Hy = D/16
        Ht = D/16
        self.H = np.array( [Hx, Hy, Ht] ) #sidelengths of boxes for weak formulation

        #keep these in the range [-1,1]
        x, y, t = torch.meshgrid( (self.points, self.points, self.points) )
        #print( x[1,0,0] - x[0,0,0] )
        #print( y[0,1,0] - y[0,0,0] )
        #print( t[0,0,1] - t[0,0,0] )
        
        self.displacement = torch.cat( ( self.H[0]/2*x[:,:,:,np.newaxis,np.newaxis],\
                                    self.H[1]/2*y[:,:,:,np.newaxis,np.newaxis],\
                                    self.H[2]/2*t[:,:,:,np.newaxis,np.newaxis]), axis=4 )
        self.displacement = self.displacement.to(device)

        #define 1d weight and its derivatives
        #since I am no longer enforcing the nonlinear equaitons, I only need first derivaives!
        phi1   = 1 - 2*self.points**2 + self.points**4
        dphi1  = -4*self.points + 4*self.points**3
        ddphi1 = -4 + 12 * self.points**2

        #construct 3d weights as products of 1d weights
        phi    = phi1[:, np.newaxis, np.newaxis] * phi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        
        phi_dx =dphi1[:, np.newaxis, np.newaxis] * phi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        phi_dy = phi1[:, np.newaxis, np.newaxis] *dphi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        phi_dt = phi1[:, np.newaxis, np.newaxis] * phi1[np.newaxis, :, np.newaxis] *dphi1[np.newaxis, np.newaxis, :]
        
        phi_ddx = ddphi1[:, np.newaxis, np.newaxis] *  phi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        phi_ddy =   phi1[:, np.newaxis, np.newaxis] *ddphi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        
        #make a 3D quadrature weight
        weight = self.weights[:, np.newaxis, np.newaxis] * self.weights[np.newaxis, :, np.newaxis] * self.weights[np.newaxis, np.newaxis, :]
        
        #add another dimension to all weight objects
        self.phi     = phi[:,:,:,np.newaxis].to(device)
        self.phi_dx  = phi_dx[:,:,:,np.newaxis].to(device)
        self.phi_dy  = phi_dy[:,:,:,np.newaxis].to(device)
        self.phi_dt  = phi_dt[:,:,:,np.newaxis].to(device)
        self.phi_ddx = phi_ddx[:,:,:,np.newaxis].to(device)
        self.phi_ddy = phi_ddy[:,:,:,np.newaxis].to(device)
        self.weight  = weight[:,:,:,np.newaxis].to(device)

    def forward(self, xs, xs_uniform):
        # equation input: 
        # xs = (x,y,t) of size [n,3]
        # xs_uniform is the same, but a static uniform mesh to estimate the state velocity on.
        #       We will compute a penalty term for not evolving in time.
        
        #Let zs be the quadrature points for all sampled subdomains
        #We will need three new axis for identifying quadrature points
        zs = xs[np.newaxis, np.newaxis, np.newaxis, ...]

        zs = zs + self.displacement

        #reshape the sample points to trick the network
        zs2 = torch.reshape( zs, [-1,3] )

        #compute fields f = [w,u,v,T,a] at all quadrature points
        f = self.model(zs2) 

        p = self.p
        f = torch.reshape( f, [p,p,p,-1,5] )

        #pull out the hydrodynamic fields
        w = f[...,0] #vorticity
        u = f[...,1] #x-component of the flow
        v = f[...,2] #y-component of flow
        T = f[...,3] #period
        a = f[...,4] #drift rate

        #Compute the forcing
        forcing = 4 * torch.cos( 4 * zs[...,1] )

        # rescalings for derivatives based on affine transformation to canonical [-1,1]
        rx = 2.0/self.H[0]
        ry = 2.0/self.H[1]
        rt = 2.0/self.H[2]
        rt2 = (2*np.pi)/T


        #equation 1: voriticity dynamics
        eq1= - rt2 * rt * self.phi_dt * w \
             - rx * self.phi_dx * ((u+a) * w) \
             - ry * self.phi_dy * (v * w) \
             - self.nu * ( rx*rx*self.phi_ddx + ry*ry*self.phi_ddy)*w \
             - self.phi * forcing
        err = torch.sum( self.weight * eq1, axis=[0,1,2] )


        #now, compute a penalty for not changing in time
        #we will just add a pole around dwdt = 0.
        '''
        fu  = self.model(xs_uniform)
        dfu = torch.autograd.grad(fu, xs_uniform, grad_outputs=torch.ones_like(fu), create_graph=True)[0]
        dwdt = dfu[..., 2]
        penalty = 1 + 1.0/torch.mean( dwdt**2 )
        err = err * penalty
        '''
        return err