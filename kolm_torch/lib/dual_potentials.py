import torch
from torch import nn
import numpy as np
from lib.model import PeriodicLayer

class CosActivation(nn.Module):
    def forward(self, x):
        #d = nn.Dropout(p=0.05)
        return torch.cos(x)

class DualPotential(nn.Module):
    def __init__(self, input_dim=3, output_dim=4, num_layers=2, num_features=32, T_init=20.0, nu=1.0/40):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_features = num_features
        self.nu = nu

        #period
        self.T = nn.Parameter(torch.tensor(T_init))  # Trainable parameter T

        activation = CosActivation()
        per = PeriodicLayer()

        # Main network
        main_layers = []

        main_layers.append(per)
        main_layers.append(nn.Linear( 6, self.num_features))  # Input layer
        main_layers.append(activation)  # Activation function
        for _ in range(num_layers):
            main_layers.append(nn.Linear(self.num_features, self.num_features))  # Hidden layers
            main_layers.append(activation)  # Activation function
        main_layers.append(nn.Linear(self.num_features, self.output_dim))  # Output layer
        self.main_model = nn.Sequential(*main_layers)

        # Subnetworks
        self.subnetworks = nn.ModuleList()
        for i in range(self.input_dim):
            sub_layers = []
            sub_input_dim = input_dim - 1  # One input dimension is discarded
            sub_output_dim = 1

            #if force_periodic:
            sub_layers.append(per)
            sub_layers.append(nn.Linear( 4, self.num_features))  # Input layer
            sub_layers.append(activation)  # Activation function
            for _ in range(num_layers):
                sub_layers.append(nn.Linear(self.num_features, self.num_features))  # Hidden layers
                sub_layers.append(activation)  # Activation function
            sub_layers.append(nn.Linear(self.num_features, sub_output_dim))  # Output layer
            self.subnetworks.append(nn.Sequential(*sub_layers))

    def period(self):
        return self.T.detach()

    def potentials(self, x):
        '''
        Evaluate the potentials
        '''
        #[At, Ax, Ay, \psi]
        potentials = self.main_model(x)
        a   = potentials[:,0:3]
        psi = potentials[:,3]

        #Compute the non-periodic degrees of freedom for A
        sub_outputs = []
        for i in range(self.input_dim):
            sub_input = torch.cat((x[:, :i], x[:, i + 1:]), dim=1)  # Discard one input variable
            sub_output = self.subnetworks[i](sub_input)
            sub_outputs.append(sub_output)

        #Add the laminar psi to the network
        #psi = psi - 1/self.nu/64*torch.cos( 4 * x[:,2] )

        return a, psi, sub_outputs


    def hydro(self, x):
        '''
        Evaluate the hydrodynamic fields using both the vector potential and the streamfunction.
        Coordinate ordering is (t,x,y)
        '''

        a, psi, h = self.potentials(x)

        ############################################
        # STEP 1: compute from streamfunction \psi
        ############################################

        # Compute gradients of psi with respect to x[:,1] and x[:,2]
        dpsi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)
        dpsi_x1  = dpsi[0][:, 1]  # Gradient of psi with respect to x[:,1]
        dpsi_x2  = dpsi[0][:, 2]  # Gradient of psi with respect to x[:,2]
        ddpsi_x1 = torch.autograd.grad(dpsi_x1, x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0][:,1]
        ddpsi_x2 = torch.autograd.grad(dpsi_x2, x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0][:,2]

        u = dpsi_x2 #u = \partial_y \psi
        v =-dpsi_x1 #u =-\partial_x \psi
        w = -ddpsi_x1 - ddpsi_x2 # \omega = -\partial^2 \psi

        ############################################
        # STEP 2: compute from vector A
        ############################################
        dat = torch.autograd.grad(a[:,0], x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        dax = torch.autograd.grad(a[:,1], x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        day = torch.autograd.grad(a[:,2], x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        

        w2 = day[:,1] - dax[:,2] - torch.squeeze(h[0]) # \omega = \partial_x A_y - \partial_y A_x - ht
        w2 = w2 * 2 * torch.pi / self.T

        #compute x and y derivs of w2
        dw2 = torch.autograd.grad(a[:,0], x, grad_outputs=torch.ones_like(psi), create_graph=True, retain_graph=True)[0]
        
        uw2 = dat[:,2] - day[:,0] + self.nu * dw2[:,1] - torch.squeeze(h[1])
        vw2 = dax[:,0] - dat[:,1] + self.nu * dw2[:,2] - torch.squeeze(h[2]) + torch.sin(4*x[:,2])

        #return hydrodynamic fields from both networks
        return w, u*w, v*w, w2, uw2, vw2

    def forward(self, x):
        w, uw, vw, w2, uw2, vw2 = self.hydro(x)

        return torch.cat( (w-w2, uw-uw2, vw-vw2) )


    def weak_forward(self, x, p, domain_size):
        '''
        Instead of reporting the differences at points, return INTEGRATED values along 1D spacetime lines.

        x - centers of lines
        p -number of points used in approximating integrals

        '''
        
        #gt points along [-1,1] for really accurate integrals
        y, weights = np.polynomial.legendre.leggauss(p)

        #make these torch tensors
        y       = torch.tensor(y,       dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        #Uniformly sample a sphere and make random unit vectors r
        n = x.shape[0]
        cos_th = 2*torch.rand((n,1))-1
        phi    = 2*torch.pi*torch.rand(n,1)
        sin_th = torch.sqrt( 1-cos_th*cos_th )
        r = torch.cat( (torch.sin(phi)*sin_th, torch.cos(phi)*sin_th, cos_th ), dim=1 )
        r = torch.reshape( r, [n,1,3] )

        #x is [N,3] currently. Let's make it [N,p,3]
        x = torch.reshape( x, [-1, 1, 3] )
        y = torch.reshape( y, [ 1, p, 1] ) * r #rotate along many random vectors so lines can be in any direction
        x = x + y*(domain_size/2)

        #trick the network into thinking we are just evaluating at more points
        x = torch.reshape( x, [-1,3] )
        w, uw, vw, w2, uw2, vw2 = self.hydro(x)

        w  = torch.reshape(w,   [n, p])
        w2 = torch.reshape(w2,  [n, p])
        uw = torch.reshape(uw,  [n, p])
        uw2= torch.reshape(uw2, [n, p])
        vw = torch.reshape(vw,  [n, p])
        vw2= torch.reshape(vw2, [n, p])
        
        weights = torch.reshape(weights, [1,p])

        err1 = torch.sum( weights*( w- w2), dim=1 )
        err2 = torch.sum( weights*(uw-uw2), dim=1 )
        err3 = torch.sum( weights*(vw-vw2), dim=1 )
        
        #Let's remove the approximate scaling degree of freedom w -> lambda w
        ws = torch.sum( weights*torch.abs(w), dim=1 )
        ws = torch.mean(ws)
        uws= torch.sum( weights*torch.abs(uw), dim=1 )
        uws= torch.mean(uws)
        #use the forcing to normalize this one
        vws= torch.sum( weights, dim=1 )
        vws= torch.mean(vws)


        err1 = err1/ws
        err2 = err2/uws
        err3 = err3/vws

        err = torch.cat( (err1,err2,err3), dim=0 )

        

        return err
