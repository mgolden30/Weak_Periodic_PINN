'''
PURPOSE:
I had a crazy idea a while ago about using a potential formalism to solve the nonlinear dynamics of NS 
exactly and then learn incompressibility and vorticity definition.

Well what if I did both? Learn the three-vector A and streamfunction psi and require they produce the same (w,uw,vw).
Should be numerically stable.
'''
import torch
import torch.nn as nn
import torch.nn.init as init

from lib.model import PeriodicLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class PotentialNetwork(nn.Module):
    def __init__(self, L):
        super().__init__()

        self.per    = PeriodicLayer()
        self.layer1 = nn.Linear(  6,  L ) #sin and cos for 3 inputs gives 6 dimensions
        self.layer2 = nn.Linear(  L,  L ) 
        self.layer3 = nn.Linear(  L,  4 ) #output is [At, Ax, Ay, psi]
        
        # initialize weights with Xavier uniform initialization
        init_range = 2
        init.xavier_uniform_(self.layer1.weight, gain=init_range)
        init.xavier_uniform_(self.layer2.weight, gain=init_range)
        init.xavier_uniform_(self.layer3.weight, gain=init_range)
    
    def forward(self, x):
        #force network to learn periodic functions of the input
        y = self.per(x)

        #linear + activation
        y = self.layer1(y)
        #y = torch.exp( -torch.square(y) )
        y = torch.cos(y)

        #linear + activation        
        y = self.layer2(y)
        #y = torch.exp( -torch.square(y) )
        y = torch.cos(y)

        #linear to get potentials
        y = self.layer3(y)

        return y
    
class PotentialHydroNetwork(nn.Module):
    def __init__(self, pot_net, nu, T0):
        super().__init__()
        self.pot_net = pot_net #store an internal potential network
        self.nu = nu
        self.T  = nn.Parameter( torch.tensor([T0]), requires_grad=True)  # period

    def forward(self, x):
        #Compute [N,4] matrix for [At, Ax, Ay, psi]
        potentials = self.pot_net(x)
        
        at  = potentials[:,0]
        ax  = potentials[:,1]
        ay  = potentials[:,2]
        psi = potentials[:,3]

        #autodiff the streamfunction
        dpsi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        u =  dpsi[:,1]
        v = -dpsi[:,0]
        w =-torch.autograd.grad(dpsi[:, 0], x, grad_outputs=torch.ones_like(dpsi[:, 0]), create_graph=True, retain_graph=True)[0][:, 0] \
           -torch.autograd.grad(dpsi[:, 1], x, grad_outputs=torch.ones_like(dpsi[:, 1]), create_graph=True, retain_graph=True )[0][:, 1]


        #autodiff the 3-vector
        w2 =  torch.autograd.grad( ay, x, grad_outputs=torch.ones_like(ay), create_graph=True, retain_graph=True)[0][:, 0] \
             -torch.autograd.grad( ax, x, grad_outputs=torch.ones_like(ax), create_graph=True, retain_graph=True)[0][:, 1]
        #rescale
        w2 = w2 * self.T / 2 / torch.pi

        uw =  torch.autograd.grad( at, x, grad_outputs=torch.ones_like(at), create_graph=True, retain_graph=True)[0][:, 1] \
             -torch.autograd.grad( ay, x, grad_outputs=torch.ones_like(ay), create_graph=True, retain_graph=True)[0][:, 2]

        vw =  torch.autograd.grad( ax, x, grad_outputs=torch.ones_like(ax), create_graph=True, retain_graph=True)[0][:, 2] \
             -torch.autograd.grad( at, x, grad_outputs=torch.ones_like(at), create_graph=True, retain_graph=True)[0][:, 0]

        dwdy = torch.autograd.grad( w2, x, grad_outputs=torch.ones_like(w2), create_graph=True, retain_graph=True)[0][:, 1]

        #Add forcing and viscous correction to vw
        vw = vw + self.nu*dwdy + torch.sin(4*x[:,1])

        u = torch.unsqueeze(u, dim=1 )
        v = torch.unsqueeze(v, dim=1 )
        w = torch.unsqueeze(w, dim=1 )
        uw= torch.unsqueeze(uw,dim=1 )
        vw= torch.unsqueeze(vw,dim=1 )
        w2= torch.unsqueeze(w2,dim=1 )
        
        f = torch.cat( (u, v, w, uw, vw, w2), dim=1)

        return f
    

class PotentialPINN(nn.Module):
    def __init__(self, hydro, points):
        super().__init__()
        self.hydro = hydro #store an internal potential network
        self.points = 
        
    def forward(self, x):
        #Compute [N,4] matrix for [At, Ax, Ay, psi]
        f = self.hydro(x)
        
        #Streamfunction variables
        u = f[:,0]
        v = f[:,1]
        w = f[:,2]

        #three-vector variables
        uw = f[:,3]
        vw = f[:,4]
        w2 = f[:,5]

        err1 =   w - w2 #vorticity should agree
        err2 = u*w - uw #product of vorticity and u_x should agree
        err3 = v*w - vw #product of vorticity and u_x should agree
        


        err1 = torch.unsqueeze(err1, dim=1)
        err2 = torch.unsqueeze(err2, dim=1)
        err3 = torch.unsqueeze(err3, dim=1)
        err = torch.cat( (err1, err2, err3), dim=1 )
        return err