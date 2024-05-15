'''
This script is for me to derive the new Fourier coefficients when changing resolution in a equivariant way
'''

import torch as torch
from scipy.io import savemat



n = 64

def my_grid(n):
    grid = torch.arange(n)/n*(2*torch.pi)
    x,y = torch.meshgrid( grid, grid, indexing="ij" )
    return x, y


#Define a test function
f = lambda x,y : torch.sin( torch.pi * torch.exp(torch.cos(x)) / (1 + 0.5*torch.sin(x+y-1) ))

xl,yl = my_grid(n)   #large grid
xs,ys = my_grid(n/2) #small grid

fl = f(xl, yl)
fs = f(xs, ys)

cl = torch.fft.rfft2(fl)
cs = torch.fft.rfft2(fs)
cs2= torch.fft.rfft2(fl[0::2,0::2])

dict = { "cl": cl, "cs": cs, "xl": xl, "yl": yl, "cs2":cs2 }
savemat("subsample.mat", dict)