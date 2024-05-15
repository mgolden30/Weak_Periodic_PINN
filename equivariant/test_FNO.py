import torch as torch
from scipy.io import savemat
import numpy as np

from lib.SymmetryFactory import SymmetryFactory
from lib.FourierNeuralOperator import FourierNeuralOperator

#Let's check if autodiff works
torch.autograd.set_detect_anomaly(True)

b  = 1
c1 = 1
c2 = 7
n1 = 64
n2 = 32


def grid(n):
    # Generate the x and y values in the range [0, 2*pi]
    x_1d = np.linspace(0, 2*np.pi, n, endpoint=False)  # Exclude endpoint to make the domain periodic
    y_1d = np.linspace(0, 2*np.pi, n, endpoint=False)
    x, y = np.meshgrid(x_1d, y_1d, indexing='ij')
    return x, y

x,y = grid(n1)
x = torch.tensor(x)
y = torch.tensor(y)


#initialize a Fourier mode at the Nyquist frequency
nyq = n1//2 #Nyquist of the downsampled data. See if it survives with correct amplitude
kx = 2 #nyq
ky = 3 #nyq
phase = 1.2345
data = torch.cos( kx * x + ky * y + phase)

data = data.view( b, c1, n1, n1)
data.requires_grad = True


#Create a fourier operator
fno = FourierNeuralOperator(c1, c2, n2)
data2 = fno.downsample(data, fourier_input=False, fourier_output=False)


symm = SymmetryFactory()
dx = 0.1
dy = 0.2
data2 = symm.continuous_translation(data2, dx, dy)

data  = symm.continuous_translation(data,  dx, dy)
data3 = fno.downsample(data, fourier_input=False, fourier_output=False)


diff = data2 - data3
err = torch.mean(torch.abs(diff))
print(f"mean(abs(diff)) under continuous translation is {err}")

#Verify backwards gradient
criterion = torch.nn.L1Loss()

loss = criterion( data2, 0*data2 )
loss.backward()

data  = data.detach().numpy()
data2 = data2.detach().numpy()
data3 = data3.detach().numpy()

dict = {"data": data, "data2": data2, "data3": data3 }
savemat("fno_test.mat", dict)