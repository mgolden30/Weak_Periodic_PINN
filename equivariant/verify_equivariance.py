'''
The purpose of this script is to check that what I am doing is actually group equivariant
'''

import torch as torch
from lib.SymmetryFactory import SymmetryFactory
from lib.EquivariantAutoencoder import EquivariantAutoencoder
from scipy.io import loadmat, savemat

import lib.utils as ut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set PyTorch seed for reproducibility
seed_value = 123
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

#Load training data
x, y, w = ut.load_vorticity_data()

#Batch size
batch = 70
w = w[:batch, :, :]

#For the equivariant autoencoder, I am also going to pass the forcing 4 cos (4y)
force = 4*torch.cos(4*y)
force = force.unsqueeze(0) #add batch index
force = force.unsqueeze(1) #add channel index
force = force.repeat((batch,1,1,1)) #repeat over batch dimension

w_batch = w[0:batch, :, :]
w_batch = w_batch.unsqueeze(1)

input = torch.cat( (w_batch, force), dim=1 )
input = input.to(device)


##############################
# Define autoencoder
##############################
lc = 2 #Number of latent images
ch = 16
enc_res = [ 64, 32, 16,  8 ] # encoder resolution sequence
enc_c   = [  2, ch, ch, ch ] # output conv channels
dec_res = [  8, 16, 32, 64 ] # decoder resolution sequence
dec_c   = [ lc, ch, ch, ch ] # output conv channels 

network = EquivariantAutoencoder( lc, enc_res, dec_res, enc_c, dec_c )

#Load a mildly trained
network.load_state_dict(torch.load("ch_scaling/ch_16.pth", map_location=torch.device('cpu')))
network = network.to(device)

symm = SymmetryFactory()

###########################################################
# Check that at all levels, the mean of each field is zero
###########################################################

latent, _ = network.encode(input)
output, _ = network.decode(latent)

def check_mean( field, name ):
    mean  = torch.mean( field, dim=[2,3] )
    bound = torch.max(torch.abs(mean))
    if bound < 1e-5:
        print(f"PASSED: The mean of {name} is bounded by {bound}. Shape is {field.shape}")
    else:
        print(f"FAILED: The mean of {name} is bounded by {bound}. Shape is {field.shape}")

print("Checking that mean vorticity is zero in all channels")
check_mean( input,  "input "  )
check_mean( latent, "latent" )
check_mean( output, "output" )
print("\n\n")


def check_equivariance(network, s1, s2):
    #Rotation first
    latent1, _ = network.encode( s1(input) )
    output1, _ = network.decode( latent1 )

    #Rotation second
    latent2, _ = network.encode( input )
    output2, _ = network.decode( latent2 )
    output2    = s1(output2)

    diff_latent = torch.mean(torch.abs( latent1 - s2(latent2) ), dim = [0,1,2,3] )
    diff_output = torch.mean(torch.abs( output1 - output2), dim=[0,1,2,3] )
    print(f"Mean difference in latent space is {diff_latent}")
    print(f"Mean difference in output space is {diff_output}")


###################################
# First test: 90 degree rotations
###################################
print("\nChecking for equivariance of 90 degree rotations")

s1 = lambda w : symm.rot90(w) #full vorticity space
s2 = lambda w : symm.rot90(w) #latent space
check_equivariance(network, s1, s2)

############################
# Translation Equivariance
############################
print("\nChecking for equivariance of translations")

n1 = 64 #vorticity resolution 
n2 =  4 #latent space resolution

#physical shifts
dx = 0.1
dy = 0.234

s1 = lambda w : symm.continuous_translation(w, dx, dy) #vorticity
s2 = lambda w : symm.continuous_translation(w, dx, dy) #latent space
check_equivariance(network, s1, s2)





###########################
# Relfection symmetry!
###########################
print("\nChecking for equivariance of reflections")


s1 = lambda w : -symm.transpose(w) #vorticity
s2 = lambda w : -symm.transpose(w) #latent
check_equivariance(network, s1, s2)


###########################
# Relfection symmetry (WRONG)
###########################
print("\nChecking for equivariance of unphysical reflections")

#Forget to change sign of w
s1 = lambda w : symm.transpose(w) #vorticity
s2 = lambda w : symm.transpose(w) #latent
check_equivariance(network, s1, s2)