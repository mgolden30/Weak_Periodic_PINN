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

#w is saved as [n,n,nt,tr] where n is grid resolution, nt is timepoints, tr is number of trials
#For our purposes, we can combint nt and tr
n = w.shape[-1]
w = torch.reshape( w, [-1,n,n] )

batch = 12

#For the equivariant autoencoder, I am also going to pass the forcing 4 cos (4y)
force = 4*torch.cos(4*y)
force = force.unsqueeze(0) #add batch index
force = force.unsqueeze(1) #add channel index
force = force.repeat((batch,1,1,1)) #repeat over batch dimension

w_batch = w[0:batch, :, :]
w_batch = w_batch.unsqueeze(1)

print( w_batch.shape )
print( force.shape   )

input = torch.cat( (w_batch, force), dim=1 )
input = input.to(device)


##############################
# Define autoencoder
##############################
lc = 10 #Number of latent images
ch =  4
enc_res = [ 64, 32, 16,  8,  4,  2 ] # encoder resolution sequence
enc_c   = [  2, ch, ch, ch, ch, ch ] # output conv channels
dec_res = [  2,  4,  8, 16, 32, 64 ] # decoder resolution sequence
dec_c   = [ lc, ch, ch, ch, ch, ch ] # output conv channels 

network = EquivariantAutoencoder( lc, enc_res, dec_res, enc_c, dec_c )
#network.load_state_dict(torch.load("gpu_equivariant_autoencoder.pth", map_location=torch.device('cpu')))
network = network.to(device)

symm = SymmetryFactory()

###########################################################
# Check that at all levels, the mean of each field is zero
###########################################################

latent, _ = network.encode(input)
output, _ = network.decode(latent)

def check_mean( field, name ):
    mean  = torch.mean( field, dim=[1,2] )
    bound = torch.max(torch.abs(mean))
    print(f"The mean of {name} is bounded by {bound}")

print("Checking that mean vorticity is zero in all channels")
check_mean( input,  "input"  )
check_mean( latent, "latent" )
check_mean( output, "output" )
print("\n\n")


###################################
# First test: 90 degree rotations
###################################

print("\nChecking for equivariance of 90 degree rotations")

output1 = network.decode(network.encode(symm.rot90(input)))
output2 = symm.rot90(network.decode(network.encode(input)))

mean1 = torch.mean( torch.abs(output1), dim=[0,2,3] )
mean2 = torch.mean( torch.abs(output2), dim=[0,2,3] )

diff90 = torch.mean( torch.abs(output1 - output2), dim=[0,2,3] )

print( f"Difference is {diff90.cpu().detach().numpy()[0]}")

output1 = output1.cpu().detach()
output2 = output2.cpu().detach()


my_dict = {"o1": output1, "o2": output2}
savemat( "diff_rot.mat", my_dict )



############################
# Translation Equivariance
############################
print("\nChecking for equivariance of translations")

#since we go from 64 -> 8 grid spacing, roll 8 times further when applied to input
shift = 16

input_sh  = torch.roll(input, shift, dims=[2])
output1   = network.decode(network.encode(input_sh))
output2   = torch.roll(network.decode(network.encode(input)), shift, dims=[2])

output1 = output1.cpu().detach()
output2 = output2.cpu().detach()
my_dict = {"o1": output1, "o2": output2}
savemat( "diff_trans.mat", my_dict )


diff_sh = torch.mean( torch.abs(output1 - output2), dim=[0,1,2,3] )
print( f"Difference is {diff_sh}")



###########################
# Relfection symmetry!
###########################
print("\nChecking for equivariance of reflections")

#Don;t forget to change sign after transpose!!!
input_tr  = -symm.transpose(input)
output1   = network.decode(network.encode(input_tr))
output2   = -symm.transpose(network.decode(network.encode(input)))

output1 = output1.cpu().detach()
output2 = output2.cpu().detach()
my_dict = {"o1": output1, "o2": output2}
savemat( "diff_reflect.mat", my_dict )

diff_tr = torch.mean( torch.abs(output1 - output2), dim=[0,1,2,3] )
print( f"Difference is {diff_tr}")


###########################
# Relfection symmetry (WRONG)
###########################
print("\nChecking for equivariance of unphysical reflections")

#Purposely forget to change sign after transpose!!!
input_tr  = symm.transpose(input)
output1   = network.decode(network.encode(input_tr))
output2   = symm.transpose(network.decode(network.encode(input)))

output1 = output1.cpu().detach()
output2 = output2.cpu().detach()
my_dict = {"o1": output1, "o2": output2}
savemat( "diff_reflect2.mat", my_dict )

diff_tr = torch.mean( torch.abs(output1 - output2), dim=[0,1,2,3] )
print( f"Difference is {diff_tr}")