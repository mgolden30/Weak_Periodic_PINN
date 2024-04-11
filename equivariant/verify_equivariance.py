import torch as torch
from lib.equivariant_networks import SymmetryFactory, EquivariantAutoencoder
from scipy.io import loadmat, savemat

# Set PyTorch seed for reproducibility
seed_value = 123
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Load data
data = loadmat("w_traj.mat")
w = torch.tensor(data["w"], dtype=torch.float32)
x = torch.tensor(data["x"], dtype=torch.float32)
y = torch.tensor(data["y"], dtype=torch.float32)

#w is saved as [n,n,nt,tr] where n is grid resolution, nt is timepoints, tr is number of trials
#For our purposes, we can combint nt and tr
n = w.shape[0]
w = torch.reshape( w, [n,n,-1] )

# Permute dimensions for training [n,n,b] -> [b,n,n]
w = w.permute(2, 0, 1)

batch = 12

#For the equivariant autoencoder, I am also going to pass the forcing 4 cos (4y)
force = 4*torch.cos(4*y)
force = force.unsqueeze(0) #add batch index
force = force.unsqueeze(1) #add channel index
force = force.repeat((batch,1,1,1)) #repeat over batch dimension

w_batch = w[0:batch, :, :]
w_batch = w_batch.unsqueeze(1)

input = torch.cat( (w_batch, force), dim=1 )


network = EquivariantAutoencoder()
#network.load_state_dict(torch.load("equivariant_autoencoder.pth"))

symm = SymmetryFactory()


##################################
# Does rotation by 90 degrees four times return the original?
##################################

i1 = symm.rot90_kernel(input)
i2 = symm.rot90_kernel(i1)
i3 = symm.rot90_kernel(i2)
i4 = symm.rot90_kernel(i3)

print("\nChecking properties of rot90_kernel")

print( f"mean error after 1 rotations is {torch.mean(torch.abs(input-i1),dim=[0,1,2,3])}" )
print( f"mean error after 2 rotations is {torch.mean(torch.abs(input-i2),dim=[0,1,2,3])}" )
print( f"mean error after 3 rotations is {torch.mean(torch.abs(input-i3),dim=[0,1,2,3])}" )
print( f"mean error after 4 rotations is {torch.mean(torch.abs(input-i4),dim=[0,1,2,3])}" )

inv = (input + i1 + i2 + i3)/4
inv_rot = symm.rot90_kernel(inv)
print( f"mean of invariant is {torch.mean(torch.abs(inv),dim=[0,1,2,3])}" )
print( f"mean change in invariant {torch.mean(torch.abs(inv-inv_rot),dim=[0,1,2,3])}" )


###################################
# First test: 90 degree rotations
###################################

print("\nChecking for equivariance of 90 degree rotations")

output1 = network.decode(network.encode(symm.rot90(input)))
output2 = symm.rot90(network.decode(network.encode(input)))

print(f"Shape of output is {output1.shape}")

mean1 = torch.mean( torch.abs(output1), dim=[0,2,3] )
mean2 = torch.mean( torch.abs(output2), dim=[0,2,3] )

diff90 = torch.mean( torch.abs(output1 - output2), dim=[0,2,3] )

print(f"mean of network(rot90()) is {mean1}")
print(f"mean of rot90(network()) is {mean2}")
print(f"mean of difference       is {diff90}")

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
print("\nChecking for equivariance of translations")

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