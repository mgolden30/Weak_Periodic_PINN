import torch
import torch.nn as nn


device = "cuda"

class SymmetryFactory():
    '''
    This class just provides functions for augmenting data via symmetry operations.
    We can use this to check if a network is actually equivariant
    '''

    def __init__(self):
        super().__init__()

    def rot90_kernel(self, tensor):
        '''
        Perform a 90 degree rotation for a doubly periodic domain around the upper left corner [0,0]
        This is the required rotations for kernels in my architecture
        '''
        c1 = 0
        c2 = 0
        needs_reshaping = False
        if len(tensor.shape) > 4:
            #You are of size [b,c1,c2,n,n] likely
            c1 = tensor.shape[1]
            c2 = tensor.shape[2]
            tensor = torch.flatten(tensor,1,2)
            needs_reshaping = True

        # at this point should be of shape [b,c,n,n]
        # a 90 degree rotation is the combination of two things:
        # 1. matrix transpose
        # 2. reverse the order of all columns (except the first) 
        #This form of rotations leaves tensor[:,:,0,0] as a fixed point
        tensor = torch.transpose(tensor, 2, 3)

        #split tensor into first column and the rest
        first_column = tensor[:,:,:,:1]
        rest_columns = tensor[:,:,:,1:]
        #reverse the ordering of rest_columns
        reversed_columns = torch.flip(rest_columns, dims=[3])
        #stack them again
        tensor = torch.cat( (first_column, reversed_columns), dim=3 )

        if needs_reshaping:
            tensor = torch.unflatten(tensor, 1, (c1,c2) )

        return tensor


    def rot90(self, tensor):
        '''
        PURPOSE:
        This applies a 90 degree rotation to a REAL tensor of size [b,c,n,n].
        THIS IS NOT FOR KERNELS
        Because of the way points are stored, we will need to circularly pad before
        '''

        c1 = 0
        c2 = 0
        needs_reshaping = False
        if len(tensor.shape) > 4:
            #You are of size [b,c1,c2,n,n] likely
            c1 = tensor.shape[1]
            c2 = tensor.shape[2]
            tensor = torch.flatten(tensor,1,2)
            needs_reshaping = True

        #define a function for padding to the right and below
        extend = nn.CircularPad2d((0,1,0,1))
        n = tensor.shape[2]

        #pad
        tensor = extend(tensor)
        
        #rotate
        tensor = torch.rot90( tensor, k=1, dims=(2,3) )
        
        #unpad
        tensor = tensor[:,:,:n,:n]

        if needs_reshaping:
            tensor = torch.unflatten(tensor, 1, (c1,c2) )

        return tensor
    
    def transpose(self, tensor):
        c1 = 0
        c2 = 0
        needs_reshaping = False
        if len(tensor.shape) > 4:
            #You are of size [b,c1,c2,n,n] likely
            c1 = tensor.shape[1]
            c2 = tensor.shape[2]
            tensor = torch.flatten(tensor,1,2)
            needs_reshaping = True

        tensor = torch.transpose(tensor, 2, 3)
        
        if needs_reshaping:
            tensor = torch.unflatten(tensor, 1, (c1,c2) )
        
        return tensor