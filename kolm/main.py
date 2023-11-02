import lib.tf_silent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
import tensorflow as tf
from PIL import Image
import os


# Define a directory to save the frames
# Get the absolute path of the current working directory
current_directory = os.getcwd()
# Define the output directory using an absolute path
output_directory = os.path.join(current_directory, 'frames/')

if __name__ == '__main__':
    """
    Modify a PINN model for 2+1D forced incompressible Navier-Stokes
    """

    #number of points to train/test on
    num_train =   100 #randomly sample this many points for training
    num_test  =    64 #make a grid for testing with this many points
    num_epochs=  1000 # Adjust the number of epochs as needed
    lr = 1e-2 #learning rate


    # kinematic viscosity of the fluid flow
    nu =  1.0 / 40

    # build a core network model
    network = Network.build()
    network.summary()

    # build a PINN model
    pinn = PINN(network, nu).build()

    seed = 32
    tf.random.set_seed(seed)  # Set seed for TensorFlow
    np.random.seed(seed)      # Set seed for NumPy

    # create training input
    z_train = 2 * np.pi * np.random.rand(num_train, 3) # z = (x,y,t) all random numbers from [0,2*pi]

    # create training output
    # all equations should be equal to zero
    e_train = np.zeros((num_train, 3)) 

    #evaluate the training data just to make sure the network is set up correctly
    output = network.predict( z_train )

    # train the model using L-BFGS-B algorithm
    #lbfgs = L_BFGS_B(model=pinn, x_train=z_train, y_train=[e_train] )
    #lbfgs.fit()

    # Assuming you have your training data available in z_train and e_train

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)  # You can adjust the learning rate if needed
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(z, e):
        with tf.GradientTape() as tape:
            predictions = pinn(z)
            loss = loss_fn(e, predictions)
        gradients = tape.gradient(loss, pinn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn.trainable_variables))
        return loss

    # Training loop
    for epoch in range(num_epochs):
        #z_train = 2 * np.pi * np.random.rand(num_train, 3) # z = (x,y,t) all random numbers from [0,2*pi]
        loss = train_step(z_train, e_train)
        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')


    # predict vorticity w(x,y,t)
    x_flat = np.linspace(0, 2*np.pi, num_test)
    y_flat = np.linspace(0, 2*np.pi, num_test)
    t_flat = np.linspace(0, 2*np.pi, num_test)
    
    x, y, t = np.meshgrid(x_flat, y_flat, t_flat)
    z_test  = np.stack([x.flatten(), y.flatten(), t.flatten()], axis=-1)
    
    f = network.predict( z_test, batch_size=num_test )
    
    #separate out and reshape the fluid fields
    w = f[...,0].reshape(t.shape)
    u = f[...,1].reshape(t.shape)
    v = f[...,2].reshape(t.shape)

    # Assuming w is your rank 3 tensor with shape (num_x, num_y, num_t)
    # num_x, num_y, num_t are the dimensions along x, y, and t respectively


    # Create the directory if it doesn't exist
    for i in range(num_test):
        print(i)
        plt.clf()
        plt.imshow(w[:, :, i], cmap='bwr')
        plt.colorbar(label='w')
        plt.clim(-1,1)
        plt.title(f'Time Step {i}')
        plt.xlabel('y')
        plt.ylabel('x')
        
        # Save the figure as an image
        plt.savefig( '%s/frame_%04d.png' % (output_directory, i) )