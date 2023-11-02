import lib.tf_silent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from lib.weak_pinn import Weak_PINN
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
    num_train  =  1024   # randomly sample this many points for training
    num_test   =   256   # make a grid for testing with this many points
    num_epochs = 50000  # Adjust the number of epochs as needed
    lr = 1e-5 #learning rate

    p = 10 #number of points for Gaussian quadrature

    # build a core network model
    network = Network.build()
    network.summary()

    # build a PINN model
    pinn = Weak_PINN(network,p).build()

    seed = 1
    tf.random.set_seed(seed)  # Set seed for TensorFlow
    np.random.seed(seed)      # Set seed for NumPy

    # create training input
    z_train = 2 * np.pi * np.random.rand(num_train, 1) # z = (x,y,t) all random numbers from [0,2*pi]

    # create training output
    # all equations should be equal to zero
    e_train = np.zeros((num_train, 3)) 

    #evaluate the training data just to make sure the network is set up correctly
    output = network.predict( z_train )

    # Assuming you have your training data available in z_train and e_train
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr)  # You can adjust the learning rate if needed
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # Use Adam optimizer
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    #store loss vs epoch
    loss_history = np.zeros( num_epochs )

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
        loss_history[epoch] = loss.numpy()
        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')

    # predict x(t), y(t), z(t) and T
    t = np.linspace(0, 2*np.pi, num_test)
    t = t[:, np.newaxis]

    f = network.predict( t, batch_size=num_test )
    
    #separate outputs
    x = f[...,0]
    y = f[...,1]
    z = f[...,2]
    T = f[...,3]

    print('Period T = %f' % (T[0]) )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lorenz')
        
    # Save the figure as an image
    plt.savefig( '%s/lorenz.png' % (output_directory) )


    plt.figure()
    steps = np.linspace(1, num_epochs, num_epochs)
    plt.plot( steps, loss_history )
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig( '%s/loss.png' % (output_directory) )
