import lib.tf_silent
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from lib.weak_pinn import Weak_PINN
from lib.network import StreamfunctionNetwork, HydroNetwork
import tensorflow as tf
from PIL import Image
import os
from keras.models import load_model
from tensorflow import keras

from scipy.io import savemat

from lib.layer import PeriodicLayer, ExtraVariableLayer


# Define a directory to save the frames
# Get the absolute path of the current working directory
current_directory = os.getcwd()
# Define the output directory using an absolute path
output_directory = os.path.join(current_directory, 'NN_search/')


#for reproducibility, set the keras seed.
seed = 32
keras.utils.set_random_seed(seed)

#if __name__ == '__main__':
num_attempts = 32
for attempt in range( num_attempts ):
    """
    Modify a PINN model for 2+1D forced incompressible Navier-Stokes
    """

    #number of points to train/test on
    num_train =    8**3   # randomly sample this many points for training
    num_test  =    [128, 128, 32]   # unifrom test grid
    num_epochs=    1024 # Adjust the number of epochs as needed

    error_cutoff = 100000000 #save if loss is lower than this

    lr = 1e-2  #learning rate
    p  = 4     #number of points for Gaussian quadrature

    scramble = True
    use_previous_model = False

    # kinematic viscosity of the fluid flow
    nu =  1.0 / 40

    # build a core network model
    tv_network = StreamfunctionNetwork.build()
    tv_network.summary()

    # returns a compiled model
    if use_previous_model:
        #for hidden layer
        def gaussian_activation(x):
            return tf.exp(-tf.square(x))
        
        #The following magic incantation is loading my custom activation functions without errors
        #Do not make significant changes to the next 8 or so lines unless you know what you are doing.
        custom_objects = {
            'PeriodicLayer': PeriodicLayer,
            'ExtraVariableLayer': ExtraVariableLayer,
            'gaussian_activation': gaussian_activation
        }
        config = tv_network.get_config()
        with keras.utils.custom_object_scope(custom_objects):
            tv_network = keras.Model.from_config(config)
            tv_network.load_weights('stream_network.h5')


    #build a hydro network out of the tv_network
    hydro_network = HydroNetwork.build( tv_network )

    # build a PINN model
    pinn_builder = Weak_PINN(hydro_network, tv_network, nu, p)
    pinn = pinn_builder.build() 

    #set the seeds for mesh generation each step.    
    tf.random.set_seed(seed)  # Set seed for TensorFlow
    np.random.seed(seed)      # Set seed for NumPy

    # create training input
    z_train = 2 * np.pi * np.random.rand(num_train, 3) # z = (x,y,t) all random numbers from [0,2*pi]

    # create training output, all equations should be equal to zero
    e_train = np.zeros((num_train, 1)) 

    # Assuming you have your training data available in z_train and e_train
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr) # You can adjust the learning rate if needed
    loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(z, e):
        with tf.GradientTape() as tape:
            predictions = pinn(z)
            loss = loss_fn(e, predictions)
        gradients = tape.gradient(loss, pinn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pinn.trainable_variables))
        return loss

    loss_history = np.zeros( num_epochs )

    # Training loop
    for epoch in range(num_epochs):
        if scramble:
            z_train = 2 * np.pi * np.random.rand(num_train, 3) # z = (x,y,t) all random numbers from [0,2*pi]
        loss = train_step(z_train, e_train)

        #also set the period and drift rate via least-squares
        pinn_builder.least_squares_fit(z_train)

        loss_history[epoch] = loss.numpy()
        print(f'Epoch {epoch+1}, Loss: {loss.numpy()}')


    #Check if it learned reasonably
    if tf.reduce_mean(loss_history[(num_epochs-10):num_epochs]) > error_cutoff :
        #model failed to learn well. Try again next attempt
        continue

    #after training save the network
    tv_network.save( 'NN_search/stream_network_%d.h5' % (attempt) )


    ##############################################
    # Network is done training. Now make plots
    ##############################################
    x_flat  = np.linspace(0, 2*np.pi, num_test[0])
    y_flat  = np.linspace(0, 2*np.pi, num_test[1])
    t_flat  = np.linspace(0, 2*np.pi, num_test[2])
    x, y, t = np.meshgrid(x_flat, y_flat, t_flat)
    z_test  = np.stack([x.flatten(), y.flatten(), t.flatten()], axis=-1) 
    f       = hydro_network.predict( z_test )
    
    #separate out and reshape the fluid fields
    w = f[...,0].reshape(t.shape)
    u = f[...,1].reshape(t.shape)
    v = f[...,2].reshape(t.shape)
    T = f[...,3].reshape(t.shape)
    a = f[...,4].reshape(t.shape)

    #save output to matfiles for analysis
    out_dic = {"w": w, "u": u, "v": v, "T": T, "a": a, "x": x, "y": y, "t": t, "loss_history": loss_history}
    savemat("NN_search/output_%d.mat" % (attempt), out_dic)

    #plot loss history
    plt.clf()
    plt.plot( range(num_epochs), loss_history )       
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('mean abs error')
    plt.savefig( '%s/kolm_loss_%d.png' % (output_directory, attempt) )
