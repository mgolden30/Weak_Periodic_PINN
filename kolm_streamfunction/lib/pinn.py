import tensorflow as tf
from .layer import GradientLayer

import numpy as np

class PINN:
    """
    Build a physics informed neural network (PINN) model for Burgers' equation.

    Attributes:
        network: keras network model with input (t, x) and output u(t, x).
        nu: kinematic viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, nu):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
            nu: kinematic viscosity.
        """

        self.network = network
        self.nu = nu
        self.grads = GradientLayer(self.network)

        #add a trainable period
        self.T = 10.0
        self.T = tf.Variable(self.T)

    def build(self):
        """
        Build a PINN model for Burgers' equation.

        Returns:
            PINN model for Kolmogorov flow with
                input: [z = (x,y,t) ],
                output: [ e1,e2,e3] the three equations of the velocity-vorticity formulation
        """

        # equation input: z = (x,y,t)
        z = tf.keras.layers.Input(shape=(3,))

        # compute gradients
        dw_dt, advec, lap_w, curl, w, div = self.grads(z)

        #compute the forcing. For Kolmogorov we will use forcing = 4*cos(4*y)
        forcing = 4 * tf.cos( 4 * z[...,1] )
        
        # equation output being zero
        rescale = 2*np.pi/self.T
        e1 = rescale * dw_dt + advec - self.nu*lap_w - forcing #vorticity dynamics
        e2 = curl - w #definition of vorticity as curl of the flow
        e3 = div #incompressibility means div(u) = 0

        print(e1.get_shape() ) 
        print(e2.get_shape() ) 
        print(e3.get_shape() ) 
        
        e1 = tf.expand_dims(e1, axis=1)
        e2 = tf.expand_dims(e2, axis=1)
        e3 = tf.expand_dims(e3, axis=1)
        e = tf.concat((e1, e2, e3), axis=1)

        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[z], outputs=[e])