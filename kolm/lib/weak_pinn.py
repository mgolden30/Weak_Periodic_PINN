import tensorflow as tf
from .layer import GradientLayer

import numpy as np

class Weak_PINN():
    """
    Build a physics informed neural network (PINN) model that evaluates the equation loss in weak form
    Attributes:
        network: keras network model with input (t, x) and output u(t, x).
        nu: kinematic viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, nu, p):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
            nu: kinematic viscosity.
        """

        self.network = network
        
        #kinematic viscosity
        self.nu = nu

        #Legendre-Gauss quadrature information
        self.p = p #number of roots to use
        self.points, self.weights = np.polynomial.legendre.leggauss(self.p)
        self.points  = np.array(self.points, dtype='float32')
        self.weights = np.array(self.weights, dtype='float32')
    
        #Weak form parameters
        self.n = 2 #power (1-x^2)^n for weak form

        H = np.pi/4.0
        self.H = np.array( [H, H, H ], dtype='float32') #sidelengths of boxes for weak formulation




    def build(self):
        """
        Build a PINN model for Burgers' equation.

        Returns:
            PINN model for Kolmogorov flow with
                input: [z = (x,y,t) ],
                output: [ e1,e2,e3] the three equations of the velocity-vorticity formulation
        """

        #add a trainable period T for the periodic solution
        self.T = tf.Variable( 20.0, trainable=True)

        # equation input: z = (x,y,t)
        # these will be at the center of sampled domains
        z = tf.keras.layers.Input(shape=(3,))

        #Let zs be the quadrature points for all sampled subdomains
        #We will need three new axis for identifying quadrature points

        zs = z[tf.newaxis, tf.newaxis, tf.newaxis, ...]

        #keep these in the range [-1,1]
        #Note this is functionally the same as Matlab's meshgrid
        #This means to accomplish ndgrid output, we need to swap x and y in output
        y, x, t = tf.meshgrid( self.points, self.points, self.points )
        #print( x[1,0,0] - x[0,0,0] )
        #print( y[0,1,0] - y[0,0,0] )
        #print( t[0,0,1] - t[0,0,0] )
        

        displacement = tf.concat( ( self.H[0]/2*x[:,:,:,tf.newaxis,tf.newaxis],\
                                    self.H[1]/2*y[:,:,:,tf.newaxis,tf.newaxis],\
                                    self.H[2]/2*t[:,:,:,tf.newaxis,tf.newaxis]), axis=4 )
        zs = zs + displacement

        #reshape the sample points to trick the network
        zs2 = tf.reshape( zs, [-1,3] )

        #compute fields f = [w,u,v] at all quadrature points
        f = self.network(zs2) 

        p = self.p
        f = tf.reshape( f, [p,p,p,-1,3] )

        #pull out the hydrodynamic fields
        w = f[...,0]
        u = f[...,1]
        v = f[...,2]
        
        #compute the forcing. For Kolmogorov we will use forcing = 4*cos(4*y)
        forcing = 4 * tf.cos( 4 * zs[...,1] )

        # rescalings for derivatives based on affine transformation to canonical [-1,1]
        rx = 2.0/self.H[0]
        ry = 2.0/self.H[1]
        rt = 2.0/self.H[2]
        rt2 = (np.pi*2)/self.T

        #define 1d weight and its derivatives
        phi1  = (1-self.points**2)**self.n
        dphi1 = -2*self.n*self.points*(1-self.points**2)**(self.n-1)
        ddphi1= ( 4*self.n*(self.n-1)*self.points**2  - 2*self.n*(1-self.points**2))*(1-self.points**2)**(self.n-2)

        #construct 3d weights as products of 1d weights
        phi    = phi1[:, np.newaxis, np.newaxis] * phi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        phi_dx =dphi1[:, np.newaxis, np.newaxis] * phi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        phi_dy = phi1[:, np.newaxis, np.newaxis] *dphi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        phi_dt = phi1[:, np.newaxis, np.newaxis] * phi1[np.newaxis, :, np.newaxis] *dphi1[np.newaxis, np.newaxis, :]
        phi_ddx = ddphi1[:, np.newaxis, np.newaxis] *  phi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        phi_ddy =   phi1[:, np.newaxis, np.newaxis] *ddphi1[np.newaxis, :, np.newaxis] * phi1[np.newaxis, np.newaxis, :]
        
        #make a 3D quadrature weight
        weight = self.weights[:, np.newaxis, np.newaxis] * self.weights[np.newaxis, :, np.newaxis] * self.weights[np.newaxis, np.newaxis, :]
        
        #add another dimension to all weight objects
        phi     = phi[:,:,:,np.newaxis]
        phi_dx  = phi_dx[:,:,:,np.newaxis]
        phi_dy  = phi_dy[:,:,:,np.newaxis]
        phi_dt  = phi_dt[:,:,:,np.newaxis]
        phi_ddx = phi_ddx[:,:,:,np.newaxis]
        phi_ddy = phi_ddy[:,:,:,np.newaxis]  
        weight  = weight[:,:,:,np.newaxis]

        #equation 1: voriticity dynamics
        eq1= - rt2 * rt * phi_dt * w \
             - rx * phi_dx * (u * w) \
             - ry * phi_dy * (v * w) \
             - self.nu * ( rx*rx*phi_ddx + ry*ry*phi_ddy)*w \
             - phi * forcing
        e1 = tf.reduce_sum( weight * eq1, axis=[0,1,2] )
       
        #equation 2: definition of vorticity as curl(u)
        eq2 = -rx * phi_dx * v \
              +ry * phi_dy *u \
              -phi * w
        e2  = tf.reduce_sum( weight * eq2, axis=[0,1,2] )
        
        #equation 3: incompressibility
        eq3 = -rx*phi_dx*u - ry*phi_dy*v 
        e3  = tf.reduce_sum( weight * eq3, axis=[0,1,2] )

        print(e1.get_shape() ) 
        print(e2.get_shape() ) 
        print(e3.get_shape() ) 
        
        e1 = tf.expand_dims(e1, axis=1)
        e2 = tf.expand_dims(e2, axis=1)
        e3 = tf.expand_dims(e3, axis=1)
        e  = tf.concat((e1, e2, e3), axis=1)

        #regularizer to discourage laminar flow
        #laminar = forcing / 16.0 / self.nu
        #reg = 1.0 + 1.0/tf.reduce_sum(tf.abs(w - laminar))
        #reg = 1

        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[z], outputs=[e])

    def print_period(self):
        print( self.T )