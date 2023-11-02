import tensorflow as tf

import numpy as np

class Weak_PINN():
    """
    Build a physics informed neural network (PINN) model that evaluates the equation loss in weak form
    Attributes:
        network: keras network model with input (t) and output [x(t), y(t), z(t), T].
    """

    def __init__(self, network, p):
        """
        Args:
            network: keras network model with input (t) and output [x(t), y(t), z(t), T].
        """

        self.network = network

        #Legendre-Gauss quadrature information
        self.p = p #number of roots to use
        self.points, self.weights = np.polynomial.legendre.leggauss(self.p)
        self.points  = np.array(self.points, dtype='float32')
        self.weights = np.array(self.weights, dtype='float32')
    
        H = np.pi/16.0
        self.H = np.array( [H], dtype='float32') #sidelengths of boxes for weak formulation




    def build(self):
        """
        Build a PINN model

        Returns:
            PINN model for Kolmogorov flow with
                input: [zz = t] at risk of overloading the definition of z
                output: [ e1,e2,e3] the three equations of the Lorenz equtions
        """

        #add a trainable period T for the periodic solution
        self.T = tf.Variable( 4.0, trainable=True)

        # equation input: z = (x,y,t)
        # these will be at the center of sampled domains
        zz= tf.keras.layers.Input(shape=(1,))

        #Let zs be the quadrature points for all sampled subdomains
        #We will need three new axis for identifying quadrature points
        zs = zz[tf.newaxis, ...]

        #rescale quadrature points from [-1,1] to physical units
        displacement = self.H[0]/2*self.points[:,tf.newaxis,tf.newaxis]
        zs = zs + displacement

        #reshape the sample points to trick the network
        zs2 = tf.reshape( zs, [-1,1] )

        #compute network output at all quadrature points
        f = self.network(zs2) 
        f = tf.reshape( f, [self.p,-1,4] ) #4 outputs

        print( np.shape(f) )

        #interpret outputs
        x = f[...,0] #first  spatial coordinate as predicted by network
        y = f[...,1] #second spatial coordinate as predicted by network
        z = f[...,2] #third  spatial coordinate as predicted by network
        T = f[...,3] #period

        # rescalings for derivatives based on affine transformation to canonical [-1,1]
        rt = 2.0/self.H[0] * (np.pi*2)/self.T #for time, also incoorporate the rescaling [0,T] -> [0,2*pi]

        #define 1d weight and its derivatives
        #Let's just use  (1-t^2) since we only need first derivatives for this problem
        phi1  = 1 - self.points**2
        dphi1 = -2*self.points

        #Add a new axis so we can multiply our data
        phi    =  phi1[:, np.newaxis]
        phi_dt = dphi1[:, np.newaxis] 
        weight = self.weights[:,np.newaxis]


        #Lorenz parameters
        sigma = 10.0
        rho   = 28.0
        beta  = 8.0/3

        #equation 1: x dynamics
        eq1= -rt * phi_dt * x - sigma*(y-x)*phi
        e1 = tf.reduce_sum( weight * eq1, axis=[0] )
       
        #equation 2: y dynamics
        eq2 = -rt * phi_dt * y - phi*x*(rho-z) + phi*y
        e2  = tf.reduce_sum( weight * eq2, axis=[0] )
        
        #equation 3: z dynamics
        eq3 = -rt * phi_dt * z - phi*x*y + phi*beta*z
        e3  = tf.reduce_sum( weight * eq3, axis=[0] )
        
        #compute velocities to bias away from equilibria
        vx = tf.reduce_sum( weight * phi_dt * rt * x, axis=[0] )
        vy = tf.reduce_sum( weight * phi_dt * rt * y, axis=[0] )
        vz = tf.reduce_sum( weight * phi_dt * rt * z, axis=[0] )

        
        e1 = tf.expand_dims(e1, axis=1)
        e2 = tf.expand_dims(e2, axis=1)
        e3 = tf.expand_dims(e3, axis=1)
        e  = tf.concat((e1, e2, e3), axis=1)

        #Add a singularity to the loss for equilbiria
        equil1 = np.array( [0.0, 0.0, 0.0] )
        temp   = np.sqrt( beta*(rho-1) )
        equil2 = np.array( [ temp,  temp, rho-1] )
        equil3 = np.array( [-temp, -temp, rho-1] )
        
        power = 4
        V0 = 100.0 #amplitude of singularity
        reg = 1.0 \
            + V0/tf.reduce_sum((f[...,0:3] - equil1[tf.newaxis,:])**power, axis=[1]) \
            + V0/tf.reduce_sum((f[...,0:3] - equil2[tf.newaxis,:])**power, axis=[1]) \
            + V0/tf.reduce_sum((f[...,0:3] - equil3[tf.newaxis,:])**power, axis=[1])
        reg = reg[:,tf.newaxis]
        
        e = e*reg        


        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[zz], outputs=[e])