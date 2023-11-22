import tensorflow as tf

import numpy as np

from .network import StreamfunctionNetwork

class Weak_PINN():
    """
    Build a physics informed neural network (PINN) model that evaluates the equation loss in weak form
    Attributes:
        network: keras network model with input (t, x) and output u(t, x).
        nu: kinematic viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, st_network, nu, p):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
            nu: kinematic viscosity.
        """

        #subnetwork for streamfunction and T,a
        self.st_network = st_network

        #full hydro network that outputs hydrodynamic fields (w,u,v,T,a)
        self.network = network
        
        #kinematic viscosity
        self.nu = nu

        #Legendre-Gauss quadrature information
        self.p = p #number of roots to use
        self.points, self.weights = np.polynomial.legendre.leggauss(self.p)
        self.points  = np.array(self.points, dtype='float32')
        self.weights = np.array(self.weights, dtype='float32')

        D = 2*np.pi #domain size
        H = D/8
        self.H = np.array( [H, H, H], dtype='float32') #sidelengths of boxes for weak formulation




    def build(self):
        """
        Build a PINN model for Burgers' equation.

        Returns:
            PINN model for Kolmogorov flow with
                input: [z = (x,y,t) ],
                network_output: (w,u,v,T)
                output: [ e1,e2,e3] the three equations of the velocity-vorticity formulation
        """

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
        f = tf.reshape( f, [p,p,p,-1,5] )

        #pull out the hydrodynamic fields
        w = f[...,0] #vorticity
        u = f[...,1] #x-component of the flow
        v = f[...,2] #y-component of flow
        T = f[...,3] #period
        a = f[...,4] #drift rate

        #Compute the forcing
        forcing = 4 * tf.sin( 4 * zs[...,1] )

        # rescalings for derivatives based on affine transformation to canonical [-1,1]
        rx = 2.0/self.H[0]
        ry = 2.0/self.H[1]
        rt = 2.0/self.H[2]
        rt2 = (2*np.pi)/T

        #define 1d weight and its derivatives
        #since I am no longer enforcing the nonlinear equaitons, I only need first derivaives!
        phi1   = 1 - 2*self.points**2 + self.points**4
        dphi1  = -4*self.points + 4*self.points**3
        ddphi1 = -4 + 12 * self.points**2

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
             - rx * phi_dx * ((u+a) * w) \
             - ry * phi_dy * (v * w) \
             - self.nu * ( rx*rx*phi_ddx + ry*ry*phi_ddy)*w \
             - phi * forcing
        e1 = tf.reduce_sum( weight * eq1, axis=[0,1,2] )

        #estimate magnitude of dynamics
        #who cares about rescaling time
        dw_dt = tf.reduce_sum( weight * phi_dt * w, axis=[0,1,2] )
        #u_int = tf.reduce_sum( weight * phi    * u, axis=[0,1,2] )

        e1 = tf.expand_dims(e1, axis=1)
        #e2 = tf.expand_dims(e2, axis=1)
        #e3 = tf.expand_dims(e3, axis=1)
        e  = tf.concat((e1), axis=1)

        #come up with a regularizer to discourage trivial solutions
        reg = 1.0 + 1.0/(tf.abs(dw_dt)**2)
        e = e*reg

        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[z], outputs=[e1])
    
    def least_squares_fit(self, z):
        # This method is being written out of frustration at optimization methods.
        # The weights of the network and the parameters (T,a) should not be on the same footing.
        # Since the NS equation is linear in 1/T and a, we can solve directly for a least-squares solution to minimize the error.

        # equation input: z = (x,y,t)
        # these will be at the center of sampled domains
        # z = tf.keras.layers.Input(shape=(3,))

        #Let zs be the quadrature points for all sampled subdomains
        #We will need three new axis for identifying quadrature points
        zs = z[tf.newaxis, tf.newaxis, tf.newaxis, ...]

        #keep these in the range [-1,1]
        #Note this is functionally the same as Matlab's meshgrid
        #This means to accomplish ndgrid output, we need to swap x and y in output
        y, x, t = tf.meshgrid( self.points, self.points, self.points )

        displacement = tf.concat( ( self.H[0]/2*x[:,:,:,tf.newaxis,tf.newaxis],\
                                    self.H[1]/2*y[:,:,:,tf.newaxis,tf.newaxis],\
                                    self.H[2]/2*t[:,:,:,tf.newaxis,tf.newaxis]), axis=4 )
        zs = zs + displacement

        #reshape the sample points to trick the network
        zs2 = tf.reshape( zs, [-1,3] )

        #compute fields f = [w,u,v] at all quadrature points
        f = self.network(zs2) 

        p = self.p
        f = tf.reshape( f, [p,p,p,-1,5] )

        #pull out the hydrodynamic fields
        w = f[...,0] #vorticity
        u = f[...,1] #x-component of the flow
        v = f[...,2] #y-component of flow
        T = f[...,3] #period
        a = f[...,4] #drift rate

        #Compute the forcing
        forcing = 4 * tf.sin( 4 * zs[...,1] )

        # rescalings for derivatives based on affine transformation to canonical [-1,1]
        rx = 2.0/self.H[0]
        ry = 2.0/self.H[1]
        rt = 2.0/self.H[2]
        rt2 = (2*np.pi)/T

        #define 1d weight and its derivatives
        #since I am no longer enforcing the nonlinear equaitons, I only need first derivaives!
        phi1   = 1 - 2*self.points**2 + self.points**4
        dphi1  = -4*self.points + 4*self.points**3
        ddphi1 = -4 + 12 * self.points**2

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

        '''
        #equation 1: voriticity dynamics
        eq1= - rt2 * rt * phi_dt * w \
             - rx * phi_dx * ((u+a) * w) \
             - ry * phi_dy * (v * w) \
             - self.nu * ( rx*rx*phi_ddx + ry*ry*phi_ddy)*w \
             - phi * forcing
        e1 = tf.reduce_sum( weight * eq1, axis=[0,1,2] )
        '''

        # Set up the linear system to solve for the physical parameters
        rhs  = rx*phi_dx*u*w + ry*phi_dy*v*w + phi*forcing # rhs is right hand side we are trying to fit
        col1 = -rt * phi_dt * w  #weighted by rt2
        col2 = -rx * phi_dx * w #weighted by drift parameter a
        
        b    = tf.reduce_sum( weight * rhs,  axis=[0,1,2] )
        col1 = tf.reduce_sum( weight * col1, axis=[0,1,2] )
        col2 = tf.reduce_sum( weight * col2, axis=[0,1,2] )

        A = tf.concat( (col1[:, tf.newaxis], col2[:, tf.newaxis]), axis=1 )

        '''
        print( tf.shape(A) )
        print( tf.shape(col1) )
        print( tf.shape(b) )

        print(A)
        print(b)
        '''

        #solve the least squares system
        params = tf.linalg.lstsq( A, b[:,tf.newaxis] )

        #convert to ordinary tensor
        #params = tf.convert_to_tensor(params)
        params = np.array(params)

        print(params)
        print( 2*np.pi/params[0] )

        # params now contains the best fit values [2*pi/T, a]
        params[0] = 2*np.pi/params[0]

        #Hard cap the period to something reasonable
        T_max = 40
        if( tf.abs(params[0]) > T_max):
            params[0] = T_max

        #use these to override the extra variable outputs of the StreamfunctionNetwork
        StreamfunctionNetwork.change_extra_variable_values(self.st_network, params )

        return 