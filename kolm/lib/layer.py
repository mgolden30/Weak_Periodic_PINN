import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for Burgers' equation.

    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """

        self.model = model
        super().__init__(**kwargs)

    def call(self, x):
        """
        Computing 1st and 2nd derivatives for Burgers' equation.

        Args:
            x: input variable.

        Returns:
            model output, 1st and 2nd derivatives.
        """

        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                f = self.model(x) #compute fields f = [w,u,v]
            df = gg.batch_jacobian(f, x) #compute jacobian
            #spacetime derivatives of omega
            dw_dx = df[..., 0, 0]
            dw_dy = df[..., 0, 1]
            dw_dt = df[..., 0, 2]            
            #spatial derivatives of u
            du_dx = df[..., 1, 0]
            du_dy = df[..., 1, 1]
            #spatial derivatives of v
            dv_dx = df[..., 2, 0]
            dv_dy = df[..., 2, 1]
        ddf = g.batch_jacobian(df, x)
        
        #for debugging
        #print( ddf.get_shape() )
 
        dw_dx2 = ddf[...,0,0,0]
        dw_dy2 = ddf[...,0,1,1]
        
        w = f[..., 0]
        u = f[..., 1]
        v = f[..., 2]

        lap_w = dw_dx2 + dw_dy2   #laplacian of omega
        advec = u*dw_dx + v*dw_dy #advection of vorticity
        div   = du_dx + dv_dy     #divergence of u
        curl  = dv_dx - du_dy     #curl of velocity

        return dw_dt, advec, lap_w, curl, w, div




class PeriodicLayer(tf.keras.layers.Layer):
    """
    Custom layer to project into period domain (both in space and in time)

    Attributes:
        model: keras network model.
    """

    def __init__(self, **kwargs):
        """
        Args:
            model: keras network model.
        """

        super().__init__(**kwargs)

    def call(self, x):
        """

        Args:
            x: input variable.

        Returns:
            model output, 1st and 2nd derivatives.
        """

        #define periodic coordinates
        y = tf.concat( [tf.sin(x), tf.cos(x)], 1 )
        
        return y