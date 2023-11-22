import tensorflow as tf
import numpy as np

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
        y = tf.concat( [tf.sin(  x), tf.cos(  x), \
                        #tf.sin(2*x), tf.cos(2*x), \
                        #tf.sin(3*x), tf.cos(3*x), \
                        #tf.sin(4*x), tf.cos(4*x), \
                        ], 1 )
        
        return y
    

class ExtraVariableLayer(tf.keras.layers.Layer):
    def __init__(self, initial_value, name, **kwargs):
        self.initial_value = initial_value
        super(ExtraVariableLayer, self).__init__(**kwargs)
        self._name = name

    def build(self, input_shape):
        # Add a trainable variable with initial value
        self.extra_variable = self.add_weight(shape=(1,),
                                             initializer=tf.keras.initializers.Constant(self.initial_value),
                                             trainable=True)
        super(ExtraVariableLayer, self).build(input_shape)

    def call(self, inputs):
        # Append the extra variable to the output
        return tf.concat([inputs, self.extra_variable * tf.ones((tf.shape(inputs)[0], 1))], axis=1)
    

class StreamfunctionGradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives of network output with respect to input

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
            model output, desired 1st and 2nd derivatives.
        """

        with tf.GradientTape() as g:
            g.watch(x)
            with tf.GradientTape() as gg:
                gg.watch(x)
                f = self.model(x) #f = [psi, T] for this case
            df = gg.batch_jacobian(f, x) #compute jacobian

            #compute x and y components of the flow as gradients of the streamfunction
            u = -df[..., 0, 1] # u = - \partial_y \psi
            v =  df[..., 0, 0] # v =   \partial_x \psi
        ddf = g.batch_jacobian(df, x) 
        w = -(ddf[...,0,0,0] + ddf[...,0,1,1]) #omega is negative laplacian of psi
        
        T = f[..., 1]
        a = f[..., 2]

        w = tf.expand_dims(w, axis=1)
        u = tf.expand_dims(u, axis=1)
        v = tf.expand_dims(v, axis=1)
        T = tf.expand_dims(T, axis=1)
        a = tf.expand_dims(a, axis=1)
        
        output = tf.concat((w,u,v,T,a), axis=1)

        return output
