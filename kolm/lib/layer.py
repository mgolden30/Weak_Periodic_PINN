import tensorflow as tf


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
                        #tf.sin(2*x), tf.cos(2*x)
                        ], 1 )
        
        return y
    

class ExtraVariableLayer(tf.keras.layers.Layer):
    def __init__(self, initial_value=6.0, **kwargs):
        self.initial_value = initial_value
        super(ExtraVariableLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add a trainable variable with initial value
        self.extra_variable = self.add_weight(shape=(1,),
                                             initializer=tf.keras.initializers.Constant(self.initial_value),
                                             trainable=True)
        super(ExtraVariableLayer, self).build(input_shape)

    def call(self, inputs):
        # Append the extra variable to the output
        return tf.concat([inputs, self.extra_variable * tf.ones((tf.shape(inputs)[0], 1))], axis=1)