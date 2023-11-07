import tensorflow as tf
from .layer import PeriodicLayer, ExtraVariableLayer


#for hidden layer
def gaussian_activation(x):
    return tf.exp(-tf.square(x))


class Network:
    """
    Build a physics informed neural network (PINN) model for Burgers' equation.
    """

    @classmethod
    def build(cls, num_inputs=3, layers=[1024], activation=gaussian_activation, num_outputs=3):
        """
        Build a PINN model for Burgers' equation with input shape (t, x) and output shape u(t, x).

        Args:
            num_inputs: number of input variables. Default is 2 for (t, x).
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables. Default is 1 for u(t, x).

        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        
        x = inputs

        #Apply periodic layer
        x = PeriodicLayer()(x)

        #hidden layers
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                kernel_initializer='he_normal')(x)
            
        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='he_normal')(x)

        #append an extra variable (period T) to output
        outputs = ExtraVariableLayer()(outputs)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
