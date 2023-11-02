import tensorflow as tf
from .layer import PeriodicLayer, ExtraVariableLayer

seed = 1
tf.random.set_seed(seed)  # Set seed for TensorFlow


class Network:
    """
    Build a physics informed neural network (PINN) model for Burgers' equation.
    """

    @classmethod
    def build(cls, num_inputs=1, layers=[8, 16], activation='tanh', num_outputs=4):
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

        seed = 1
        tf.random.set_seed(seed)  # Set seed for TensorFlow
        
        #hidden layers
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                kernel_initializer='he_normal')(x)

        # output layer
        outputs = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='he_normal')(x)

        #rescale the outputs to be O(30) since we know periodic orbits have this size
        outputs = 20*outputs

        # append a trainable period to the output
        extra_layer = ExtraVariableLayer(initial_value=5)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
