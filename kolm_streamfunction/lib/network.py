import tensorflow as tf
from .layer import PeriodicLayer, ExtraVariableLayer, StreamfunctionGradientLayer
import numpy as np

#for hidden layer
def gaussian_activation(x):
    return tf.exp(-tf.square(x))


class StreamfunctionNetwork:
    """
    This network learns a streamfunction psi as a function of spacetime (x,y,t).
    
    This network learns [psi, T, a]
    psi(x,y,t) - the value of the streamfunction at this point of spacetime. We can autodiff to obtian the velocity components and the vorticity.
    T - period of solution
    a - drift rate in x.
    """

    @classmethod
    def build(cls, num_inputs=3, layers=[32, 32], activation=gaussian_activation, num_outputs=1):
        """
        See class doc

        Args:
            num_inputs: number of input variables.
            layers: number of hidden layers for the periodic part of A_i
            layers_1D: layer structure for the 1D corrections to A_i which have linear growth.
            activation: activation function in hidden layers.
            num_outpus: number of output variables before appending T.

        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        
        x = inputs

        #Apply periodic layer to enforce boundary conditions
        per = PeriodicLayer()
        x = per(x)

        #hidden layers
        for layer in layers:
            x = tf.keras.layers.Dense(layer, activation=activation,
                kernel_initializer='he_normal')(x)
            
        # output layer of the main path of network for computing periodic one-form A_i
        psi = tf.keras.layers.Dense(num_outputs,
            kernel_initializer='he_normal')(x)
        
        #append an extra variable (period T) to output
        outputs = ExtraVariableLayer( 20.0, "extra_variable_layer")(psi)

        #again for a drift rate
        outputs = ExtraVariableLayer( 0.0, "extra_variable_layer_1")(outputs)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    
    def change_extra_variable_values(self, new_values):
        """
        Change the values of the ExtraVariableLayer layers.

        Args:
            new_values: List of new values for the ExtraVariableLayer layers.
                        The length of this list should match the number of ExtraVariableLayer layers in the network.
        """
        if len(new_values) != 2:  # Assuming there are two ExtraVariableLayer layers in your network
            raise ValueError("The length of new_values should match the number of ExtraVariableLayer layers.")
        
        # Iterate through layers and update weights
        for i in range(len(new_values)):
            layer_name = f"extra_variable_layer_{i}" if i > 0 else "extra_variable_layer"
            layer = self.get_layer(layer_name)
            layer.set_weights([tf.constant([new_values[i]], dtype=tf.float32)[0]])

        print("ExtraVariableLayer values updated.")

class HydroNetwork:
    """
    This network learns hydrodynamic fields (w, u, v) as a function of spacetime (x,y,t).
    
    The total output is then [w, u, v, T, a], where T and a are independent of the network inputs.
    """

    @classmethod
    def build(cls, model, num_inputs=3):
        """
        See class doc

        Args:
            num_inputs: number of input variables.
            layers: number of hidden layers.
            activation: activation function in hidden layers.
            num_outpus: number of output variables before appending T.

        Returns:
            keras network model
        """

        # input layer
        inputs = tf.keras.layers.Input(shape=(num_inputs,), dtype=tf.float32)

        #Now we can autodiff and compute hydrodynamic fields [w, u, v, T] that EXACTLY satisfy the nonlinear vorticity equation
        outputs = StreamfunctionGradientLayer(model)(inputs)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)