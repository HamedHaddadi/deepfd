
# ############################ #
# sanity check models          #
# all models are established   #
# and have known results       #
# ############################ #
import tensorflow as tf 
from tensorflow.keras import Model
from ..nn.architectures import DenseCoder 

# #### Dense Autoencoder #### #
class DenseAutoEncoder(Model):
    """
    Autoencoder benchmark model using dense layer
    input_features = output_features is the number of features in the dataset
    example: 784 for a flatten 28x28 MNIST dataset
    """
    def __init__(self, features = None, start_features = None, change_factor = None, 
            number_of_levels = None, weight_init = 'glorot_normal', bias_init = 'zeros', 
                batch_normal = False, activation = 'relu', dropout = 0.0, input_features = None):

        super(DenseAutoEncoder, self).__init__()

        coder_inputs = {key:value for key,value in locals().items() if key not in ['__class__', 'self', 'input_features']}
        self.encoder = DenseCoder(**coder_inputs) 

        coder_inputs['features'] = {True: input_features,
                 False: self.encoder.features[-2::-1] + [input_features]}[len(self.encoder.features) == 1]
        self.decoder =  DenseCoder(**coder_inputs)

    def call(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y 
