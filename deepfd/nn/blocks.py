
# Blocks used in developing models #

import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras import initializers, activations 

activate = {'relu': activations.relu, 
                'sigmoid': activations.sigmoid}
init_variable = {'glorot_normal': initializers.GlorotNormal(seed = None), 
                    'glorot_uniform': initializers.GlorotUniform(seed = None), 
                        'zeros': initializers.Zeros()}

class Dense(layers.Layer):
    """
    Dense layer with possibility of adding batch normalization & weight decay
    """
    init_keys = ['units', 'weight_init', 'bias_init', 'batch_normal', 'activation', 'dropout']
    def __init__(self, units = None, weight_init = 'glorot_normal', bias_init = 'zeros', 
                    batch_normal = True, activation = 'relu', dropout = 0.0):
        super(Dense, self).__init__()
        self.units = units 
        self.weight_init = weight_init 
        self.bias_init = bias_init 
        self.activation = self._set_activation(activation)
        self.dropout = dropout 
        self.drop = layers.Dropout(self.dropout)
        self.batch_normal = batch_normal 
        self.batch_norm = {True:layers.BatchNormalization(axis = -1, momentum = 0.99, epsilon = 0.001), 
                                False: None}[self.batch_normal]  

    @staticmethod 
    def _set_activation(activation):
        if isinstance(activation, str):
            return activate.get(activation, None)
        elif callable(activation):
            return activation 
    
    def build(self, input_shape):
        self.w = tf.Variable(initial_value = init_variable[self.weight_init](shape = (input_shape[-1], self.units),
                             dtype = 'float32'), trainable = True) 
        self.b = tf.Variable(initial_value = init_variable[self.bias_init](shape = (self.units), dtype = 'float32'), trainable = True)

    def get_config(self):
        configs = super(Dense, self).get_config().copy()
        added_keys = {key:getattr(self, key) for key in Dense.init_keys}
        return {**configs, **added_keys}

    def call(self, x, training = True):
        y = tf.matmul(x, self.w) + self.b 
        if self.batch_normal and training:
            y = self.batch_norm(y)
        if self.activation is not None:
            y = self.activation(y)
        if self.dropout != 0.0:
            y = self.drop(y, training = training)
        return y 
    
    @classmethod 
    def from_config(cls, **configs):
        configs = {key:value for key,value in configs.items() if key in cls.init_keys}
        return cls(**configs)

    



        

