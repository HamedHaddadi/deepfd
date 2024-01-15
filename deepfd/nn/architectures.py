
import tensorflow as tf 
from tensorflow.keras import layers 
from . import blocks 
from typing import Iterable, Optional, overload  
import sys 


class DenseCoder(layers.Layer):
    """
    class can be used as both encoder and decoder for dense layers
    if features input as a list: a decoder is setup
    otherwise entry_features aned change factors can be used to setup the layer
    Note: for downsampling change_factor < 1
    """
    @overload 
    def __init__(self, features: list, weight_init: str, bias_init: str,
        batch_normal: bool,  activation: str, dropout: int, 
            start_features: Optional[int] = None, change_factor: Optional[int] = None, 
                number_of_levels: Optional[int] = None, block_configs: Optional[dict] = None) -> None:
        ... 
    
    @overload 
    def __init__(self, start_features: int, change_factor: int, number_of_levels: int, 
            weight_init: str, bias_init: str, batch_normal: bool, activation: str, dropout: int,
                    features: Optional[list] = None, block_configs: Optional[dict] = None) -> None:
        ... 

    @overload 
    def __init__(self, block_configs = dict, start_features: Optional[int] = None, change_factor: Optional[int] = None, 
                number_of_levels: Optional[int] = None, features: Optional[list] = None, 
                    weight_init: Optional[str] = None, bias_init: Optional[str] = None, 
                        batch_normal: Optional[bool] = None, activation: Optional[str] = None, 
                            dropout: Optional[int] = None) -> None:
        ...


    def __init__(self, block_configs = None, features = None, start_features = None, change_factor = None, number_of_levels = None, 
                weight_init = 'glorot_normal', bias_init = 'zeros', 
                        batch_normal = False, activation = 'relu', dropout = 0.0):
                        
        super(DenseCoder, self).__init__()
        self.features = []
        self.number_of_levels = None 

        if block_configs is None:
            if features is None:
                self.number_of_levels = number_of_levels 
                self.features.extend([int(start_features*(change_factor**count)) for count in range(self.number_of_levels)])
            elif isinstance(features, Iterable):
                self.features = features 
                self.number_of_levels =len(self.features)
            elif isinstance(features, int):
                self.features.append(features)
                self.number_of_levels = 1 
                        
            dense_inputs = {key:value for key,value in locals().items() if key in blocks.Dense.init_keys}
            for n_level in range(self.number_of_levels):
                setattr(self, 'dense_block_' + str(n_level), blocks.Dense(units = self.features[n_level], **dense_inputs))
        
        elif isinstance(block_configs, dict):
            for key,configs in block_configs.items():
                setattr(self, key, blocks.Dense.from_config(**configs))
                self.features.append(configs['units'])
            self.number_of_levels = len(self.features)

    
    def get_config(self):
        new_configs = super(DenseCoder, self).get_config().copy()
        for n_level in range(self.number_of_levels):
            block_name = 'dense_block_' + str(n_level)
            block_configs = getattr(self, block_name).get_config()
            new_configs[block_name] = block_configs 
        return new_configs 
    
    def call(self, x):
        for n_level in range(self.number_of_levels):
            x = getattr(self, 'dense_block_' + str(n_level))(x)
        return x

    @classmethod 
    def from_config(cls, **configs):
        configs = {key:config for key,config in configs.items() if 'dense_block_' in key} 
        return cls(block_configs = configs)





