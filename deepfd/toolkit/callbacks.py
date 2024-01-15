
from readline import add_history
import tensorflow as tf 
import numpy as np 
import pandas as pd 
from tensorflow.keras.callbacks import Callback, CallbackList 
from tensorflow.keras.layers import Concatenate 
from os import path, makedirs 
from datetime import datetime 
from PIL import Image 


# ##### Sanity Check callback ###### #
class SanityCheck(Callback):
    """
    sanity check takes predictions of the model during training 
        and generates an image output 
        knowing that it may receive batches of data, only 'num_outputs' are chosen
    """
    model_call = False
    def __init__(self, save_path = None, reshape_to = None):
        super(Callback, self).__init__()
        save_path = path.join(save_path, 'sanity_check_outputs')
        if not path.exists(save_path):
            makedirs(save_path)
        self.save_path = save_path 
        self.reshape_to = list(reshape_to) 
    
    def on_epoch_end(self, epoch, num_outputs = 1, logs = None, predictions = None, **kwargs):
        """
        data must have a shape of (num_natches,...)
        """
        random_index = tf.random.shuffle(tf.range(0, tf.shape(predictions)[0]))[:num_outputs]
        random_images = tf.reshape(tf.gather(predictions, random_index), [-1] + self.reshape_to).numpy()
        for n_output in range(num_outputs):
            img_name = f'sanity_check_img_epoch_{epoch}_number_{n_output}.tiff'
            img = Image.fromarray(random_images[n_output])
            img.save(path.join(self.save_path, img_name))


# ##### Callback List ##### #
class MyCallbackList(CallbackList):
    def set_model(self, model):
        """
        this method sets model only for callbacks that need a reference to model
        """
        self.model = model 
        for callback in self.callbacks:
            if callback.model_call is True:
                callback.set_model(model)
    
    def on_epoch_end(self, epoch, logs = None, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs = logs, **kwargs)
    





            
    