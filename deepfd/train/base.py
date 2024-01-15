
# ############################### #
# Base classes to train ML models #
# ############################### #
import tensorflow as tf
from abc import ABCMeta, abstractmethod 
from os import path, makedirs 
from datetime import datetime 
from ..toolkit import paths 


class Trainer(metaclass = ABCMeta):
	"""
	model and parameters are not-None inputs
		if model is used to retrain
	"""

	def __init__(self, model = None, **parameters):
		self.parameters = parameters 
		self.model = model 
		self.train_data = None 
		self.test_data = None 
		self._train_batch_size = None 
		save_path = path.join(paths.SAVE_IN,
			 self.__class__.__name__ + '_training_on_' + datetime.now().strftime('%Y-%m-%d-%H-%M'))
		if not path.exists(save_path):
			makedirs(save_path)
		self.save_path = save_path

	@abstractmethod
	def set_model(self):
		... 
	
	@abstractmethod
	def configure_data(self):
		...
	
	@abstractmethod
	def configure_model(self):
		...
	
	@abstractmethod
	def train(self):
		...
	
	@abstractmethod
	def save(self):
		...
	
	@staticmethod
	def mse_loss(x, y):
		return tf.reduce_mean(tf.square(x - y))
	
	@staticmethod
	def mae_loss(x, y):
		return tf.reduce_mean(tf.abs(x - y))
	


		

