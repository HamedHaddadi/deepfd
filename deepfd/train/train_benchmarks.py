
import tensorflow as tf 
import numpy as np 
from os import lseek, path 
from . base import Trainer 
from .. models import benchmarks 
from .. toolkit import nntools, callbacks  

class DenseAEMnist(Trainer):
	"""
	training dense auto encoder on mnist data
	image reconstruction 
	"""
	def __init__(self, model = None):
		super(DenseAEMnist, self).__init__(model = model)
		self._input_features = None 
		self._image_shape = None

	@property 
	def input_features(self):
		return self._input_features 
	
	@input_features.setter 
	def input_features(self, new_features):
		if self._input_features is None and isinstance(new_features, int):
			self._input_features = new_features 

	def set_model(self, features = None, start_features = None, change_factor = None, 
					number_of_levels = None, weight_init = 'glorot_normal', 
						bias_init = 'zeros', batch_normal = False, activation = 'relu', 
							input_features = None):
		self.input_features = input_features 
		inputs = {key:value for key,value in locals().items() if key not in ['__class__', 'self']}
		self.model = benchmarks.DenseAutoEncoder(**inputs)
	
	def configure_data(self, batch_size = None):
		(img_train,_),(img_test,_) = tf.keras.datasets.mnist.load_data()
		img_train = (img_train/(img_train.max() - img_train.min())).astype(np.float32)
		img_test = (img_test/(img_test.max() - img_test.min())).astype(np.float32)
		_,row,col = img_train.shape
		self.image_shape = (row, col)
		self.input_features = row*col 
		img_train = img_train.reshape(-1, self.input_features)
		img_test = img_test.reshape(-1, self.input_features)
		train_size = len(img_train)
		self.train_data = tf.data.Dataset.from_tensor_slices(img_train).shuffle(
				buffer_size = train_size).batch(batch_size, drop_remainder = True)
		self.test_data = tf.data.Dataset.from_tensor_slices(img_test)
	
	def configure_model(self, optimizer = 'adam', loss = 'mse', **optimizer_inputs):
		optimizer = nntools.OPTIMIZERS[optimizer](**optimizer_inputs)
		loss = {'mse': self.mse_loss, 'mae': self.mae_loss}[loss]
		self.model.compile(optimizer = optimizer, loss = loss)
	
	@tf.function
	def train_step(self, input_img):
		with tf.GradientTape() as gt:
			predicted_img = self.model(input_img)
			step_loss = self.model.loss(predicted_img, input_img)
		gradients = gt.gradient(step_loss, self.model.trainable_weights)
		self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
		return step_loss, predicted_img 

	def save(self):
		"""
		knowing the use of custom objects, model weights are saved
		"""
		self.model.save_weights(path.join(self.save_path, 'saved_weights'), overwrite = True, save_format = 'tf')

	def train(self, num_epochs = 200):
		epoch_loss = [] 
		training_callbacks = []
		training_callbacks.append(callbacks.SanityCheck(save_path = self.save_path, reshape_to = self.image_shape))
		callback_list = callbacks.MyCallbackList(callbacks=training_callbacks, model = None)
		for epoch in range(num_epochs):
			print(f'start training epoch {epoch} >>>')
			step_loss = 0
			for n_step,input_img in enumerate(self.train_data):
				loss, predicted_img = self.train_step(input_img)
				step_loss += loss 
			epoch_loss.append(step_loss/n_step)
			callback_list.on_epoch_end(epoch, logs = None, num_outputs =  1, predictions =  predicted_img)
		return epoch_loss 

		


		 

		




	





		



	

