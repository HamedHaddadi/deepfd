import tensorflow as tf 
from os import path, getcwd, makedirs 
from shutil import rmtree 
from copy import deepcopy 
from deepfd import benchmarks 
from deepfd import architectures 
from deepfd import blocks 


class TestDenseAutoEncoder(tf.test.TestCase):
	"""
	test class for dense auto encoder 
	"""

	@classmethod 
	def setUpClass(cls):
		save_path = path.join(getcwd(), 'test_save_path')
		makedirs(save_path)
		cls.save_path = save_path 
	
	@classmethod 
	def tearDownClass(cls):
		rmtree(cls.save_path)
	
	@staticmethod 
	def mse_loss(x, y):
		return tf.reduce_mean(tf.square(x-y))
		 
	@property
	def generic_inputs(self):
		return {'weight_init': 'glorot_normal', 
					'bias_init': 'zeros', 'batch_normal': False,
					 'activation': 'relu', 'dropout': 0.0}
	
	@property
	def generic_compiled_model(self):
		inputs = self.generic_inputs 
		inputs.update({'features': [256], 'input_features': 784})
		model = benchmarks.DenseAutoEncoder(**inputs)
		model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,
			 beta_1 = 0.9, beta_2 = 0.999), loss = self.mse_loss)
		return model 

	def test_direct_init_from_inputs_feature_list(self):
		inputs = self.generic_inputs
		inputs['features'] = [128, 64]
		dense_ae = benchmarks.DenseAutoEncoder(**inputs)
		self.assertIsInstance(dense_ae.encoder, architectures.DenseCoder)
		self.assertIsInstance(dense_ae.decoder, architectures.DenseCoder)
	
	def test_decoder_layer_feature_consistency(self):
		inputs = self.generic_inputs 
		features_to_test = [[256, 128, 64], [256, 128], [256]]
		expected_features = [[128, 256, 784], [256, 784], [784]]
		inputs['input_features'] = 784 

		for to_test, expected in zip(features_to_test, expected_features):
			inputs['features'] = to_test
			dense_ae = benchmarks.DenseAutoEncoder(**inputs)
			self.assertSequenceEqual(dense_ae.decoder.features, expected)
	
	def test_model_compiles_with_optimizer_and_custom_loss(self):
		inputs = self.generic_inputs 
		inputs['features'] = [128]
		inputs['input_features'] = 784 
		dense_ae = benchmarks.DenseAutoEncoder(**inputs)
		dense_ae.compile(optimizer = 'adam', loss = self.mse_loss)
		
		self.assertIsInstance(dense_ae.optimizer, tf.keras.optimizers.Optimizer)
		
		x = tf.random.uniform((784,32))
		y = tf.random.uniform((784, 32))
		z = self.mse_loss(x,y)
		z_model = dense_ae.loss(x,y)
		self.assertAllClose(z, z_model)
	
	def test_saving_model_by_saving_weights(self):
		model = self.generic_compiled_model
		x = tf.random.uniform((12,784), minval = 0, maxval = 1)
		y = model(x)
		filename = path.join(self.save_path, 'saved_weights')
		model.save_weights(filename, overwrite = True, save_format = 'tf')
		model.load_weights(filename, skip_mismatch = False, by_name = False)
		y_new = model(x)
		self.assertAllClose(y, y_new)

		









		


	
	
	

