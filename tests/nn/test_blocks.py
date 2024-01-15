import tensorflow as tf 
import numpy as np
from deepfd import blocks 
from random import randint


class TestDense(tf.test.TestCase):
	
	def setUp(self):
		super(TestDense, self).setUp()
		self._units = 32 
		self.dense = blocks.Dense(self._units, weight_init = 'glorot_normal', 
					bias_init = 'zeros', batch_normal = True, activation = 'relu',
						dropout = 0.0)
	
	def test_build_dense_layer_for_2d_arrays(self):
		features = [2**count for count in range(5)]
		x_inputs = [np.random.uniform(0,1, (14, 80, 80, feature)) for feature in features]
		for x in x_inputs:
			self.dense.build(x.shape)
			w_shape = np.zeros((x.shape[-1], self._units))
			self.assertShapeEqual(w_shape, tf.convert_to_tensor(self.dense.w))
	
	def test_get_config_method(self):
		configs = self.dense.get_config()
		self.assertContainsSubsequence(configs.keys(), self.dense.init_keys)
	
	def test_from_config_class_method(self):
		configs = self.dense.get_config()
		new_dense = blocks.Dense.from_config(**configs)
		self.assertIsInstance(new_dense, blocks.Dense)
	
	def test_dense_block_shape_consistency(self):
		features = [2**count for count in range(5)]
		x_inputs = [tf.random.uniform((randint(10,20), 80, 80, 80, feature))
			 		for feature in features]
		for x in x_inputs:
			self.dense.build(x.shape)
			y = self.dense(x)
			self.assertSequenceEqual(y.shape, list(x.shape[:-1]) + [self.dense.units])

	def tearDown(self):
		self.dense = None 
	