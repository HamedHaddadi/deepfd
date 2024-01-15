
import tensorflow as tf 
from deepfd import architectures, blocks  


class TestDenseCoder(tf.test.TestCase):
	"""
	test class for dense coder
	"""

	@property
	def generic_coder(self):
		inputs = {'features': [256, 64],
		 		'weight_init': 'glorot_normal',
				 	'bias_init': 'zeros', 'batch_normal': False, 'activation': 'relu', 
					 	'dropout': 0.0}
		return architectures.DenseCoder(**inputs)
	
	def test_encoder_init_by_dense_blocks(self):
		"""
		generates an encoder objects using generic inputs
		tests if Dense blocks are instantiated 
		"""
		inputs = {'start_features': 8, 'change_factor': 2, 'number_of_levels': 5}
		encoder = architectures.DenseCoder(**inputs)
		for n_level in range(encoder.number_of_levels):
			self.assertIsInstance(getattr(encoder, 'dense_block_' + str(n_level)), blocks.Dense)
	
	def test_decoder_init_by_dense_blocks(self):
		inputs = {'start_features': 8, 'change_factor': 2, 'number_of_levels': 5}
		encoder = architectures.DenseCoder(**inputs)
		features = encoder.features 
		decoder = architectures.DenseCoder(features = features)
		for n_level in range(decoder.number_of_levels):
			self.assertIsInstance(getattr(decoder, 'dense_block_' + str(n_level)), blocks.Dense)
	
	def test_coder_init_for_feature_inputs(self):
		encoder1 = architectures.DenseCoder(features = [128, 64])
		self.assertIsInstance(encoder1, architectures.DenseCoder)
		encoder2 = architectures.DenseCoder(features = None, start_features = 128, change_factor = 0.5, 
					number_of_levels = 2)
		self.assertIsInstance(encoder2, architectures.DenseCoder)
		encoder3 = architectures.DenseCoder(features = 128)
		self.assertIsInstance(encoder3, architectures.DenseCoder)
	
	def test_coder_init_from_configs(self):
		coder = self.generic_coder
		x = tf.random.uniform((12, 784))
		y = coder(x)
		configs = coder.get_config()
		new_coder = architectures.DenseCoder.from_config(**configs)
		self.assertIsInstance(new_coder, architectures.DenseCoder)
		


	
	
