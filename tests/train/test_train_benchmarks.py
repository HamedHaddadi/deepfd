
import tensorflow as tf 
from os import path, getcwd, makedirs 
from shutil import rmtree 
from deepfd import train_benchmarks
from deepfd import benchmarks  


class TestAEMnist(tf.test.TestCase):

	@classmethod
	def setUpClass(cls):
		save_path = path.join(getcwd(), 'test_train')
		makedirs(save_path)
		cls.save_path = save_path 
	
	@classmethod 
	def tearDownClass(cls):
		rmtree(cls.save_path)
	
	@property
	def generic_model_inputs(self):
		return {'features': [392, 196], 'weight_init': 'glorot_normal', 
					'bias_init': 'zeros', 'batch_normal': False, 'activation': 'relu', 
						'input_features': 784}
	
	def test_set_model_generates_ae_model(self):
		trainer = train_benchmarks.DenseAEMnist(model = None)
		trainer.set_model(**self.generic_model_inputs)
		self.assertIsInstance(trainer.model, benchmarks.DenseAutoEncoder)
	
	def test_configure_data(self):
		trainer = train_benchmarks.DenseAEMnist(model = None)
		trainer.configure_data(batch_fraction = 0.1)
		self.assertEqual(trainer.input_features, 28*28)
		self.assertIsInstance(trainer.train_data, tf.data.Dataset)
		self.assertIsInstance(trainer.test_data, tf.data.Dataset)
		train_iter = iter(trainer.train_data)
		self.assertEqual(train_iter.get_next().shape[-1], 784)
	







	
