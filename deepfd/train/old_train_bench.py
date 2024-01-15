
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np 
from .. models.benchmarks import AutoEncoder 
from ..toolkit import paths, callbacks  
from datetime import datetime
from PIL import Image
from functools import wraps 
from os import path, makedirs  
import sys 

class TrainAE:
    """
    training loop for reconstruction of MNIST digits 
    """
    init_keys = ['debug_mode', 'dataset', 'task', 'save_path', 'batch_fraction']
    model_keys = ['blocks', 'number_of_levels', 
                'encoder_entry_features', 'encoder_change_factor', 
                'weight_init', 'bias_init', 'activation']
    optimizer_keys = ['optimizer']
    loss_keys= ['loss']
    train_keys = ['num_epochs', 'save_model', 'checkpoint']
    all_keys = init_keys + model_keys + optimizer_keys + train_keys 
    
    def __init__(self, debug_mode = True, dataset = 'mnist',
                 task = 'image-reconstruction', save_path = None,
                        batch_fraction = None,  
                            **model_inputs):
        self.debug_mode = debug_mode 
        self._training_parameters = None 
        if self.debug_mode:
            tf.config.run_functions_eagerly(True)
        
        if save_path is None:
            save_path = path.join(paths.SAVE_RESULTS,
                 'auto_encoder_' + datetime.today().strftime('%m-%d-%H-%m'))
            if not path.exists(save_path):
                makedirs(save_path)
        self.save_path = save_path         
        
        self.training_data = None 
        self.test_data = None 
        {'mnist-image-reconstruction':self._configure_mnist_reconstruction}[dataset + '-' + task](batch_fraction = batch_fraction)

        self.model = None 
        if task == 'image-reconstruction':
            model_inputs['encoder_entry_features'] = self._mnist_features
        
        self._training_parameters = model_inputs 
        self.model = AutoEncoder(**model_inputs)
        
        image_save = path.join(self.save_path, 'predictions_during_training')
        if not path.exists(image_save):
            makedirs(image_save)
        self.image_save = image_save 

        log_dir = path.join(self.save_path, 'log_files')
        if not path.exists(log_dir):
            makedirs(log_dir)
        self.log_dir = log_dir 
        self._writer = tf.summary.create_file_writer(self.log_dir)

        # write the graph 
        self.train_callback = tf.keras.callbacks.TensorBoard(log_dir)
        self.train_callback.set_model(self.model)
    
    def _generate_model_parameter_file(self):
        with open(path.join(self.save_path, 'model_parameters.txt'), 'w') as f:
            for key,value in self._training_parameters.items():
                f.write(key + ':' + str(value) + '\n')
                
    def _configure_mnist_reconstruction(self, batch_fraction = None):
        """
        batching training datasets are performed in this method 
        betch_per_worker is an integer  
        """
        (img_train,_),(img_test,_) = tf.keras.datasets.mnist.load_data()
        img_train = (img_train/(img_train.max() - img_train.min())).astype(np.float32)
        img_test = (img_test/(img_test.max() - img_test.min())).astype(np.float32)
        _,row,col = img_train.shape
        self._image_shape = (row, col)
        features = row*col
        img_train = img_train.reshape(-1, features)
        img_test = img_test.reshape(-1, features)
        train_size = len(img_train)
        test_size = len(img_test)
        self._mnist_features = features  
        # define batch using data_size and buffer size 
        self.train_batch_size = int(batch_fraction*train_size)
        self.test_batch_size = int(batch_fraction*test_size) 
        self.training_data = tf.data.Dataset.from_tensor_slices(img_train).shuffle(buffer_size = 
                train_size).batch(int(batch_fraction*train_size), drop_remainder = False)
        self.test_data = tf.data.Dataset.from_tensor_slices(img_test).shuffle(buffer_size = test_size).batch(int(batch_fraction*test_size),
                     drop_remainder = True)

    # ** Optimizer ** #
    @staticmethod 
    def _constant_lr(**lr_kw):
        learning_rate = lr_kw['initial_rate']
        return learning_rate 
    
    def configure_optimizer(self, optimizer = None):
        learning_rate = {'constant': TrainAE._constant_lr}[optimizer['learning_rate']['method']](**optimizer['learning_rate'])
        if optimizer['method'] == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate, 
                    beta_1 = optimizer['beta_1'], beta_2 = optimizer['beta_2'], 
                          epsilon = optimizer['epsilon'])
        else:
            print('this optimizer is not defined ', optimizer['method'])
            sys.exit(0)
     
    # loss methods 
    _mse_loss = staticmethod(lambda x, y: tf.reduce_mean(tf.square(x - y)))
    _mae_loss = staticmethod(lambda x, y: tf.reduce_mean(tf.abs(x - y)))

    def configure_loss(self, loss = 'mse'):
        self.loss = {'mse':self._mse_loss, 
                        'mae': self._mae_loss}[loss]  
            
    # ### Training loops and steps ### #
    @tf.function 
    def train_step(self, input_images):
        with tf.GradientTape() as gt:
            predicted_images = self.model(input_images)
            step_loss = self.loss(predicted_images, input_images)
        gradients = gt.gradient(step_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return step_loss, predicted_images
    
    def train_model(self, num_epochs = None, save_model = True, checkpoint = True):
        acc_metric = metrics.MeanSquaredError(dtype = tf.float32)

        callback_list = [callbacks.OutputPrediction(save_path = self.image_save, 
                                image_shape = self._image_shape)]
        if checkpoint:
            checkpoint_dir = path.join(self.save_path, 'checkpoints')
            checkpoint_file = path.join(checkpoint_dir, 'training_checkpoints.ckpt')
            callback_list.append(ModelCheckpoint(filepath = checkpoint_file, save_weights_only = True, 
                            save_best_only = False, save_freq = 'epoch'))
            self._training_parameters['checkpoint_dir'] = checkpoint_dir

        train_callbacks = callbacks.MyCallbackList(callback_list, 
                    add_history = False, add_progbar = False, model = self.model)
        
       # sanity_check.set_model(self.model)
        train_callbacks.on_train_begin(logs = None)
        for n_epoch in range(num_epochs):
            epoch_loss = [] 
            num_batches = 0 
            for n_step, input_images in enumerate(self.training_data):
                step_loss, predicted_images = self.train_step(input_images)
                num_batches += 1
                with self._writer.as_default():
                    tf.summary.scalar('step_loss', step_loss, step = n_step*n_epoch)
                
                epoch_loss.append([n_step, step_loss])
                acc_metric.update_state(input_images, predicted_images)
            _acc = acc_metric.result()
            print(f'model accuracy at the end of epoch {n_epoch} is {1 - _acc}')
            acc_metric.reset_states()
            # get a prediction 
            idx = np.random.randint(0, self.train_batch_size) 
            train_callbacks.on_epoch_end(epoch = n_epoch, input = input_images[idx].numpy(), 
                                                output = predicted_images[idx].numpy(), 
                                                        loss = epoch_loss)
        if save_model:
            model_weights_dir = path.join(self.save_path, 'model_final_weights')
            self.model.save_weights(path.join(model_weights_dir, 'weights'))
            self._training_parameters['model_weights_dir'] = model_weights_dir 
        
        self._generate_model_parameter_file()
        