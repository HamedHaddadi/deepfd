
import tensorflow as tf 
from tensorflow.keras import metrics, losses 
import numpy as np 
from .. models.benchmarks import AutoEncoder 
from .. toolkit import paths 


strategy = tf.distribute.MirroredStrategy()
print(f'initiated a mirrored strategy training with {strategy.num_replicas_in_sync} machines!')

def configure_mnist_reconstruction(batch_per_worker = None, global_batch_size = None):
    (img_train,_),(img_test,_) = tf.keras.datasets.mnist.load_data()
    img_train = (img_train/(img_train.max() - img_train.min())).astype(np.float32)
    img_test = (img_test/(img_test.max() - img_test.min())).astype(np.float32)
    _,row,col = img_train.shape
    image_shape = (row, col)
    features = row*col
    img_train = img_train.reshape(-1, features)
    data_size = len(img_train)
    _mnist_features = features  
    global_batch_size = batch_per_worker*strategy.num_replicas_in_sync
    # define batch using data_size and buffer size 
    _training_data = tf.data.Dataset.from_tensor_slices(img_train).shuffle(buffer_size = data_size).batch(global_batch_size, drop_remainder = True)
    _test_data = tf.data.Dataset.from_tensor_slices(img_test).batch(global_batch_size, drop_remainder = True)
    batched_training_data = strategy.experimental_distribute_dataset(_training_data)
    batched_test_data = strategy.experimental_distribute_dataset(_test_data)  
    return batched_training_data, batched_test_data, (image_shape, _mnist_features)

with strategy.scope():
    _loss = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.NONE)
    def compute_loss(input_images, predictions, global_batch_size = None):
        per_machine_loss = _loss(input_images, predictions)
        loss = tf.nn.compute_average_loss(per_machine_loss, global_batch_size = global_batch_size)






