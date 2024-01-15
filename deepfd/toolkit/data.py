import tensorflow as tf 
import numpy as np
import tensorflow.keras.datasets as datasets 
from os import path

# #### mnist image dataset #### #
def configure_mnist_reconstruction(batch_size = None, 
                batch_fraction = None):
    """
    configures mnist dataset for image reconstruction
    """
    (img_train,_),(img_test,_) = datasets.mnist.load_data()
    img_train = (img_train/(img_train.max() - img_train.min())).astype(np.float32)
    img_test = (img_test/(img_test.max() - img_test.min())).astype(np.float32)
    _,row,col = img_train.shape
    image_shape = (row, col)
    features = row*col
    img_train = img_train.reshape(-1, features)
    img_test = img_test.reshape(-1, features)
    train_size = len(img_train)
    test_size = len(img_test)
    mnist_features = features  
    # define batch using data_size and buffer size 
    if batch_fraction is not None:
        train_batch_size = int(batch_fraction*train_size)
        test_batch_size = int(batch_fraction*test_size) 
    else:
        train_batch_size = batch_size 
        test_batch_size = batch_size 

    training_data = tf.data.Dataset.from_tensor_slices(img_train).shuffle(buffer_size = 
                train_size).batch(train_batch_size, drop_remainder = False)
    test_data = tf.data.Dataset.from_tensor_slices(img_test).shuffle(buffer_size = test_size).batch(test_batch_size,
                     drop_remainder = True)  
    return (training_data, test_data), (image_shape, mnist_features, train_batch_size, test_batch_size)