
from os import path 
from ..models.benchmarks import AutoEncoder 
from .. toolkit.data import configure_mnist_reconstruction
from .. toolkit import callbacks as my_callbacks 


class PredictOnAE:
    """
    performs predictions
        on a trained autoencoder
    model is built by loading weights from a previous model
    """
    def __init__(self, model_param_file = None, save_path = None, dataset = 'mnist'):
        model_inputs = PredictOnAE._parse_params(model_param_file)
        print(model_inputs)
        self._model = AutoEncoder(**model_inputs)
        self._model.load_weights(model_inputs['model_weights_dir'])
        self.save_path = save_path
        self._data_attributes = None 
        if dataset == 'mnist':
            datasets, attributes = configure_mnist_reconstruction(batch_size = 1)
            self.dataset = datasets[1]
            self._data_attributes = attributes[0]            
    
    @staticmethod 
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    @staticmethod
    def _parse_params(model_param_file):
        lines = open(model_param_file).read().splitlines()
        parameters = {}
        for line in lines:
            info = line.split(':')
            key,value = info[0], info[1]
            if value.isdigit():
                parameters[key] = int(value)
            elif PredictOnAE.is_float(value):
                parameters[key] = float(value)
            else:
                parameters[key] = value 
        return parameters 
    
    def __call__(self, num_predictions = 100):
        my_callbacks.PredictOnData(save_path = self.save_path, 
                                        dataset = self.dataset, num_predictions = num_predictions, 
                                                    model = self._model, image_shape = self._data_attributes)()
                                        