
import yaml 
import argparse 
from deepfd.train import train_benchmarks 

def parse_inputs(input_file):
	try:
		stream = open(input_file, 'r')
		inputs = yaml.safe_load(stream)
	except:
		print("not readig the yaml file")
	return inputs 

def run_dense_autoencoder_mnist(**inputs):
	if inputs['init'] == 'start_new':
		trainer = train_benchmarks.DenseAEMnist()
	else:
		raise NotImplementedError
	trainer.set_model(**inputs['set_model'])
	trainer.configure_data(**inputs['configure_data'])
	trainer.configure_model(**inputs['configure_model'])
	trainer.train(**inputs['train'])
	trainer.save()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='reading input.yaml file names')
	parser.add_argument('-filenames', nargs = '*', type = str, help = 'filenames for training models')
	args = parser.parse_args()
	
	for _file in args.filenames:
		inputs = parse_inputs(_file)
		{'benchmarks-dense-ae-mnist': run_dense_autoencoder_mnist}[inputs['trainer']](**inputs)

