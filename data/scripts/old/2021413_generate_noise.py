import numpy as np 
import matplotlib.pyplot as plt
import os
import pickle as pkl

def GenerateNoise(shape , var, rng):

	noise_data = []
	for i in range(shape[0]):
		noise_data.append(rng.normal(0, np.sqrt(var), shape[1]))
	
	return noise_data


save_data_pth = '/home/az396/project/deepfiltering/data/data_sets/noise'

noise_temps = np.concatenate(([0.1], np.arange(0.5, 10.5, 0.5)))

rng = np.random.default_rng()

n_train = 60100
n_test = 6010
N = 8192

for temp in noise_temps:
	var = 4 * 1.38e-23 * 100e6 * 50 * temp / 2
	
	train_noise_data = GenerateNoise((n_train, N), var, rng)
	test_noise_data = GenerateNoise((n_test, N), var, rng)
	val_noise_data = GenerateNoise((n_test, N), var, rng)

	noise_data = {'train': train_noise_data, 'test': test_noise_data, 'val': val_noise_data}

	with open(os.path.join(save_data_pth, str(temp) + '.pkl'), 'wb') as outfile:
		pkl.dump(noise_data, outfile)

	print('Done with %.1f' % temp)
