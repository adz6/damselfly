import numpy as np
import matplotlib.pyplot
import pickle as pkl
#import torch
import os

raw_signals = '/home/az396/project/deepfiltering/data/raw_summed_signals/21517_variable_pitch_angle.pkl'

save_path = '/home/az396/project/deepfiltering/data/datasets'

with open(raw_signals, 'rb') as infile:
	raw_data = pkl.load(infile)


def ConvertRawData(raw_data, noise_temp, n_copies_train = 25, n_copies_test = 10, seed = 666):

	pitch_angles = []
	for n in range(len(raw_data['sims'])):
		pitch_angles.append(float(raw_data['sims'][n].split('Angle')[1].split('_')[0]))

	pitch_angles = np.array(pitch_angles)

	sorted_pitch_angles = pitch_angles[np.flip(pitch_angles.argsort())]

	sorted_signals = raw_data['x'][np.flip(pitch_angles.argsort())]

	sorted_signals = np.array(sorted_signals)

	train_data = np.real(sorted_signals[np.arange(0, 1201, 2)])
	test_data = np.real(sorted_signals[np.arange(1, 1201, 2)])

	train_pitch_angles = sorted_pitch_angles[np.arange(0, 1201, 2)]
	test_pitch_angles = sorted_pitch_angles[np.arange(1, 1201, 2)]

	# random number generation
	
	N = 8192
	rng = np.random.default_rng()

	noise_var = 1.38e-23 * 100e6 * 50 * noise_temp
	# scale noise variance by factor of 1/2 since we convert to real data.

	print('Starting training data.')
	noise_train_data = []
	for n in range(train_data.shape[0]):
		for m in range(n_copies_train):
			noise = rng.normal(loc = 0, scale = np.sqrt(noise_var), size = N)
			noise_train_data.append(noise + train_data[n, :])
		if n % 100 == 99:
			print('Done with %d' % (n + 1))

	print('Finished training data. Starting test Data.')

	noise_test_data = []
	noise_val_data = []
	for n in range(test_data.shape[0]):
		for m in range(n_copies_test):
			noise = rng.normal(loc = 0, scale = np.sqrt(noise_var), size = N)
			noise_test_data.append(noise + test_data[n, :])

			noise = rng.normal(loc = 0, scale = np.sqrt(noise_var), size = N)
			noise_val_data.append(noise + test_data[n, :])
		if n % 100 == 99:
			print('Done with %d' % (n + 1))
	
	print(len(noise_train_data))
	noisy_data_set = {'train': noise_train_data, 'test': noise_test_data, 'val': noise_val_data,
			'train_signals': train_data, 'test_signals': test_data, 'train_pa': train_pitch_angles, 'test_pa': test_pitch_angles}

	return noisy_data_set
	
noise_temps = np.concatenate(([0.1], np.arange(0.5, 10.5, 0.5)))

for temp in noise_temps:
	data = ConvertRawData(raw_data, temp)
	print(temp)
	with open(os.path.join(save_data_pth, str(temp) + '.pkl'), 'wb') as outfile:
		pkl.dump(data, outfile)
	




