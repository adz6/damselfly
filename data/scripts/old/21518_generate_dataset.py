import numpy as np
import matplotlib.pyplot
import pickle as pkl
import deepfilter as df
import os

domain = 'time'
raw_data_path = '/home/az396/project/deepfiltering/data/raw_summed_signals'
raw_data_name = f'{domain}/21514_variable_energy.pkl'

save_data_path = '/home/az396/project/deepfiltering/data/datasets'


with open(os.path.join(raw_data_path, raw_data_name), 'rb') as infile:
	summed_signals = pkl.load(infile)

noise_temps = [5.0, 8.0, 10.0]

for temp in noise_temps:
    save_data_name = f'{domain}/210602_variable_energy_temp{temp}.pkl'
    data_set = df.data.GenerateTrainTestData(summed_signals, os.path.join(save_data_path, save_data_name), T=temp, domain=domain, n_copies_train=25)
    print(temp)
    with open(os.path.join(save_data_path, save_data_name), 'wb') as outfile:
	    pkl.dump(data_set, outfile)
	




