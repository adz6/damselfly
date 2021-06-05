import numpy as np
import matplotlib.pyplot
import pickle as pkl
#import torch
import os

time_path = '/home/az396/project/deepfiltering/data/raw_summed_signals/time'

freq_path = '/home/az396/project/deepfiltering/data/raw_summed_signals/freq'

name = '210528_df2.pkl'

with open(os.path.join(time_path, name), 'rb') as infile:
    data = pkl.load(infile)


def Time2Freq(data):

    freq_data = np.fft.fftshift(np.fft.fft(data) / np.sqrt(data.size))


    return freq_data
    

freq_signals = []

for signal in data['x']:
    freq_signals.append(Time2Freq(signal))
    #print('time', np.mean(abs(signal)**2))
    #print('freq', np.mean(abs(Time2Freq(signal))**2))

#freq_data = {'E': data['E'], 'pa': data['pa'], 'r': data['r'], 'x': freq_signals}
freq_data = {'E': data['E'], 'pa': data['pa'], 'r': data['r'], 'z': data['z'], 'x': freq_signals}
with open(os.path.join(freq_path, name), 'wb') as outfile:
    pkl.dump(freq_data, outfile)




