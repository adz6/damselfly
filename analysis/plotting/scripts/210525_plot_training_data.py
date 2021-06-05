import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import deepfilter as df
import torch

temp=10.0
dataset_path = '/home/az396/project/deepfiltering/data/datasets'
domain = 'freq'
dataset_name = '21520_variable_energy'
name = dataset_name + f'_temp{temp}.pkl'
pa = 90.0

raw_data_path = '/home/az396/project/deepfiltering/data/raw_summed_signals'
raw_time_data_name = f'time/21514_variable_energy.pkl'
raw_freq_data_name = f'freq/21514_variable_energy.pkl'

with open(os.path.join(raw_data_path, raw_time_data_name), 'rb') as infile:
    time_signals = pkl.load(infile)
with open(os.path.join(raw_data_path, raw_freq_data_name), 'rb') as infile:
    freq_signals = pkl.load(infile)

time_data_path = os.path.join(dataset_path, 'time', name)
freq_data_path = os.path.join(dataset_path, 'freq', name)
X_time, y_time = df.utils.LoadDataSetAndLabels(time_data_path)
X_freq, y_freq = df.utils.LoadDataSetAndLabels(freq_data_path)

plot_date = '210527'
plots = '/home/az396/project/deepfiltering/analysis/plotting/plots'
plot_name = f'{plot_date}_example_training_signals_temp{temp}K_angle{pa}deg.png'


fig1 = plt.figure(figsize=(15,6))

ax1 = plt.subplot(121)

noisy_freq_signal_real = X_freq['train'][np.where(abs(np.array(X_freq['train_pa'])-pa)<0.0001)[0][0], 0, :]
noisy_freq_signal_imag = X_freq['train'][np.where(abs(np.array(X_freq['train_pa'])-pa)<0.0001)[0][0], 1, :]
noisy_freq_signal = noisy_freq_signal_real**2 + noisy_freq_signal_imag**2
noisy_time_signal = X_time['train'][np.where(abs(np.array(X_time['train_pa'])-pa)<0.0001)[0][0], 0, :]

time_ind = np.where(abs(np.array(time_signals['pa'])-pa) < 0.0001)[0][0]
freq_ind = np.where(abs(np.array(freq_signals['pa'])-pa) < 0.0001)[0][8]
freq_signal = abs(freq_signals['x'][freq_ind])**2


t = np.arange(0, 8192, 1) * 1/200e6
f = np.fft.fftshift(np.fft.fftfreq(8192, 1/200e6))
#print(X_time.keys())
ax1.plot(t, noisy_time_signal, label = 'Noisy Training Signal')
ax1.tick_params(length = 10, width = 1.5, labelsize=12)
ax1.plot(t, time_signals['x'][time_ind].real, label = 'Locust Signal (Real)')
#ax1.plot(X_time['train'][9001, 1, :])
ax1.set_xlim(0, t[500])
ax1.set_title(f'Time Series\n {pa} deg Training Signal, 10K Noise', size=14)
ax1.set_xlabel('Time', size=14)
ax1.set_ylabel('V', size=14)
ax1.legend()

ax2 = plt.subplot(122)
ax2.plot(f,noisy_freq_signal, label = 'Noisy Training Signal')
ax2.tick_params(length = 10, width = 1.5, labelsize=12)
ax2.plot(f,freq_signal, label = 'Locust Signal')

x1 = np.array(freq_signals['x'][freq_ind].real)
x2 = np.array(freq_signals['x'][freq_ind].imag)

y = np.sqrt(x1**2 + x2**2)
#ax2.plot(f,y, label = 'Locust Signal')
ax2.set_title(f'Power Spectum\n {pa} deg Training Signal, 10K Noise', size=14)
ax2.set_xlabel('Frequency', size=14)
ax2.set_ylabel(f'$|V|^2$', size=14)
ax2.legend()


#ax2.plot()
#ax2.plot(X_freq['train'][9001, 1, :])
#ax2.plot(X_freq['train'][12001, 0, :]**2 + X_freq['train'][12001, 1, :]**2)

plt.savefig(os.path.join(plots, plot_name))
