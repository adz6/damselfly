import deepfilter as df
import torch
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


temp = 10.0
epoch = 45
summed_signals = '/home/az396/project/deepfiltering/data/raw_summed_signals'
signal_domain = 'freq'
signal_date = '21517'
signal_type = 'variable_pitch_angle'
signal_name = f'{signal_domain}/{signal_date}_{signal_type}.pkl'


result_date = '210528'
results = '/home/az396/project/deepfiltering/analysis/results'
result_name = f'{result_date}_mf_scores_{signal_date}_{signal_type}_temp{temp}.pkl'

plots = '/home/az396/project/deepfiltering/analysis/plotting/plots'


baseline = {}

# load data at specified temperature
with open(os.path.join(summed_signals, signal_name), 'rb') as infile:
    signal_data = pkl.load(infile)
    #print(np.array(signal_data['pa']).shape)
    #print(np.array(signal_data['x']).shape)

pure_signals = np.asarray(signal_data['x'])
#pure_signals = pure_signals[0:pure_signals.shape[0] - 1, :]
#noisy_signals = np.asarray(signal['test'])
#noise = np.asarray(noise['test'])
####

# create normalized templates from signals
var = 1.38e-23 * 200e6 * temp * 50 

signal_energy = (abs(pure_signals)**2).sum(axis=1)
#print(signal_energy / (8192 * 50))

norm = (1 / np.sqrt(signal_energy * (var))).reshape((signal_energy.shape[0], 1)).repeat(pure_signals.shape[1], axis=1)

templates = norm * pure_signals

N_trials = 20
mf_scores = []
mf_scores_noise = []
for n in range(N_trials):
    print(n + 1)
    # create noisy signals 

    noise = np.random.multivariate_normal([0, 0], np.eye(2) * var / 2, pure_signals.size)
    noise = (noise[:, 0] + 1j * noise[:, 1]).reshape(pure_signals.shape)

    noisy_signals = noise + pure_signals
    
    mf_scores.extend(abs(np.diagonal(np.matmul(noisy_signals, templates.conjugate().T))))
    
    mf_scores_noise.extend(abs(np.diagonal(np.matmul(noise, templates.conjugate().T))))

#print(mf_scores, mf_scores_noise)

fig = plt.figure()
ax = plt.subplot(1,1,1)

hist1 = ax.hist(mf_scores)
hist2 = ax.hist(mf_scores_noise)

plt.savefig('/home/az396/project/deepfiltering/analysis/plotting/plots/test_distribution.png')
result = {'mf_scores': np.array(mf_scores), 'mf_scores_noise': np.array(mf_scores_noise)}

with open(os.path.join(results, result_name), 'wb') as outfile:
    pkl.dump(result, outfile)
    
tpr, fpr = df.utils.ROCFromMF(result)

plot = df.plot.ROC({'tpr': tpr, 'fpr': fpr})

plot[1].set_title(f'ROC Curve Ideal Matched Filter, data = {signal_type}')

plot_name = f'{result_date}_mf_roc_{signal_date}_{signal_type}.png'
plt.savefig(os.path.join(plots, plot_name))



