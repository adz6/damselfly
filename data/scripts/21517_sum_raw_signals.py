import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import deepfilter as df

path_to_eggs = '/home/az396/project/sims/sim_data/210528_deepfilter_set1'
v_range = 5.5e-8
slice_num = 0
N_start = 1500
N_slice = 8192

list_sim = os.listdir(path_to_eggs)
sims_per_file = 100


n = 0
signals = []
pa = []
rads = []
energies = []
z_ax = []
for i, sim in enumerate(list_sim):
    if os.path.isdir(os.path.join(path_to_eggs, list_sim[i])):
    
        sim = list_sim[i]
        angle = sim.split('angle')[-1].split('_')[0]
        rad = sim.split('rad')[-1].split('_')[0]
        #energy = sim.split('energy')[-1]
        energy = sim.split('energy')[-1].split('_')[0]
        z = sim.split('axial')[-1]
        
        #egg_file = os.path.join(path_to_eggs, sim, 'angle' + angle + '_rad' + rad + '_energy' + energy  + '_locust.egg')
        egg_file = os.path.join(path_to_eggs, sim, 'angle' + angle + '_rad' + rad + '_energy' + energy + '_axial' + z + '_locust.egg')
        try:
            parsed_egg = df.data.EggReader(egg_file)
        except:
            print(f'Resubmit: {sim}')
            continue
        egg_time_series = parsed_egg[:, N_start : N_start + N_slice]
        egg_signal = df.data.SumSignals(egg_time_series)
        
        signals.append(egg_signal)
        pa.append(float(angle))
        rads.append(float(rad))
        energies.append(float(energy))
        z_ax.append(float(z))
        if i % 50 == 49:
            print(f'Done with {i + 1}')
	

#data = {'E': energies, 'pa': pa, 'r': rads, 'x': signals}
data = {'E': energies, 'pa': pa, 'r': rads, 'z': z_ax, 'x': signals}
with open(f'/home/az396/project/deepfiltering/data/raw_summed_signals/210528_df1.pkl', 'wb') as outfile:
	pkl.dump(data, outfile)
	#if i % 50 == 49:
	#    print('Done with %.2f' % ((i + 1) / len(list_sim)))

