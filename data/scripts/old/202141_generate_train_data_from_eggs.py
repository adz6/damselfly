import MFParse
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

path_to_eggs = '/home/az396/project/data/phase3sim/Trap_V00_00_10_prml_run1/subarray_summation'
v_range = 5.5e-8

list_sim = os.listdir(path_to_eggs)

simulation_names = []
simulation_time_series = []
for i, sim in enumerate(list_sim):
    if os.path.isdir(os.path.join(path_to_eggs, sim)):
        egg_file = os.path.join(path_to_eggs, sim, 'locust_mc_' + sim + '.egg')
        parsed_egg = MFParse.parse_egg(egg_file, Vrange = v_range)
        egg_time_series = np.sum(parsed_egg, axis = 0)[2*8192 : 3*8192]
        
        simulation_names.append(sim)
        simulation_time_series.append(egg_time_series)
    if i % 100 == 99:
        print('Done with %.d' % (i + 1))

data = {'sims': simulation_names, 'x': np.array(simulation_time_series)}

with open('raw_data_all.pkl', 'wb') as outfile:
    pkl.dump(data, outfile)
#parsed_egg = MFParse.parse_egg(test_egg, Vrange = v_range)
#sum_egg = np.sum(parsed_egg, axis = 0)


#egg_summed_time_series = sum_egg[2*8192:3*8192]

