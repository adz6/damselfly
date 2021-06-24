import h5py
import numpy as np
import os
import damselfly as df

damselpath = '/home/az396/project/damselfly'
simdata_name = 'df1'
simdata_date = '210617'

#augdata_date = '210617'
#augdata_name = 'df4'

#ncopies_list = [25, 10, 10]

#split_train = 0.6
#splitgroups = ['train', 'test', 'val']

#noisefractions = [0.2, 0.5, 0.5]

noise_temp = 10

#nch = 3

## open sim data h5 file for reading, create new dataset h5 file

h5simdata = h5py.File(os.path.join(damselpath, f'data/sim_data/{simdata_date}_{simdata_name}.h5'), 'r')

#h5augdata = h5py.File(os.path.join(damselpath, f'data/datasets/{augdata_date}_{augdata_name}.h5'), 'w')
#h5augdata.attrs.create('T', noise_temp)


## 

## get info about simdata, split signals into train, test, val groups

simkeys = np.array(list(h5simdata['signal'].keys()))
nsim = simkeys.size

#trainkeys = simkeys[0:int(split_train * nsim)]
#testkeys = simkeys[int(split_train * nsim):int((split_train + (1 - split_train) / 2) * nsim)]
#valkeys = simkeys[int((split_train + (1 - split_train) / 2) * nsim):]

#groupkey_list = [trainkeys, testkeys, valkeys]

## 

## iterate through all simulation signals and add noise, format as 2 x N_s numpy array
## additionally create label dataset, N_signal x 1 shape numpy array. Label 1 for signal 0 for noise

for i, simkey in enumerate(simkeys):

    
    #datasetshape = ( 
    #                int(ncopies_list[i] * groupkey_list[i].size * (1 / (1 - noisefractions[i]))),
    #                nch, 
    #                h5simdata['signal']['0'][:].size
    #                )
    
    #numpy_dataset = np.zeros(datasetshape)
    #numpy_labels = np.zeros(datasetshape[0])

    #for j, key in enumerate(groupkey_list[i]):
        #simdata_items = h5simdata['signal'][key].attrs.items()

    #    for k in range(ncopies_list[i]):
    for item in h5simdata['signal'][simkey].attrs.items():
        print(item[1])
    print(10 * np.log10(np.max(abs(df.data.AddNoise(h5simdata['signal'][simkey][:], noise_temp))**2) / np.mean(abs(df.data.AddNoise(h5simdata['signal'][simkey][:], noise_temp))**2)))
    input()
#h5augdata.close()
