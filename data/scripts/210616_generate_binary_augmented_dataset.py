import h5py
import numpy as np
import os
import damselfly as df

damselpath = '/home/az396/project/damselfly'
simdata_name = 'df1'
simdata_date = '210617'

augdata_date = '210622'
augdata_name = 'df5'

ncopies_list = [25, 10, 10]

split_train = 0.6
splitgroups = ['train', 'test', 'val']

noisefractions = [0.2, 0.5, 0.5]

noise_temp = 20

nch = 3

## open sim data h5 file for reading, create new dataset h5 file

h5simdata = h5py.File(os.path.join(damselpath, f'data/sim_data/{simdata_date}_{simdata_name}.h5'), 'r')

h5augdata = h5py.File(os.path.join(damselpath, f'data/datasets/{augdata_date}_{augdata_name}.h5'), 'w')
#h5augdata.attrs.create('T', noise_temp)

augdatagroups = []
for name in splitgroups:
    augdatagroups.append(h5augdata.create_group(name))

## 

## get info about simdata, split signals into train, test, val groups

simkeys = np.array(list(h5simdata['signal'].keys()))
nsim = simkeys.size

trainkeys = simkeys[0:int(split_train * nsim)]
testkeys = simkeys[int(split_train * nsim):int((split_train + (1 - split_train) / 2) * nsim)]
valkeys = simkeys[int((split_train + (1 - split_train) / 2) * nsim):]

groupkey_list = [trainkeys, testkeys, valkeys]

## 

## iterate through all simulation signals and add noise, format as 2 x N_s numpy array
## additionally create label dataset, N_signal x 1 shape numpy array. Label 1 for signal 0 for noise

for i, auggroup in enumerate(augdatagroups):

    
    datasetshape = ( 
                    int(ncopies_list[i] * groupkey_list[i].size * (1 / (1 - noisefractions[i]))),
                    nch, 
                    h5simdata['signal']['0'][:].size
                    )
    
    numpy_dataset = np.zeros(datasetshape)
    numpy_labels = np.zeros(datasetshape[0])

    for j, key in enumerate(groupkey_list[i]):
        #simdata_items = h5simdata['signal'][key].attrs.items()

        for k in range(ncopies_list[i]):
            simdata_signal = df.data.AddNoise(h5simdata['signal'][key][:], noise_temp)
            
            if nch == 2:
                numpy_dataset[k + j * ncopies_list[i], :, :] = np.array([simdata_signal.real, simdata_signal.imag])
            elif nch == 3:
                numpy_dataset[k + j * ncopies_list[i], :, :] = np.array([simdata_signal.real, simdata_signal.imag, abs(simdata_signal) ** 2])
            #dset = datagroup.create_dataset(f'{(k + j * ncopies_list[i])}', data = simdata_reshape)
            #for item in simdata_items:
            #    dset.attrs.create(item[0], item[1])
            
            numpy_labels[k + j * ncopies_list[i]] = 1

    nnoise = int((noisefractions[i] / (1 - noisefractions[i])) * groupkey_list[i].size * ncopies_list[i])
    
    for n in range(nnoise):
        noisesignal = df.data.AddNoise(np.zeros(datasetshape[2]), noise_temp)
        
        if nch == 2:
            numpy_dataset[groupkey_list[i].size * ncopies_list[i] + n, :, :] = np.array([noisesignal.real, noisesignal.imag])
        elif nch == 3:
            numpy_dataset[groupkey_list[i].size * ncopies_list[i] + n, :, :] = np.array([noisesignal.real, noisesignal.imag, abs(noisesignal) ** 2])
        #dset = datagroup.create_dataset(f'{ncopies_list[i] * groupkey_list[i].size + n}', data = noisesignal_reshape)
        #dset.attrs.create('noise', 'noise')
    
    dataset = auggroup.create_dataset('data', data = numpy_dataset)
    dataset.attrs.create('T', noise_temp)
    dataset.attrs.create('ncopies', ncopies_list[i])
    dataset.attrs.create('nunique', groupkey_list[i].size)
    dataset.attrs.create('nclass',2)
    #dataset.attrs.create('classdict',{'0': 'noise', '1': 'signal'})
    
    labelset = auggroup.create_dataset('label', data = numpy_labels)
    #labelset.attrs.create('nclass', 2)
    
    
h5simdata.close()
h5augdata.close()
