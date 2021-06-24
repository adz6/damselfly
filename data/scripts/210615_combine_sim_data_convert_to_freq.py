import h5py
import os
import numpy as np

simpath = '/home/az396/project/sims'
damselpath = '/home/az396/project/damselfly'

combineddata_date = '210617'
combineddata_name = 'df1'

## create a list of simulation datasets to combine
simdata_date = '210615'
simdata_name = 'df_run'

nsimdatarun = 10
simdatalist = []
for i in range(nsimdatarun):
    simdatalist.append(f'{simdata_date}_{simdata_name}{i + 1}.h5')

####


## open destination h5 file, iterate through list of signals copying each one

h5combined = h5py.File(os.path.join(damselpath, f'data/sim_data/{combineddata_date}_{combineddata_name}.h5'), 'w')
combinedgrp = h5combined.create_group('signal')

ncombine = 0
for simdata in simdatalist:
    h5simdata = h5py.File(os.path.join(simpath,f'datasets/{simdata}'), 'r')
    simdatakeys = list(h5simdata['signal'].keys())
    print(simdata)
    for key in simdatakeys:
        simdataattrs = h5simdata['signal'][key].attrs.items()
        combineddset = combinedgrp.create_dataset(
                                                f'{ncombine}', 
                                                data = np.fft.fftshift(np.fft.fft(h5simdata['signal'][key][:]) / h5simdata['signal'][key][:].size)
                                                )
        #print(combineddset)
        ncombine += 1
        for item in simdataattrs:
            combineddset.attrs.create(item[0], item[1])
    
    h5simdata.close()

h5combined.close()

####
