import h5py
import numpy as np
import matplotlib.pyplot as plt
import os




damselpath = '/home/az396/project/damselfly'

datasetdate = '210617'
datasetname = 'df1'

h5dataset = h5py.File(os.path.join(damselpath, f'data/sim_data/{datasetdate}_{datasetname}.h5'))

signalkeys = np.array(list(h5dataset['signal'].keys()))

randomsignalkeys = signalkeys[np.random.randint(0, len(signalkeys), 4)]

#print(h5dataset['signal'][randomsignalkeys[0]][:])

var = 50 * 1.38e-23 * 200e6 * 10 / 8192

fig = plt.figure(figsize=(10,10))

for i, key in enumerate(randomsignalkeys):
    ax = plt.subplot(2,2,i + 1)
    
    noise = np.random.multivariate_normal([0, 0], np.eye(2) * var / 2, 8192)
    noise = noise[:, 0] + 1j * noise[:, 1]
    
    print(np.sum(abs(h5dataset['signal'][key][:])**2))
    ax.plot(abs(noise)**2 , label = np.mean(abs(noise)**2) )
    ax.plot(abs(h5dataset['signal'][key][:])**2 , label = np.sum(abs(h5dataset['signal'][key][:])**2))
    
    #ax.plot(h5dataset['signal'][key][:].imag)
    title = ''
    for item in h5dataset['signal'][key].attrs.items():
        title += f'{item[0]}_{item[1]}_'
    #ax.set_xlim(0, 256)
    ax.set_title(title)
    plt.legend(loc=0)
    #ax.set_ylim(-10, 15)


plt.savefig('test1.png')

#fig = plt.figure(figsize=(10,10))

#for i, key in enumerate(randomsignalkeys):
#    ax = plt.subplot(2,2,i + 1)
#    ax.plot(abs(np.fft.fftshift(np.fft.fft(h5dataset['signal'][key][:])))**2)
#    print(0.02 * np.sum(abs(np.fft.fftshift(np.fft.fft(h5dataset['signal'][key][:])) / 8192)**2))
#    title = ''
#    for item in h5dataset['signal'][key].attrs.items():
#        title += f'{item[0]}_{item[1]}_'
#    #ax.set_xlim(0, 256)
#    ax.set_title(title)



#plt.savefig('test2.png')

h5dataset.close()
