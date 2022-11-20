import h5py
import numpy as np
import os
import pickle as pkl
import torch

def EggReader(pathtoegg, Vrange=5.5e-8, nbit=8):

    f=h5py.File(pathtoegg,'r')
    dset=f['streams']['stream0']['acquisitions']['0']
    channels=list(f['channels'].keys())

    Nsamp=dset.shape[1]//(2*len(channels))
    ind=[]
    for ch in channels:
        ind.append(int(ch.split('l')[1]))
        #print(ch.split('l'))
    ind=np.array(ind)

    data=dset[0,:].reshape(ind.size,2*Nsamp)

    Idata=np.float64(data[:,np.arange(0,2*Nsamp,2)])
    Qdata=np.float64(data[:,np.arange(1,2*Nsamp,2)])

    for i in range(len(channels)):
        Idata[i,:]-=np.mean(Idata[i,:])
        Qdata[i,:]-=np.mean(Qdata[i,:])

    for i in range(len(channels)):
        Idata[i,:]*=Vrange/(2**nbit)
        Qdata[i,:]*=Vrange/(2**nbit)

    complexdata=Idata+1j*Qdata

    f.close()

    return complexdata