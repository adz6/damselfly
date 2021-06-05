__name__='MFParse'

import numpy as np
from os import listdir
from os.path import join, isdir
from scipy.signal import hilbert
import h5py

def parse_egg(pathtoegg,Vmin=1e-8,Vrange=2e-8,nbit=8 ):

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

def slice_egg(egg_data,slicenum,slicesize):

    start=slicenum*slicesize
    end=(slicenum+1)*slicesize

    return egg_data[:,start:end]

def parse_testbed(path_to_data,antennakey,V_pp=0.5,adc_bits=14):
    parse_data={}

    position_paths=[]
    for i in [0,1,2]:
        if isdir(join(path_to_data,str(i))):
            parse_data.update({int(i):{}})
            position_paths.append(join(path_to_data,str(i)))

    # for each position get the acqusitions
    for i,p in enumerate(position_paths):
        for j in [0,1,2]:
            if isdir(join(p,str(j))):
                parse_data[i].update({int(j):{}})

    # Read and parse the csv antenna files
    for i,k in enumerate(parse_data): # positions
        for j,l in enumerate(parse_data[k]): # acquisitions
            for antenna in antennakey[l]: # antennas
                if antenna not in parse_data[k][l].keys():
                    parse_data[k][l].update({90-antenna:[]})

    for i,pos_path in enumerate(position_paths):
        #print(pos_path)
        for j,acq in enumerate(parse_data[i]):
            for u in range(len(listdir(join(pos_path,str(acq))))): # This loop iterates through the antenna data
                x=[]
                with open(join(join(pos_path,str(acq)),'wave'+str(u)+'.txt'),newline='\n') as infile:
                    for n,m in enumerate(infile):
                        if n>=7:
                            x.append(float(m))
                parse_data[i][acq][90-antennakey[acq][u]]=np.asarray(x)

    # Remove DC offset and convert from adc bits to voltages
    for i,pos in enumerate(parse_data): # positions
        for j,acq in enumerate(parse_data[pos]): # iterates through acquisitions
            for n,ant in enumerate(parse_data[pos][acq]): # iterates through the acquisitions that antenna was in
                parse_data[pos][acq][ant]-=np.mean(parse_data[pos][acq][ant])
                parse_data[pos][acq][ant]*=(V_pp/2**adc_bits)

    #for i,pos in enumerate(parse_data):
    #    for j,acq in enumerate(parse_data[pos]):
    #        for n,ant in enumerate(parse_data[pos][acq]):
    #            print(np.sqrt(np.mean(np.array(parse_data[pos][acq][ant])**2)),pos)


    return parse_data

def combine_and_calc_phis(parse_data,Nsamples=8200,Fsample=500e6):

    phis={}

    for i,pos in enumerate(parse_data):
        phis.update({pos:{}})
        for j,acq in enumerate(parse_data[pos]):
            phis[pos].update({acq:{}})
            for n,ant in enumerate(parse_data[pos][acq]):
                alpha1=np.real((np.fft.fft(parse_data[pos][acq][ant])[:Nsamples//2])
                [np.argmax(abs(np.fft.fft(parse_data[pos][acq][ant])[:Nsamples//2]))])
                alpha2=np.imag((np.fft.fft(parse_data[pos][acq][ant])[:Nsamples//2])
                [np.argmax(abs(np.fft.fft(parse_data[pos][acq][ant])[:Nsamples//2]))])
                phis[pos][acq].update({ant:np.arctan2(-alpha2,alpha1)})

    corrected_data={}
    for i,pos in enumerate(parse_data):
        corrected_data.update({pos:{}})
        for j,acq in enumerate(parse_data[pos]):
            corrected_data[pos].update({acq:{}})
            for n,ant in enumerate(parse_data[pos][acq]):
                corrected_data[pos][acq].update({ant:hilbert(parse_data[pos][acq][ant])*
                np.exp(-1j*(phis[pos][0][90]-phis[pos][acq][90]))})


    antennas={}
    # create dictionary {position:{antenna:acquisition}}
    for i,pos in enumerate(corrected_data): #positions
        antennas.update({pos:{}})
        for j,acq in enumerate(corrected_data[pos]): #acquisitions
            for ant in corrected_data[pos][acq].keys(): # antennas
                if n not in antennas[pos].keys():
                    antennas[pos].update({ant:acq})

    combined_data={}
    for i,pos in enumerate(antennas):
        combined_data.update({pos:{}})
        for n,ant in enumerate(antennas[pos]):
            if ant not in combined_data[pos].keys():
                combined_data[pos].update({ant:corrected_data[pos][antennas[pos][ant]][ant]})

    #fold all antennas into the upper RH quadrant, mirror symmetry

    mirror_combined_data={}
    for i,pos in enumerate(combined_data):
        mirror_combined_data.update({pos:{}})
        for n,ant in enumerate(combined_data[pos]):
            mirror_combined_data[pos].update({abs(ant):combined_data[pos][ant]})

    for i,pos in enumerate(mirror_combined_data): #positions
        if pos == 2:
            for n,ant in enumerate(mirror_combined_data[pos]): # antennas
                mirror_combined_data[pos][ant]*=np.exp(-1j*(phis[1][0][90]-phis[2][0][90]))

    #for i,pos in enumerate(mirror_combined_data):
    #    for n,ant in enumerate(mirror_combined_data[pos]):
    #        print(np.sqrt(np.mean(np.real(mirror_combined_data[pos][ant])**2)),ant,pos)

    #for i,pos in enumerate(mirror_combined_data):
    #    for n, ant in enumerate(mirror_combined_data[pos]):
    #        mirror_combined_data[pos][ant]=np.real(mirror_combined_data[pos][ant])

    return mirror_combined_data,phis

def generate_array_data(combined_data,Nsamples=8200):
    array_data={}
    for i,pos in enumerate(combined_data):
        if pos ==0 or pos==1:
            array_data.update({pos:{}})
        for n,ant in enumerate(combined_data[pos]):
            if pos ==2:
                continue
            elif pos==0:
                if ant != 90 and ant !=0:
                    array_data[pos].update({ant:combined_data[pos][ant]})
                    array_data[pos].update({-ant:combined_data[pos][ant]})
                    array_data[pos].update({180-ant:combined_data[pos][ant]})
                    array_data[pos].update({-180+ant:combined_data[pos][ant]})
                elif ant ==90 or ant==0:
                    array_data[pos].update({ant:combined_data[pos][ant]})
                    array_data[pos].update({ant-180:combined_data[pos][ant]})
            elif pos ==1:
                if ant != 90 and ant !=0:
                    array_data[pos].update({ant:combined_data[pos][ant]})
                    array_data[pos].update({-ant:combined_data[pos][ant]})
                    array_data[pos].update({180-ant:combined_data[pos+1][ant]})
                    array_data[pos].update({-180+ant:combined_data[pos+1][ant]})
                elif ant ==90: #don't double count 90 degrees
                    array_data[pos].update({ant:combined_data[pos][ant]})
                    array_data[pos].update({ant-180:combined_data[pos][ant]})
                elif ant ==0: #don't double count 90 degrees
                    array_data[pos].update({ant:combined_data[pos][ant]})
                    array_data[pos].update({ant-180:combined_data[pos+1][ant]})
    return array_data

def check_egg_slice(eggSlice):
    data_grad=np.sqrt(np.gradient(np.real(eggSlice[0,:]))**2)
    mean_data_grad=np.sqrt(np.mean(np.gradient(np.real(eggSlice[0,:]))**2))
    zero_grad=np.where(data_grad==0)[0]
    N_zero_grad=np.where(np.diff(zero_grad)==1)[0].size
    if N_zero_grad>42:
        return False
    else:
        return True
