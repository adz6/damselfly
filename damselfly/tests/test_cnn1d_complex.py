import damselfly as df 
from damselfly.models import cnn1d_complex
from damselfly.data import loaders

from pathlib import Path
import torch.nn as nn

datapath = Path.home()/'group'/'project'/'datasets'/'data'
datapath = datapath/'220609_dl_test_data_85to88deg_18575to18580ev_5mm_random.h5'

train_data, val_data,  = loaders.LoadH5ParamRange(
    path=datapath, 
    target_energy_range=(18575, 18580),
    target_pitch_range=(87.9, 88.0),
    target_radius_range=(0.005, 0.005),)

print(train_data.shape, val_data.shape)

conv_dict = {
    'channels': [1,10,10,15,15,20,20],
    'kernels': [5,5,5,5,5,5,5],
    'strides': [1,1,2,1,4,8],
    'act': cnn1d_complex.ComplexLeakyRelu
}
linear_dict = {
    'sizes': [cnn1d_complex.output_size(conv_dict, 8192) , 1024, 1024, 512, 256, 2],
    'act': nn.LeakyReLU
}

train_data = train_data.unsqueeze(dim=1)

model = cnn1d_complex.ComplexCNN(conv_dict, linear_dict)

output = model(train_data)

print(output.shape)
