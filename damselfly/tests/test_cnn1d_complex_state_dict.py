import damselfly as df 
from damselfly.models import cnn1d_complex
from damselfly.data import loaders

import torch.nn as nn
import torch

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

model = cnn1d_complex.ComplexCNN(conv_dict, linear_dict)


for tensor in model.state_dict():
    print(tensor, '\t', model.state_dict()[tensor].size())

x = torch.rand((5,1,8192), dtype=torch.cfloat)

print(model(x).shape)

