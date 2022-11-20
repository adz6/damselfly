import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math 
from pathlib import Path

# functions #

def output_size(conv_dict, input_size):

    conv = ConvLayers(**conv_dict)

    x = torch.rand((2,2,input_size), dtype=torch.float)

    x = conv(x)

    return x.shape[-1]* x.shape[-2]

# constructors #

def ConvLayer(in_features, out_features, kernel, stride, act, padding='same', groups=1):
    return torch.nn.Sequential(
        nn.Conv1d(
            in_features,
            out_features,
            kernel_size=kernel, 
            stride=stride,
            padding=padding,
            padding_mode='zeros',
            groups=groups
            ),
        act(),
        nn.BatchNorm1d(out_features, eps=1e-5, momentum=0.1, affine=True,\
                        track_running_stats=True),
        )

def ConvLayers(channels=None, kernels=None, strides=None, act=None, groups=1, pool=None):

    layers = []
    for i in range((len(strides))):
        if strides[i] == 1:
            padding = 'same'
        else:
            padding = 0
        layers.append(
            ConvLayer(
                channels[i],
                channels[i+1],
                kernels[i],
                strides[i],
                act,
                padding=padding,
                groups=groups
                )
            )
        layers.append(act())
        if pool is not None:
            if pool[i] > 0:
                layers.append(nn.MaxPool1d(pool[i]))
        layers.append(nn.BatchNorm1d(channels[i+1], eps=1e-5, momentum=0.1, affine=True,\
                        track_running_stats=True))

    return torch.nn.Sequential(*layers)

def LinearLayer(in_features, out_features, act):

    return torch.nn.Sequential(
        nn.Linear(in_features, out_features),
        act(),
        nn.BatchNorm1d(out_features, eps=1e-5, momentum=0.1, affine=True,\
                        track_running_stats=True),
    )

def LinearLayers(sizes, act):

    layers = []

    for i in range(len(sizes)-2):
        layers.append(LinearLayer(sizes[i], sizes[i+1], act))

    layers.append(nn.Linear(sizes[-2], sizes[-1]))

    return torch.nn.Sequential(*layers)

# cnn1d #

class Cnn1d(torch.nn.Module):
    def __init__(
        self,
        conv_dict,
        linear_dict,
        ):
        super(Cnn1d, self).__init__()
        self.conv_dict = conv_dict
        self.linear_dict = linear_dict

        self.conv = ConvLayers(**self.conv_dict)
        
        self.linear = LinearLayers(self.linear_dict['sizes'], self.linear_dict['act'])

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.NumFlatFeatures(x))
        #print(x.shape())
        x = self.linear(x)
        
        return x

## hardcoded cnn variations ##

class Cnn1d_v1(torch.nn.Module):
    def __init__(self):
        super(Cnn1d_v1, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv1d(
                in_channels=2,
                out_channels=15,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
                ),
            nn.LeakyReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(15),
            nn.Conv1d(
                in_channels=15,
                out_channels=20,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(20),
            nn.Conv1d(
                in_channels=20,
                out_channels=25,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.MaxPool1d(4),
            nn.BatchNorm1d(25),
        )

        self.linear = torch.nn.Sequential(
            nn.Linear(3175, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,2)
        )

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.NumFlatFeatures(x))
        x = self.linear(x)
        
        return x 


## older codes ##

def ConvStack1D_dropout(conv_max_list, activation):
    conv_max_blocks = []
    #print(conv_max_list, conv_max_list.shape)
    for conv_max_set in conv_max_list:
    #    print(conv_max_set)
        conv_max_blocks.append(ConvMaxpoolStack1D_dropout(
            conv_max_set[0], 
            conv_max_set[1], 
            conv_max_set[2], 
            conv_max_set[3], 
            conv_max_set[4], 
            conv_max_set[5],
            activation,
            ))
    #[ConvMaxpoolStack1D(conv_max_set[0], conv_max_set[1], conv_max_set[2], conv_max_set[3], conv_max_set[4]) for conv_max_set in conv_max_list]
    
    return nn.Sequential(*conv_max_blocks)

def ConvMaxpoolStack1D_dropout(
    conv_in_f,
    conv_out_f,
    conv_kernel,
    conv_dilation,
    conv_dropout,
    maxpool_kernel,
    activation
    ):

    conv_blocks = []
    for convset in zip(
        conv_in_f,
        conv_out_f,
        conv_kernel,
        conv_dilation,
        conv_dropout
        ):
        #print(convset)
        conv_blocks.append(Conv_dropout(
            convset[0],
            convset[1],
            convset[2],
            convset[3],
            convset[4],
            activation
            ))
    
    return nn.Sequential(
                        *conv_blocks,
                        nn.MaxPool1d(kernel_size=maxpool_kernel),
                        )

def Conv_dropout(
    in_f,
    out_f,
    kernel,
    dilation,
    dropout,
    activation,
    norm=True
    ):

    if norm:
        return nn.Sequential(
                        nn.Conv1d(
                            in_f,
                            out_f,
                            kernel_size=kernel,
                            stride=1,
                            dilation=dilation,
                            padding='same',
                            padding_mode='circular'
                            ),
                        nn.BatchNorm1d(out_f),
                        nn.Dropout(p=dropout),
                        activation, 
                            )
    else:
        return nn.Sequential(
                        nn.Conv1d(
                            in_f,
                            out_f,
                            kernel_size=kernel,
                            stride=1,
                            dilation=dilation,
                            padding='same',
                            padding_mode='circular'
                            ),
                        nn.Dropout(p=dropout),
                        activation,
                            )

## sparse CNN ## 

'''
    SparseCNN Config
    
    config = {
        'nclass':,
        'nch':,
        'conv':,
        'lin':,
        'inds':,
    }
'''

class SparseCNN(torch.nn.Module):
    def __init__(self, config):
        super(SparseCNN, self).__init__()
        self.conv_list = config['conv']
        self.lin_list = config['lin']
        self.inds = config['inds']

        self.conv = ConvStack1D(self.conv_list)
        
        self.linear = StackLinear(*self.lin_list)
        
        self.linear_out = nn.Linear(self.lin_list[1][-1], config['nclass'])

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
    def forward(self, x):

        # only looks at specified indices
        x = x[:, :, self.inds]
        #################################

        x = self.conv(x)
        x = x.view(-1, self.NumFlatFeatures(x))
        x = self.linear(x)
        x = self.linear_out(x)
        
        return x

def ConvRelu(in_f, out_f, kernel, dilation, norm=True):

    if norm:
        return nn.Sequential(
                        nn.Conv1d(in_f, out_f, kernel_size=kernel, stride=1, dilation=dilation, padding='same', padding_mode='circular'),
                        nn.BatchNorm1d(out_f),
                        nn.ReLU(), 
                            )
    else:
        return nn.Sequential(
                        nn.Conv1d(in_f, out_f, kernel_size=kernel, stride=1, dilation=dilation, padding='same', padding_mode='circular'),
                        nn.ReLU(),
                            )

def ConvMaxpoolStack1D(conv_in_f, conv_out_f, conv_kernel, conv_dilation, maxpool_kernel):

    conv_relu_blocks = []
    for convset in zip(conv_in_f, conv_out_f, conv_kernel, conv_dilation):
        #print(convset)
        conv_relu_blocks.append(ConvRelu(convset[0], convset[1], convset[2], convset[3]))
    #[ConvRelu(convset[0], convset[1], convset[2], convset[3]) for convset in zip(conv_in_f, conv_out_f, conv_kernel, conv_dilation)]
    
    return nn.Sequential(
                        *conv_relu_blocks,
                        nn.MaxPool1d(kernel_size=maxpool_kernel)
                        )

def ConvStack1D(conv_max_list):
    conv_max_blocks = []
    #print(conv_max_list, conv_max_list.shape)
    for conv_max_set in conv_max_list:
    #    print(conv_max_set)
        conv_max_blocks.append(ConvMaxpoolStack1D(conv_max_set[0], conv_max_set[1], conv_max_set[2], conv_max_set[3], conv_max_set[4]))
    #[ConvMaxpoolStack1D(conv_max_set[0], conv_max_set[1], conv_max_set[2], conv_max_set[3], conv_max_set[4]) for conv_max_set in conv_max_list]
    
    return nn.Sequential(*conv_max_blocks)

def GetConv1DOutputSize(conv_max_list, ninput_ch, ninput):
    conv_stack = ConvStack1D(conv_max_list)
    
    x = torch.rand((1, ninput_ch, ninput))
    #print(x.shape)
    x = conv_stack(x)
    
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    
    x = x.view(-1, num_features)
    
    return int(x.shape[-1])

def LinearDropout(in_f, out_f, pdrop):

    return nn.Sequential(
                        nn.Linear(in_f, out_f),
                        nn.BatchNorm1d(out_f),
                        nn.LeakyReLU(),
                        nn.Dropout(p=pdrop)
                        )

def StackLinear(in_f, out_f, pdrop):

    linear_dropout_blocks = [LinearDropout(linearset[0], linearset[1], linearset[2]) for linearset in zip(in_f, out_f, pdrop)]
    
    return nn.Sequential(
                        *linear_dropout_blocks
                        )
                        
class DFCNN(torch.nn.Module):
    def __init__(self, nclass, ninput_ch, conv_list, linear_list):
        super(DFCNN, self).__init__()
        self.conv_list = conv_list
        #print(self.conv_list.shape)

        self.conv = ConvStack1D(self.conv_list)
        
        self.linear = StackLinear(linear_list[0], linear_list[1], linear_list[2])
        
        self.linear_out = nn.Linear(linear_list[1][-1], nclass)

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.NumFlatFeatures(x))
        #print(x.shape())
        x = self.linear(x)
        x = self.linear_out(x)
        
        return x

class DFCNNParallel3(torch.nn.Module):
    def __init__(self, nclass, ninput_ch, conv_list1, conv_list2, linear_list, dev0, dev1, dev2):
        super(DFCNNParallel3, self).__init__()

        self.dev0 = dev0
        self.dev1 = dev1
        self.dev2 = dev2

        self.conv1 = ConvStack1D(conv_list1).to(dev0)
        self.conv2 = ConvStack1D(conv_list2).to(dev1)
        
        self.linear = StackLinear(linear_list[0], linear_list[1], linear_list[2]).to(dev2)
        self.linear_out = nn.Linear(linear_list[1][-1], nclass).to(dev2)

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
    def forward(self, x):

        x = x.to(self.dev0)
        x = self.conv1(x)
        
        x = x.to(self.dev1)
        x = self.conv2(x)

        x = x.view(-1, self.NumFlatFeatures(x))
        x = x.to(self.dev2)

        x = self.linear(x)
        x = self.linear_out(x)
        
        return x

class DFCNNParallel5(torch.nn.Module):
    def __init__(self, nclass, ninput_ch, conv_list1, conv_list2, conv_list3, conv_list4, linear_list, dev0, dev1, dev2, dev3, dev4):
        super(DFCNNParallel5, self).__init__()

        self.dev0 = dev0
        self.dev1 = dev1
        self.dev2 = dev2
        self.dev3 = dev3
        self.dev4 = dev4

        self.conv1 = ConvStack1D(conv_list1).to(dev0)
        self.conv2 = ConvStack1D(conv_list2).to(dev1)
        self.conv3 = ConvStack1D(conv_list3).to(dev2)
        self.conv4 = ConvStack1D(conv_list4).to(dev3)

        self.norm1 = nn.BatchNorm1d(conv_list1[0][1][0]).to(dev0)
        self.norm2 = nn.BatchNorm1d(conv_list2[0][1][0]).to(dev1)
        self.norm3 = nn.BatchNorm1d(conv_list3[0][1][0]).to(dev2)
        self.norm4 = nn.BatchNorm1d(conv_list4[0][1][0]).to(dev3)
        
        self.linear = StackLinear(linear_list[0], linear_list[1], linear_list[2]).to(dev4)
        self.linear_out = nn.Linear(linear_list[1][-1], nclass).to(dev4)

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
    def forward(self, x):

        x = x.to(self.dev0)
        x = self.conv1(x)
        x = self.norm1(x)
        
        x = x.to(self.dev1)
        x = self.conv2(x)
        x = self.norm2(x)

        x = x.to(self.dev2)
        x = self.conv3(x)
        x = self.norm3(x)

        x = x.to(self.dev3)
        x = self.conv4(x)
        x = self.norm4(x)

        x = x.view(-1, self.NumFlatFeatures(x))
        x = x.to(self.dev4)

        x = self.linear(x)
        x = self.linear_out(x)
        
        return x
