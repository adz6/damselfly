import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math 
from pathlib import Path


def Conv2DRelu(in_ch, out_ch, kern_size, stride=1, padding='same', dilation=1, bias=True, padding_mode='circular'):
    
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            in_ch, 
            out_ch, 
            kern_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            bias=bias, 
            padding_mode=padding_mode
        ),
        torch.nn.ReLU()
    )

def ConvMaxpoolStack2D(in_ch_list, out_ch_list, kern_size_list, max_pool_kern_size):
    
    conv_relu_list = []
    
    for item in zip(in_ch_list, out_ch_list, kern_size_list):
        conv_relu_list.append(Conv2DRelu(item[0], item[1], item[2]))
        
    return torch.nn.Sequential(
        *conv_relu_list, 
        torch.nn.MaxPool2d(max_pool_kern_size)
    )

def ConvStack2D(conv_list):
    blocks = []
    
    for item in conv_list:
        blocks.append(ConvMaxpoolStack2D(item[0], item[1], item[2], item[3]))
        
    return torch.nn.Sequential(*blocks)

def GetConv2DOutputSize(conv_list, ninput_ch, input_shape):
    
    conv_stack = ConvStack2D(conv_list)
    
    x = torch.rand((1, ninput_ch, *input_shape))
    x = conv_stack(x)
    
    size= x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
        
    x = x.view(-1, num_features)
    
    return int(x.shape[-1])

def LinearDropout(in_f, out_f, pdrop):

    return torch.nn.Sequential(
                        torch.nn.Linear(in_f, out_f),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(p=pdrop)
                        )

def StackLinear(in_f, out_f, pdrop):

    linear_dropout_blocks = [LinearDropout(linearset[0], linearset[1], linearset[2]) for linearset in zip(in_f, out_f, pdrop)]
    
    return torch.nn.Sequential(
                        *linear_dropout_blocks
                        )

class DFCNN2D(torch.nn.Module):
    def __init__(self, nclass, ninput_ch, conv_list, linear_list):
        super(DFCNN2D, self).__init__()

        self.conv = ConvStack2D(conv_list)
        
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
