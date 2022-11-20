import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math 
from pathlib import Path

def InceptionConv(n_in_filter, n_out_filter, filter_sizes, dilations):

    conv_layers = []

    for i, size in enumerate(filter_sizes):
        conv_layers.append(ConvRelu(n_in_filter, n_out_filter, size, dilations[i]))

    return conv_layers

class InceptionModule(torch.nn.Module):

    def __init__(self, n_in_filter, n_out_filter, filter_sizes, dilations):
        
        super(InceptionModule, self).__init__()

        self.n_in_filter = n_in_filter
        self.n_out_filter = n_out_filter
        self.filter_sizes = filter_sizes
        self.dilations = dilations

        self.conv_layer_list = nn.ModuleList(InceptionConv(self.n_in_filter, self.n_out_filter, self.filter_sizes, self.dilations))

    def forward(self, x):

        output_list = []
        for conv_layer in self.conv_layer_list:
            output_list.append(conv_layer(x))

        return torch.cat(output_list, 1)

def InceptionStack(in_filter_list, out_filter_list, filter_sizes, dilations, maxpool_kernel_sizes, ):
    
    stack_list = []
    
    for i in range(len(in_filter_list)):
        stack_list.append(InceptionModule(in_filter_list[i], out_filter_list[i], filter_sizes, dilations))
        stack_list.append(torch.nn.MaxPool1d(maxpool_kernel_sizes[i]))
        
    return torch.nn.Sequential(*stack_list)

class InceptionParallel5(torch.nn.Module):
    def __init__(self, nclass, ninput_ch, inception_list1, inception_list2, inception_list3, inception_list4, linear_list, dev0, dev1, dev2, dev3, dev4):
        super(InceptionParallel5, self).__init__()

        self.dev0 = dev0
        self.dev1 = dev1
        self.dev2 = dev2
        self.dev3 = dev3
        self.dev4 = dev4

        self.inception1 = InceptionStack(inception_list1[0], inception_list1[1], inception_list1[2], inception_list1[3], inception_list1[4],).to(dev0)
        self.inception2 = InceptionStack(inception_list2[0], inception_list2[1], inception_list2[2], inception_list2[3], inception_list2[4],).to(dev1)
        self.inception3 = InceptionStack(inception_list3[0], inception_list3[1], inception_list3[2], inception_list3[3], inception_list3[4],).to(dev2)
        self.inception4 = InceptionStack(inception_list4[0], inception_list4[1], inception_list4[2], inception_list4[3], inception_list4[4],).to(dev3)

        self.norm1 = nn.BatchNorm1d(len(inception_list1[2]) * inception_list1[1][0]).to(dev0)
        self.norm2 = nn.BatchNorm1d(len(inception_list2[2]) * inception_list2[1][0]).to(dev1)
        self.norm3 = nn.BatchNorm1d(len(inception_list3[2]) * inception_list3[1][0]).to(dev2)
        self.norm4 = nn.BatchNorm1d(len(inception_list4[2]) * inception_list4[1][0]).to(dev3)
        
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
        x = self.inception1(x)
        x = self.norm1(x)
        
        x = x.to(self.dev1)
        x = self.inception2(x)
        x = self.norm2(x)

        x = x.to(self.dev2)
        x = self.inception3(x)
        x = self.norm3(x)

        x = x.to(self.dev3)
        x = self.inception4(x)
        x = self.norm4(x)

        x = x.view(-1, self.NumFlatFeatures(x))
        x = x.to(self.dev4)

        x = self.linear(x)
        x = self.linear_out(x)
        
        return x
