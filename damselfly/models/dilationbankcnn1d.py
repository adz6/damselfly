import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math 
from pathlib import Path


def ListDilationLayers(in_ch, out_ch, kernel_sizes, dilations):

    dilation_list = []

    for dilation in dilations:
        layer = torch.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_sizes,
            dilation=dilation,
            stride=1,
            groups=1,
            padding_mode='circular',
            padding='same'
        )

        dilation_list.append(
            torch.nn.Sequential(
                layer,
                torch.nn.BatchNorm1d(out_ch),
                torch.nn.ReLU()
            )
        )
        
    return torch.nn.ModuleList(dilation_list)

class DilationBank(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes, dilations):
        super(DilationBank, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations

        self.layer_list = ListDilationLayers(
            self.in_ch, 
            self.out_ch, 
            self.kernel_sizes,
            self.dilations,
        )

    def forward(self, x):

        output_list = []

        for layer in self.layer_list:

            output_list.append(layer(x))

        output = torch.cat(output_list, dim=1)

        return output

def DilatedConvBlock(conv_config, dilation_config):

    conv_relu_blocks = []
    for set in zip(conv_config['in_ch'], conv_config['out_ch'], 
                    conv_config['kernel_size'], conv_config['dilation']):
        conv_relu_blocks.append(ConvRelu(*set))

    dilation_bank = DilationBank(
        dilation_config['in_ch'], 
        dilation_config['out_ch'],
        dilation_config['kernel_size'],
        dilation_config['dilations']
        )

    return torch.nn.Sequential(
        *conv_relu_blocks,
        dilation_bank,
        torch.nn.MaxPool1d(kernel_size=conv_config['maxpool_kernel'])
    )

def DilatedConvStack(dilation_config):

    #conv_relu_blocks = []
    #for set in zip(conv_config['in_ch'], conv_config['out_ch'], 
    #                conv_config['kernel_size'], conv_config['dilation']):
    #    conv_relu_blocks.append(ConvRelu(*set))

    dilation_list = []

    for i, key in enumerate(dilation_config['layers']):

        layer_i = DilationBank(
            dilation_config['layers'][key]['in_ch'], 
            dilation_config['layers'][key]['out_ch'],
            dilation_config['layers'][key]['kernel_size'],
            dilation_config['layers'][key]['dilations'],
            )

        dilation_list.append(layer_i)

    return torch.nn.Sequential(
        *dilation_list,
        torch.nn.MaxPool1d(kernel_size=dilation_config['maxpool_kernel'])
    )

class DilationBankCNN(torch.nn.Module):
    def __init__(self, config):
        super(DilationBankCNN, self).__init__()

        self.dilated_conv_block = DilatedConvBlock(
            config['dilated_conv']['conv'],
            config['dilated_conv']['dilation']
            )
        self.conv_block = ConvStack1D(config['conv'])
        self.linear_block = StackLinear(config['linear'][0], config['linear'][1], config['linear'][2])
        self.linear_out = nn.Linear(config['linear'][1][-1], config['nclass'])

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
    def forward(self, x):

        x = self.dilated_conv_block(x)
        x = self.conv_block(x)
        x = x.view(-1, self.NumFlatFeatures(x))
        #print(x.shape())
        x = self.linear_block(x)
        x = self.linear_out(x)
        
        return x

class DilationBankCNN_v2(torch.nn.Module):
    def __init__(self, config):
        super(DilationBankCNN_v2, self).__init__()

        self.dilated_conv_block = DilatedConvStack(config['dilated_conv'])
        self.conv_block = ConvStack1D(config['conv'])
        self.linear_block = StackLinear(config['linear'][0], config['linear'][1], config['linear'][2])
        self.linear_out = nn.Linear(config['linear'][1][-1], config['nclass'])

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
    def forward(self, x):

        x = self.dilated_conv_block(x)
        x = self.conv_block(x)
        x = x.view(-1, self.NumFlatFeatures(x))
        #print(x.shape())
        x = self.linear_block(x)
        x = self.linear_out(x)
        
        return x
