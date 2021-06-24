import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl


def ConvRelu(in_f, out_f, kernel, dilation):

    return nn.Sequential(
                    nn.Conv1d(in_f, out_f, kernel_size=kernel, stride=1, dilation=dilation),
                    nn.ReLU()
                         )

def ConvMaxpoolBlock(conv_in_f, conv_out_f, conv_kernel, conv_dilation, maxpool_kernel):

    conv_relu_blocks = []
    for convset in zip(conv_in_f, conv_out_f, conv_kernel, conv_dilation):
        #print(convset)
        conv_relu_blocks.append(ConvRelu(convset[0], convset[1], convset[2], convset[3]))
    #[ConvRelu(convset[0], convset[1], convset[2], convset[3]) for convset in zip(conv_in_f, conv_out_f, conv_kernel, conv_dilation)]
    
    return nn.Sequential(
                        *conv_relu_blocks,
                        nn.MaxPool1d(kernel_size=maxpool_kernel)
                        )

def StackConvMaxpool(conv_max_list):
    conv_max_blocks = []
    for conv_max_set in conv_max_list:
        #print(conv_max_set)
        conv_max_blocks.append(ConvMaxpoolBlock(conv_max_set[0], conv_max_set[1], conv_max_set[2], conv_max_set[3], conv_max_set[4]))
    #[ConvMaxpoolBlock(conv_max_set[0], conv_max_set[1], conv_max_set[2], conv_max_set[3], conv_max_set[4]) for conv_max_set in conv_max_list]
    
    return nn.Sequential(*conv_max_blocks)

def CalcConvMaxpoolOutputSize(conv_max_list, ninput_ch, ninput):
    conv_stack = StackConvMaxpool(conv_max_list)
    
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
                        nn.ReLU(),
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

        self.conv = StackConvMaxpool(conv_list)
        
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
        x = self.linear(x)
        x = self.linear_out(x)
        
        return x



