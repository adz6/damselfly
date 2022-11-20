import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math 
from pathlib import Path


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
                       

class MLP(torch.nn.Module):
    def __init__(self, nclass, linear_list):
        super(MLP, self).__init__()
        
        self.linear = StackLinear(linear_list[0], linear_list[1], linear_list[2])
        
        self.linear_out = nn.Linear(linear_list[1][-1], nclass)

    def NumFlatFeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
          num_features *= s
        return num_features
        
    def forward(self, x):
        #x = self.conv(x)
        x = x.view(-1, self.NumFlatFeatures(x))
        x = self.linear(x)
        x = self.linear_out(x)
        
        return x
