import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math 
from pathlib import Path


def norm1d(planes):
    
    return torch.nn.BatchNorm1d(planes)

def conv1xn(in_planes, out_planes, kernel_size, stride = 1):

    return torch.nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=1, padding_mode='circular', padding = kernel_size // 2, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    
    return torch.nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def downsample(in_planes, out_planes, stride = 1):
    
    return torch.nn.Sequential(conv1x1(in_planes, out_planes, stride = stride), norm1d(out_planes))

class ResBlock(torch.nn.Module):
    
    def __init__(self, inplanes, planes, in_kernel, out_kernel, stride=1):
        
        super(ResBlock, self).__init__()
        
        self.inplanes = inplanes
        
        self.planes = planes
        
        self.conv1 = conv1xn(self.inplanes, self.planes, in_kernel, stride=stride)
        
        self.bn1 = norm1d(self.planes)
        
        self.relu = torch.nn.ReLU(inplace=True)
        
        self.conv2 = conv1xn(self.planes, self.planes, out_kernel)
        
        self.bn2 = norm1d(self.planes)
        
        if self.inplanes != self.planes:
        
            self.downsample = downsample(self.inplanes, self.planes, stride=stride)
        
    def forward(self, x):
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        #print(out.shape, identity.shape)
        
        if self.inplanes != self.planes:
        
            identity = self.downsample(x)
            
        #print(out.shape, identity.shape)
        
        out += identity
        out = self.relu(out)
        
        return out

class resnet1d(torch.nn.Module):
    
    def __init__(self, block, block_list, stride=4, nclass = 2):
        
        super(resnet1d, self).__init__()
        
        self.inplanes = 64
        self.kernel_size = 7
        self.stride = stride
        self.output_size = 4096 // 4 ** 3
        
        self.conv1 = conv1xn(3, self.inplanes, 7, stride=1)
        self.bn1 = norm1d(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(2, padding = 0)
        
        self.layer1 = self._make_layer(block, self.inplanes, self.inplanes, self.kernel_size, block_list[0], self.stride)
        self.layer2 = self._make_layer(block, self.inplanes, 2 * self.inplanes, self.kernel_size, block_list[1], self.stride)
        self.layer3 = self._make_layer(block, 2 * self.inplanes, 4 * self.inplanes, self.kernel_size, block_list[2], self.stride)
        self.layer4 = self._make_layer(block, 4 * self.inplanes, 8 * self.inplanes, self.kernel_size, block_list[3], self.stride)
        
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        
        self.fc = torch.nn.Linear(8 * self.inplanes, nclass)
        
    def _make_layer(self, block, inplanes, outplanes, kernel_size, blocks, stride):
        
        layers = []
        layer_planes = inplanes
        
        if layer_planes == outplanes:
            layers.append(block(layer_planes, outplanes, kernel_size, kernel_size))
        else:
            layers.append(block(layer_planes, outplanes, kernel_size, kernel_size, stride=stride))
        
        layer_planes = outplanes
            
        for _ in range(1, blocks):
            layers.append(block(layer_planes, layer_planes, kernel_size, kernel_size))
            
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x
        