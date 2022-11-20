import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl

class ConvDecoder2(torch.nn.Module):
    
    def __init__(self, deconv_block1, deconv_block2, maxunpool_kerns, linear_block):
        super(ConvDecoder2, self).__init__()
        
        self.layer1 = DecLinearBlock(linear_block)
        
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size = (deconv_block1[0][0], linear_block[0][0] ))
        self.maxunpool1 = torch.nn.MaxUnpool1d(kernel_size=maxunpool_kerns[0])
        self.layer2 = DeconvBlock(deconv_block1)
        self.maxunpool2 = torch.nn.MaxUnpool1d(kernel_size=maxunpool_kerns[1])
        self.layer3 = DeconvBlock(deconv_block2)
        
    def forward(self, x, indices):
        
        x = self.layer1(x)
        
        x = self.unflatten(x)
        x = self.maxunpool1(x, indices[-1])
        x = self.layer2(x)
        x = self.maxunpool2(x, indices[-2])
        x = self.layer3(x)
        
        return x
    
class ConvEncoder2(torch.nn.Module):
    
    def __init__(self, conv_block1, conv_block2, maxpool_kerns, linear_block ):
        super(ConvEncoder2, self).__init__()
        
        self.layer1 = ConvBlock(conv_block1)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=maxpool_kerns[0], return_indices=True)
        self.layer2 = ConvBlock(conv_block2)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=maxpool_kerns[1], return_indices=True)
        self.flatten = torch.nn.Flatten()
        
        self.layer3 = EncLinearBlock(linear_block)
        
    def forward(self, x):
        
        x = self.layer1(x)
        x, indices1 = self.maxpool1(x)
        x = self.layer2(x)
        x, indices2 = self.maxpool2(x)
        x = self.flatten(x)
        
        x = self.layer3(x)
        
        return x, (indices1, indices2)
    
class ConvBlock(torch.nn.Module):
    
    def __init__(self, conv_list):
        super(ConvBlock, self).__init__()
        
        self.activation = torch.nn.ReLU
        self.layer1 = self._make_block(conv_list, self.activation)
        
        
    def forward(self, x):
        
        return self.layer1(x)
        
    def _make_block(self, conv_list, activation):
        layers = []
        
        for item in conv_list:
            layers.append(torch.nn.Conv1d(item[0], item[1], kernel_size=item[2], 
                                          dilation=item[3], stride=1, padding=item[2]//2, 
                                          padding_mode='circular'))
            layers.append(activation())
            
        return torch.nn.Sequential(*layers)
    
class DeconvBlock(torch.nn.Module):
    
    def __init__(self, deconv_list):
        super(DeconvBlock, self).__init__()
        
        self.activation = torch.nn.ReLU
        self.layer1 = self._make_block(deconv_list, self.activation)
        
        
    def forward(self, x):
        
        return self.layer1(x)
        
    def _make_block(self, deconv_list, activation):
        layers = []
        
        for item in deconv_list:
            layers.append(activation())
            layers.append(torch.nn.ConvTranspose1d(item[0], item[1], kernel_size=item[2], 
                                                   stride=1, padding=item[2]//2))
            
            
        return torch.nn.Sequential(*layers)
    
class EncLinearBlock(torch.nn.Module):

    def __init__(self, linear_block):
        super(EncLinearBlock, self).__init__()
        
        self.activation = torch.nn.ReLU
        self.layer1 = self._make_block(linear_block, self.activation)
        
    def forward(self, x):
        
        return self.layer1(x)
    
    def _make_block(self, linear_block, activation):
        layers = []
        
        for i in range(len(linear_block)):
            layers.append(torch.nn.Linear(linear_block[i][0], linear_block[i][1]))
            if i < len(linear_block) - 1:
                layers.append(activation())

        return torch.nn.Sequential(*layers)

class DecLinearBlock(torch.nn.Module):
    
    def __init__(self, linear_block):
        super(DecLinearBlock, self).__init__()
        
        self.activation = torch.nn.ReLU
        self.layer1 = self._make_block(linear_block, self.activation)
        
    def forward(self, x):
        
        return self.layer1(x)
    
    def _make_block(self, linear_block, activation):
        layers = []
        
        for i in range(len(linear_block)):
            layers.append(activation())
            layers.append(torch.nn.Linear(linear_block[i][0], linear_block[i][1]))

        return torch.nn.Sequential(*layers)

class ConvAE(torch.nn.Module):
    
    def __init__(self, 
                 conv_block1, 
                 conv_block2, 
                 maxpool_kerns, 
                 enc_linear_block, 
                 deconv_block1, 
                 deconv_block2, 
                 maxunpool_kerns, 
                 dec_linear_block
                ):
        super(ConvAE, self).__init__()
        
        self.encoder = ConvEncoder2(conv_block1, conv_block2, maxpool_kerns, enc_linear_block)
        self.decoder = ConvDecoder2(deconv_block1, deconv_block2, maxunpool_kerns, dec_linear_block)
        
    def forward(self, x):
        
        x, indices = self.encoder(x)
        x = self.decoder(x, indices)
        
        return x
