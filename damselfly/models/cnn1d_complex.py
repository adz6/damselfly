import torch
import torch.nn as nn
import torch.nn.functional as F

# Some of the code here has been kindly "borrowed" from this repository
# https://github.com/wavefrontshaping/complexPyTorch
#

# functions #

def complex_relu(input):
    return F.relu(input.real) + 1j * F.relu(input.imag)

def complex_leaky_relu(input):
    return F.leaky_relu(input.real) + 1j * F.leaky_relu(input.imag)

def output_size(conv_dict, input_size):

    conv = ConvLayers(**conv_dict)

    x = torch.rand((2,1,input_size), dtype=torch.cfloat)

    x = conv(x)

    return x.shape[-1]* x.shape[-2]

def apply_complex(fr, fi, input, dtype = torch.cfloat):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

# layers #

class ComplexLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

class ComplexLeakyRelu(nn.Module):
    def forward(self, input):
        return complex_leaky_relu(input)

class ComplexRelu(nn.Module):
    def forward(self, input):
        return complex_relu(input)

class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,3))
            self.bias = nn.Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype = torch.cfloat))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:,:2],1.4142135623730951)
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)

class ComplexBatchNorm1d(_ComplexBatchNorm):

    def forward(self, input):

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):            
            # calculate mean of real and imaginary part
            mean_r = input.real.mean(dim=0).type(torch.cfloat)
            mean_i = input.imag.mean(dim=0).type(torch.cfloat)
            mean = mean_r + 1j*mean_i
        else:
            mean = self.running_mean
        
        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                print( mean.shape)
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, ...]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = input.real.var(dim=0,unbiased=False)+self.eps
            Cii = input.imag.var(dim=0,unbiased=False)+self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=0)
        else:
            Crr = self.running_covar[:,0]+self.eps
            Cii = self.running_covar[:,1]+self.eps
            Cri = self.running_covar[:,2]
            
        if self.training and self.track_running_stats:
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,2]

        # calculate the inverse square root the covariance matrix
        det = Crr*Cii-Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii+Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st
        
        input = (Rrr[None,:]*input.real+Rri[None,:]*input.imag).type(torch.cfloat) \
                + 1j*(Rii[None,:]*input.imag+Rri[None,:]*input.real).type(torch.cfloat)

        if self.affine:
            input = (self.weight[None,:,0]*input.real+self.weight[None,:,2]*input.imag+\
                    self.bias[None,:,0]).type(torch.cfloat) \
                    +1j*(self.weight[None,:,2]*input.real+self.weight[None,:,1]*input.imag+\
                    self.bias[None,:,1]).type(torch.cfloat)


        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return input

class NaiveComplexBatchNorm1d(nn.Module):
    '''
    Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, \
                 track_running_stats=True):
        super(NaiveComplexBatchNorm1d, self).__init__()
        self.bn_r = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_i = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self,input):
        return self.bn_r(input.real).type(torch.cfloat) +1j*self.bn_i(input.imag).type(torch.cfloat)

# model constructors #

def ConvLayer(in_features, out_features, kernel, stride, act, padding='same'):
    return torch.nn.Sequential(
        nn.Conv1d(
            in_features,
            out_features,
            kernel_size=kernel, 
            stride=stride,
            padding=padding,
            padding_mode='zeros',
            dtype=torch.cfloat
            ),
        NaiveComplexBatchNorm1d(out_features, eps=1e-5, momentum=0.1, affine=True,\
                        track_running_stats=True),
        act()
        )

def ConvLayers(channels, kernels, strides, act):

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
                padding=padding
                )
            )

    return torch.nn.Sequential(*layers)

def ComplexLinearLayer(in_features, out_features, act):

    return torch.nn.Sequential(
        ComplexLinear(in_features, out_features),
        NaiveComplexBatchNorm1d(out_features, eps=1e-5, momentum=0.1, affine=True,\
                        track_running_stats=True),
        act(),
    )

def ComplexLinearLayers(sizes, act):

    layers = []

    for i in range(len(sizes)-2):
        layers.append(ComplexLinearLayer(sizes[i], sizes[i+1], act))

    layers.append(ComplexLinear(sizes[-2], sizes[-1]))

    return torch.nn.Sequential(*layers)

# models #

'''
conv_dict = {
    'channels':[...],
    'kernels':[...],
    'strides':[...],
    'act': act_layer,
}
'''

'''
linear_dict = {
    'sizes':[...],
    'act:act_layer,
}
'''

class ComplexCNN(nn.Module):
    def __init__(
        self,
        conv_dict,
        linear_dict,
        ):
        super(ComplexCNN, self).__init__()
        self.conv_dict = conv_dict
        self.linear_dict = linear_dict
        
        self.conv = ConvLayers(
            self.conv_dict['channels'],
            self.conv_dict['kernels'],
            self.conv_dict['strides'],
            self.conv_dict['act'],
            )
        
        self.linear = ComplexLinearLayers(
            self.linear_dict['sizes'],
            self.linear_dict['act']
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
        
        return x.abs()

