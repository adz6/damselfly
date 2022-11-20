import torch
import math

def AddNoiseComplex(batch, var):

    shape = batch.shape
    
    noise_r = torch.normal(
        mean = 0,
        std = math.sqrt(var / 2),
        size = shape
        )
    noise_i = torch.normal(
        mean = 0,
        std = math.sqrt(var / 2),
        size = shape
    )

    return batch + (noise_r + 1j * noise_i).to(torch.cfloat)

def AddNoise(batch, var):

    shape = batch.shape
    
    noise = torch.normal(
        mean = 0,
        std = math.sqrt(var / 2),
        size = shape
        )

    return batch + noise

def NormBatchComplex(batch):

    batch = batch / torch.max(torch.abs(batch), dim=-1, keepdim=True)[0]

    return batch

def NormBatch(batch):

    batch = batch / torch.max(torch.max(abs(batch), dim=-1, keepdim=True)[0], dim=1, keepdim=True)[0]

    return batch

def FFT(signal):

    return torch.fft.fftshift(torch.fft.fft(signal, dim=-1, norm='forward'))

def CircularShift(batch, n):

    return torch.roll(batch, n, dims=-1)

