import torch
import damselfly as df
import os

damselpath = '/home/az396/project/damselfly'
path2data = os.path.join(damselpath, 'data/datasets/210616_df1.h5')

dataset = df.data.DFDataset(path2data, 'train')
print(dataset.data.shape)
batchsize = 5000

dataloader = torch.utils.data.DataLoader(
                                        torch.utils.data.TensorDataset(dataset.data, 
                                        dataset.label), 
                                        batchsize, 
                                        shuffle = True
                                        )
for ep in [0, 1, 2]:
    for batch, labels in dataloader:
        print(ep, labels[0:4])
