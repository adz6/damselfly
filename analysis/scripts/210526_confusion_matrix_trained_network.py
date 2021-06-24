import damselfly as df
import torch
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def ChooseBestEpoch(checkpoint):

    with open(os.path.join(checkpoint, 'info.pkl'), 'rb') as infile:
        info = pkl.load(infile)
    
    best_epoch = -1
    best_acc = -1
    for key, val in enumerate(info['val']):
        #print(torch.mean(torch.tensor(info['val'][key])).item())
        if torch.mean(torch.tensor(info['val'][key])).item() > best_acc:
            best_acc = torch.mean(torch.tensor(info['val'][key])).item()
            best_epoch = key
    return best_epoch

damselpath = '/home/az396/project/damselfly'
saved_networks = '/home/az396/project/damselfly/training/checkpoints'
datasets = '/home/az396/project/damselfly/data/datasets'
results = '/home/az396/project/damselfly/analysis/results/confusion_matrices'

train_temp = 20.0
batchsize = 500
train_date = '210623'
train_dset = 'df7'
train_model = 'dfcnn'
#train_epoch = 40
threshold = 0.5

evaldata_date = '210622'
evaldata_name = 'df7'

result_date = '210623'
result_name = f'{train_model}_temp{train_temp}_evaldata_{evaldata_name}_threshold{threshold}'

n_class = 2
nch = 3

conv_list = [
                [
                    [nch, 20], # in_f 
                    [20, 20], # out_f
                    [12, 12], # conv_kernels
                    [1, 1], # dilations
                    12 # maxpool_kernel + size
                ],
                [
                    [20, 40],
                    [40, 40],
                    [6, 6],
                    [1, 1],
                    6
                ],
                [
                    [40, 80],
                    [80, 80],
                    [3, 3],
                    [1, 1],
                    3
                ]
            ]
            
linear_list = [
                [df.models.CalcConvMaxpoolOutputSize(conv_list, nch, 8192), 416],
                [416, 213],
                [0.5, 0.5]
            ]

model = df.models.DFCNN(n_class, nch, conv_list, linear_list)

checkpoints = f'{train_date}_dset_name_{train_dset}_temp{train_temp}_model_{train_model}'

best_epoch = ChooseBestEpoch(os.path.join(saved_networks, checkpoints))
#checkpoint = f'{train_date}_dset_name_{train_dset}_temp{train_temp}_model_{train_model}/epoch{train_epoch}.pth'

model.load_state_dict(torch.load(os.path.join(saved_networks, checkpoints, f'epoch{best_epoch}.pth')))
model.eval()

path2data = os.path.join(damselpath, f'data/datasets/{evaldata_date}_{evaldata_name}.h5')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == torch.device("cuda:0"):
    print('Using GPU')
else:
    print('Using CPU, this will be awhile.')

# confusion matrices for fixed threshold of 0.5
confusionmatrix_list = df.utils.EvaluateModel(model, path2data, device, batchsize, threshold=threshold, nclass=n_class)

np.savez(os.path.join(results, f'{result_date}_{result_name}'), 
            train = np.flip(confusionmatrix_list[0]), 
            test = np.flip(confusionmatrix_list[2]),
            val = np.flip(confusionmatrix_list[1]),
        )

for i in confusionmatrix_list:
    print(i)

#print(confusionmatrix_list)


#print(train_matrix)
#print(test_matrix)
#print(val_matrix)

#conf_matrix_result = {'train': train_matrix, 'test': test_matrix, 'val': val_matrix}
#with open(os.path.join(results, confusion_matrix_result_name), 'wb') as outfile:
#    pkl.dump(conf_matrix_result, outfile)







