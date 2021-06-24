import damselfly as df
import torch
import os
import numpy as np

damselpath = '/home/az396/project/damselfly'

temps = [20.0]
batchsize = 1000
epochs = 50
ep_per_check = 1
lr = 1e-3
traindate = '210623'

n_multiclass = 2
nch = 3

# define convolutional + maxpool layers
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


model = df.models.DFCNN(n_multiclass, nch, conv_list, linear_list)
modelname = 'dfcnn'

binary = True
binary_weight = np.array([5.0, 1.0])
multi_weight = np.array([7.0, 1.0, 1.0, 1.0, 1.0])

dataset_date = '210622'
dataset_name = 'df7'


path2data = os.path.join(damselpath, f'data/datasets/{dataset_date}_{dataset_name}.h5')


for temp in temps:

    print('Starting temperature %.1f' % temp)
    new_dir = f'{traindate}_dset_name_{dataset_name}_temp{temp}_model_{modelname}'

    if not os.path.isdir(os.path.join(damselpath, 'training/checkpoints', new_dir)):
        os.mkdir(os.path.join(damselpath, 'training/checkpoints', new_dir))


    loop_model = model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    checkpoint_save_path = os.path.join(damselpath, 'training/checkpoints', new_dir)
    df.utils.TrainModel(
                loop_model, 
                path2data, 
                device, 
                epochs, 
                batchsize, 
                lr, 
                checkpoint_save_path,
                epochs_per_checkpoint = ep_per_check,
                binary = binary,
                binary_weight = binary_weight,
                multiclass_weight = multi_weight
                        )
    


