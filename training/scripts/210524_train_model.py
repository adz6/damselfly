import deepfilter as df
import torch
import os
import numpy as np


temps = [10.0]
batchsize = 500
epochs = 99
ep_per_check = 3
lr = 1e-3
date = '210602'
model = df.models.df_conv9_fc2_dropout()
modelname = 'df_conv9_fc2_dropout'
domain = 'freq'
dataset_name = '210602_binary_variable_energy_class_pa_split_True'


dataset_path = '/home/az396/project/deepfiltering/data/datasets'
train_path = '/home/az396/project/deepfiltering/training'


for temp in temps:

    name = dataset_name + f'_temp{temp}.pkl'
    print('Starting temperature %.1f' % temp)
    new_dir = f'{date}_dset_name{dataset_name}_temp{temp}_model{modelname}_domain_{domain}'

    if not os.path.isdir(os.path.join(train_path + '/checkpoints', new_dir)):
        os.mkdir(os.path.join(train_path + '/checkpoints', new_dir))
    if not os.path.isdir(os.path.join(train_path + '/data', new_dir)):
        os.mkdir(os.path.join(train_path + '/data', new_dir))


    loop_model = model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loop_data_path = os.path.join(dataset_path, domain, name)
    dset_and_labels = df.utils.LoadDataSetAndLabels(loop_data_path)
    
    X_train = dset_and_labels['train']['X']
    y_train = dset_and_labels['train']['y']
    
    X_val = dset_and_labels['test']['X']
    y_val = dset_and_labels['test']['y']
    
    checkpoint_save_path = os.path.join(train_path + '/checkpoints', new_dir)
    df.utils.TrainModel(
                loop_model, X_train, y_train, X_val, y_val, 
                device, epochs, batchsize, lr, checkpoint_save_path,epochs_per_checkpoint = ep_per_check
                        )
    


