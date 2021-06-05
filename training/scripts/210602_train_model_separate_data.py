import deepfilter as df
import torch
import os
import numpy as np


temps = [10.0]
batchsize = 500
epochs = 99
ep_per_check = 3
lr = 1e-3
date = '210604'
model = df.models.df_conv6_fc2_3ch()
modelname = 'df_conv6_fc2_3ch'
domain = 'freq'
train_dataset_name = '210602_df1_ch3_class_pa_split_False'
val_dataset_name = '210602_df2_val_ch3_class_pa_split_False'
binary_weight = np.array([7.0, 1.0])




dataset_path = '/home/az396/project/deepfiltering/data/datasets'
train_path = '/home/az396/project/deepfiltering/training'


for temp in temps:

    train_name = train_dataset_name + f'_temp{temp}.pkl'
    val_name = val_dataset_name + f'_temp{temp}.pkl'
    print('Starting temperature %.1f' % temp)
    new_dir = f'{date}_dset_name{train_dataset_name}_temp{temp}_model{modelname}_domain_{domain}'

    if not os.path.isdir(os.path.join(train_path + '/checkpoints', new_dir)):
        os.mkdir(os.path.join(train_path + '/checkpoints', new_dir))
    #if not os.path.isdir(os.path.join(train_path + '/data', new_dir)):
    #    os.mkdir(os.path.join(train_path + '/data', new_dir))


    loop_model = model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loop_train_data_path = os.path.join(dataset_path, domain, train_name)
    loop_val_data_path = os.path.join(dataset_path, domain, val_name)
    
    train_dset_and_labels = df.utils.LoadDataSetAndLabels(loop_train_data_path)
    val_dset_and_labels = df.utils.LoadDataSetAndLabels(loop_val_data_path)
    
    #print(train_dset_and_labels.keys())
    X_train = train_dset_and_labels['X']
    y_train = train_dset_and_labels['y']
    
    X_val = val_dset_and_labels['X']
    y_val = val_dset_and_labels['y']
    
    checkpoint_save_path = os.path.join(train_path + '/checkpoints', new_dir)
    df.utils.TrainModel(
                loop_model, 
                X_train, 
                y_train, 
                X_val, 
                y_val, 
                device, 
                epochs, 
                batchsize, 
                lr, 
                checkpoint_save_path,
                epochs_per_checkpoint = ep_per_check, 
                binary_weight = binary_weight
                        )
    


