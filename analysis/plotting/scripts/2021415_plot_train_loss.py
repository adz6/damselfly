import deepfilter as df
import numpy as np
import pickle as pkl
import os

#temps = np.concatenate(([0.1], np.arange(0.5, 10.5, 0.5)))
temps = [10.0]
#compare_temps = [0.1, 0.5, 1.0, 5.0, 10.0]
N = 100

train_path = '/home/az396/project/deepfiltering/training/checkpoints'
train_date = '210604'
train_dset = '210602_df1_ch3'
train_model = 'df_conv6_fc2_3ch'
train_domain = 'freq'
train_multiclass_type = 'pa'
split = False



# plot individual training losses
for temp in temps:

    print(temp)
    #training_info = f'date{train_date}_dset_name{train_dset}_temp{temp}_model{train_model}_domain_{train_domain}'
    training_info = f'{train_date}_dset_name{train_dset}_class_{train_multiclass_type}_split_{split}_temp{temp}_model{train_model}_domain_{train_domain}'
    training_name = f'{train_date}_training_info_dset_name{train_dset}_temp{temp}_model{train_model}_domain_{train_domain}'
    info_path = os.path.join(train_path, training_info, 'info.pkl')
    
    plot_save_path = '/home/az396/project/deepfiltering/analysis/plotting/plots'

    name = training_name + '.png'

    with open(info_path, 'rb') as infile:
        test_info = pkl.load(infile)


    df.plot.TrainingInfo(test_info, plot_save_path, name, epochs_per_xtick=6, )
####

# compare training loss on same plot

#info_list = []
#name1 = date + '_compare_trainloss' + model + '.png'
#name2 = date + '_compare_trainacc' + model + '.png'
#name3 = date + '_compare_valacc' + model + '.png'
#for temp in compare_temps:

#	print(temp)
#	info_path = os.path.join(top, date + '_temp' + str(temp) + model + epochs_str, 'info.pkl')

#	with open(info_path, 'rb') as infile:
#		info_list.append(pkl.load(infile))
#	
#save_path = '/home/az396/project/deepfiltering/analysis/plot/training_loss/compare_loss'

#df.plot.CompareTrainLoss(info_list, compare_temps, save_path, name1, epochs_per_xtick=20)
#df.plot.CompareTrainAccuracy(info_list, compare_temps, save_path, name2, epochs_per_xtick=20)
#df.plot.CompareValAccuracy(info_list, compare_temps, save_path, name3, epochs_per_xtick=20)
####










