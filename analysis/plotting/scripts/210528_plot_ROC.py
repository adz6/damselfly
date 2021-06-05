import deepfilter as df
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt

temp = 10.0
result_date = '210604'
result_train_dset = '210602_df1_ch3'
result_test_dset = '210602_df2_test_ch3'
result_model = 'df_conv6_fc2_3ch'
result_domain = 'freq'
result_epoch = 33

plot_date = '210604'
mean_only = True


#saved_networks = '/home/az396/project/deepfiltering/training/checkpoints'
#datasets = '/home/az396/project/deepfiltering/data/datasets'
#dataset = f'{checkpoint_domain}/{checkpoint_dset}_temp{temp}.pkl'
#checkpoint = f'date{checkpoint_date}_dset_name{checkpoint_dset}_temp{temp}_model{checkpoint_model}_domain_{checkpoint_domain}/epoch{epoch}.pth'
ROC_result_name = f'{result_date}_roc_train_dset_{result_train_dset}_test_dset_{result_test_dset}_model_{result_model}_domain_{result_domain}_epoch{result_epoch}.pkl'
results = '/home/az396/project/deepfiltering/analysis/results'
plots = '/home/az396/project/deepfiltering/analysis/plotting/plots'

# load the matrices,
with open(os.path.join(results, ROC_result_name), 'rb') as infile:
    roc_data = pkl.load(infile)
    
plot = df.plot.ROC(roc_data, mean_only = mean_only)

plot[1].set_title(f'ROC Curve, model = {result_model}')

if mean_only:
    plot_name = f'{plot_date}_roc_mean_only_train_dset_{result_train_dset}_test_dset_{result_test_dset}_model_{result_model}_domain_{result_domain}_epoch_{result_epoch}.png'
else:
    plot_name = f'{plot_date}_roc_train_dset_{result_train_dset}_test_dset_{result_test_dset}_model_{result_model}_domain_{result_domain}_epoch_{result_epoch}.png'
plt.savefig(os.path.join(plots, plot_name))






