import deepfilter as df
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt

temp = 10.0
result_date = '210602'
result_train_dset = '210602_df1_ch3'
result_test_dset = '210602_df2_test_ch3'
result_model = 'df_conv6_fc2_3ch'
result_domain = 'freq'
result_epoch = 48

plot_date = '210604'
plot_font_size = 18
#saved_networks = '/home/az396/project/deepfiltering/training/checkpoints'
#datasets = '/home/az396/project/deepfiltering/data/datasets'
#dataset = f'{checkpoint_domain}/{checkpoint_dset}_temp{temp}.pkl'
#checkpoint = f'date{checkpoint_date}_dset_name{checkpoint_dset}_temp{temp}_model{checkpoint_model}_domain_{checkpoint_domain}/epoch{epoch}.pth'
confusion_matrix_result_name = f'{result_date}_confusion_matrix_train_dset_{result_train_dset}_test_dset_{result_test_dset}_model_{result_model}_domain_{result_domain}_epoch{result_epoch}.pkl'
results = '/home/az396/project/deepfiltering/analysis/results'
plots = '/home/az396/project/deepfiltering/analysis/plotting/plots'

# load the matrices,
with open(os.path.join(results, confusion_matrix_result_name), 'rb') as infile:
    matrices = pkl.load(infile)

for s in ['train', 'test']:
    

    plot = df.plot.ConfusionMatrix(matrices[s], font_size=plot_font_size)
    plot_name = f'{plot_date}_{s}_confusion_matrix_train_dset_{result_train_dset}_test_dset_{result_test_dset}_model_{result_model}_domain_{result_domain}_epoch_{result_epoch}.png'
    plot[1].set_title(f'Confusion Matrix, Set = {s}', size=24)
    plt.savefig(os.path.join(plots, plot_name))

#tpr_list = []
#fpr_list = []

#eff_mats_train = []
#eff_mats_test = []

#for temp in compare_temps:

#	model_path = date + '_temp' + str(temp) + model + epoch_str

#	train_path = os.path.join(top, model_path, 'train_mat.pkl')
#	test_path = os.path.join(top, model_path, 'test_mat.pkl')

#	tpr_path = os.path.join(top, model_path, 'tpr.pkl')
#	fpr_path = os.path.join(top, model_path, 'fpr.pkl')

#	with open(train_path, 'rb') as infile:
#		train_matrices.append(pkl.load(infile))
#	with open(test_path, 'rb') as infile:
#		test_matrices.append(pkl.load(infile))
#	with open(tpr_path, 'rb') as infile:
#		tpr_list.append(pkl.load(infile))
#	with open(fpr_path, 'rb') as infile:
#		fpr_list.append(pkl.load(infile))

####

# do the same for the efficiency temps
#for temp in efficiency_temps:

#	train_path = os.path.join(top, date + '_temp' + str(temp) + model + epoch_str, 'train_mat.pkl')
#	test_path = os.path.join(top, date + '_temp' + str(temp) + model + epoch_str, 'test_mat.pkl')

#	with open(train_path, 'rb') as infile:
#		eff_mats_train.append(pkl.load(infile))
#	with open(test_path, 'rb') as infile:
#		eff_mats_test.append(pkl.load(infile))
####

## plot selected classification matrices

#train_name = date + '_compare_train' + model + '.png'
#test_name = date + '_compare_test' + model + '.png'

#df.plot.ClassificationMatrix(train_matrices, compare_temps, save_path, train_name)
#df.plot.ClassificationMatrix(test_matrices, compare_temps, save_path, test_name)
####

## plot true positive rate, false alarms ##

#det_eff_name = date + '_det_eff' + model + '.png'
#fa_name = date + '_false_alarms' + model + '.png'

#df.plot.ClassMatrixObservable(eff_mats_train, eff_mats_test, efficiency_temps, [0, 0], save_path, det_eff_name)
#df.plot.ClassMatrixObservable(eff_mats_train, eff_mats_test, efficiency_temps, [1, 0], save_path, fa_name)

####

## ROC curve

#roc_name = date + '_roc' + model + '.png'
#df.plot.ROC(tpr_list, fpr_list, compare_temps, save_path, roc_name)

#info_list = []
#name = date + '_compare_10epoch' + model + '.png'
#for temp in compare_temps:

#	print(temp)
#	info_path = os.path.join(top, date + '_temp' + str(temp) + model, 'info.pkl')

#	with open(info_path, 'rb') as infile:
#		info_list.append(pkl.load(infile))
	
#save_path = '/home/az396/project/deepfiltering/analysis/plot/training_loss/compare_loss'

#df.plot.CompareTrainLoss(info_list, compare_temps, save_path, name)







