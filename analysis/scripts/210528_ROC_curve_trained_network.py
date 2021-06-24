import damselfly as df
import torch
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

### Configure ###
temp = 10.0
batchsize = 500
checkpoint_date = '210607'
checkpoint_dset = '210607_df1_multiclass_ch3'
checkpoint_model = 'df_conv6_fc2_multiclass_3ch'
checkpoint_domain = 'freq'
class_split_type = 'pa'
split = False
epoch = 54
result_date = '210608'

result_dset = '210607_df2_multiclass_test_ch3'

N_multi = 5
model = df.models.df_conv6_fc2_multiclass_3ch(N_multi)

### NO CHANGES BELOW HERE ###

saved_networks = '/home/az396/project/damselfly/training/checkpoints'
datasets = '/home/az396/project/damselfly/data/datasets'
#dataset = f'{checkpoint_domain}/{result_dset}_temp{temp}.pkl'
dataset = f'{checkpoint_domain}/{result_dset}_class_{class_split_type}_split_{split}_temp{temp}.pkl'

#checkpoint = f'date{checkpoint_date}_dset_name{checkpoint_dset}_temp{temp}_model{checkpoint_model}_domain_{checkpoint_domain}/epoch{epoch}.pth'
#checkpoint = f'date{checkpoint_date}_dset_name{checkpoint_dset}_class_{class_split_type}_split_{split}_temp{temp}_model{checkpoint_model}_domain_{checkpoint_domain}/epoch{epoch}.pth'
checkpoint = f'{checkpoint_date}_dset_name{checkpoint_dset}_class_{class_split_type}_split_{split}_temp{temp}_model{checkpoint_model}_domain_{checkpoint_domain}/epoch{epoch}.pth'

results = '/home/az396/project/damselfly/analysis/results'
ROC_result_name = f'{result_date}_roc_train_dset_{checkpoint_dset}_test_dset_{result_dset}_model_{checkpoint_model}_domain_{checkpoint_domain}_epoch{epoch}.pkl'

# load and prep model

model.load_state_dict(torch.load(os.path.join(saved_networks, checkpoint)))
model.eval()

# load data at specified temperature
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == torch.device("cuda:0"):
    print('Using GPU')
else:
    print('Using CPU, this will be awhile.')

data_all = df.utils.LoadDataSetAndLabels(os.path.join(datasets, dataset))

if split:
    X_test = data_all['test']['X']
    y_test = data_all['test']['y']
else:
    X_test = data_all['X']
    y_test = data_all['y']


#print(X.keys())
####

# confusion matrices for fixed threshold of 0.5
gamma = 0.5

ROC_result = df.utils.ROC(model, X_test, y_test, device, batchsize=batchsize)

#ROC_result = {'tpr': tpr, 'fpr': fpr}
with open(os.path.join(results, ROC_result_name), 'wb') as outfile:
    pkl.dump(ROC_result, outfile)

#for i, mat in enumerate(['train_mat.pkl', 'test_mat.pkl', 'val_mat.pkl']):
#    with open(os.path.join(model_save_path, mat), 'wb') as outfile:
#        pkl.dump(matrices[i], outfile)


#	# ROC curve, compute true positive rate and false positive rate

#	gammas = np.arange(0, 1, 0.05)
#	tpr = []
#	fpr = []
#	for gamma in gammas:
#		test_matrix = df.utils.ConfusionMatrix(model, X_test, y_test, device, threshold = gamma)

#		tpr.append(test_matrix[0, 0] / test_matrix[0, :].sum())
#		fpr.append(test_matrix[1, 0] / test_matrix[1, :].sum())

#	rates = [tpr, fpr]
#	for i, rate in enumerate(['tpr.pkl', 'fpr.pkl']):
#		with open(os.path.join(model_save_path, rate), 'wb') as outfile:
#			#print(model_save_path)
#			pkl.dump(rates[i], outfile)
#		








