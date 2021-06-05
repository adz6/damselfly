import deepfilter as df
import torch
import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


temp = 10.0
batchsize = 500
checkpoint_date = '210604'
checkpoint_dset = '210602_df1_ch3'
checkpoint_model = 'df_conv6_fc2_3ch'
model = df.models.df_conv6_fc2_3ch()
checkpoint_domain = 'freq'
class_split_type = 'pa'
data_split = False
epoch = 48
#multiclass = True
result_date = '210602'

if data_split:
    result_dset = '210602_df2'
else:
    train_dset = '210602_df1_ch3'
    test_dset = '210602_df2_test_ch3'
    val_dset = '210602_df2_val_ch3'

saved_networks = '/home/az396/project/deepfiltering/training/checkpoints'
datasets = '/home/az396/project/deepfiltering/data/datasets'

if data_split:
    #dataset = f'{checkpoint_domain}/{result_dset}_temp{temp}.pkl'
    dataset = f'{checkpoint_domain}/{result_dset}_class_{class_split_type}_split_{data_split}_temp{temp}.pkl'
else:
    train_dataset = f'{checkpoint_domain}/{train_dset}_class_{class_split_type}_split_{data_split}_temp{temp}.pkl'
    test_dataset = f'{checkpoint_domain}/{test_dset}_class_{class_split_type}_split_{data_split}_temp{temp}.pkl'
    val_dataset = f'{checkpoint_domain}/{val_dset}_class_{class_split_type}_split_{data_split}_temp{temp}.pkl'

#checkpoint = f'date{checkpoint_date}_dset_name{checkpoint_dset}_temp{temp}_model{checkpoint_model}_domain_{checkpoint_domain}/epoch{epoch}.pth'
#checkpoint = f'date{checkpoint_date}_dset_name{checkpoint_dset}_class_{class_split_type}_split_{data_split}_temp{temp}_model{checkpoint_model}_domain_{checkpoint_domain}/epoch{epoch}.pth'
checkpoint = f'{checkpoint_date}_dset_name{checkpoint_dset}_class_{class_split_type}_split_{data_split}_temp{temp}_model{checkpoint_model}_domain_{checkpoint_domain}/epoch{epoch}.pth'

results = '/home/az396/project/deepfiltering/analysis/results'

if data_split:
    confusion_matrix_result_name = f'{result_date}_confusion_matrix_train_dset_{checkpoint_dset}_test_dset_{result_dset}_model_{checkpoint_model}_domain_{checkpoint_domain}_epoch{epoch}.pkl'
else:
    confusion_matrix_result_name = f'{result_date}_confusion_matrix_train_dset_{train_dset}_test_dset_{test_dset}_model_{checkpoint_model}_domain_{checkpoint_domain}_epoch{epoch}.pkl'




# load and prep model

model.load_state_dict(torch.load(os.path.join(saved_networks, checkpoint)))
model.eval()

# load data at specified temperature
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == torch.device("cuda:0"):
    print('Using GPU')
else:
    print('Using CPU, this will be awhile.')
    
if data_split:

    dataset = df.utils.LoadDataSetAndLabels(os.path.join(datasets, dataset))

    X_train = dataset['train']['X']
    X_test = dataset['test']['X']
    X_val = dataset['val']['X']

    y_train = dataset['train']['y']
    y_test = dataset['test']['y']
    y_val = dataset['val']['y']
    
else:
    train_dataset = df.utils.LoadDataSetAndLabels(os.path.join(datasets, train_dataset))
    test_dataset = df.utils.LoadDataSetAndLabels(os.path.join(datasets, test_dataset))
    val_dataset = df.utils.LoadDataSetAndLabels(os.path.join(datasets, val_dataset))
    
    X_train = train_dataset['X']
    X_test = test_dataset['X']
    X_val = val_dataset['X']

    y_train = train_dataset['y']
    y_test = test_dataset['y']
    y_val = val_dataset['y']

#print(X.keys())
####

# confusion matrices for fixed threshold of 0.5
gamma = 0.5

train_matrix = df.utils.ConfusionMatrix(model, X_train, y_train, device, batchsize=batchsize)
test_matrix = df.utils.ConfusionMatrix(model, X_test, y_test, device, batchsize=batchsize)
val_matrix = df.utils.ConfusionMatrix(model, X_val, y_val, device, batchsize=batchsize)



print(train_matrix)
print(test_matrix)
print(val_matrix)

conf_matrix_result = {'train': train_matrix, 'test': test_matrix, 'val': val_matrix}
with open(os.path.join(results, confusion_matrix_result_name), 'wb') as outfile:
    pkl.dump(conf_matrix_result, outfile)

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








