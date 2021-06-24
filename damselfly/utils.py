# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import os
import time
import damselfly as df

# output size for 1d convolution
# L_out = [[L_in + 2 * padding - dilation * (kernel_size - 1) - 1]/stride + 1]

def TrainModel(model, datapath, device, epochs, batchsize, learning_rate, model_save_path, 
                epochs_per_checkpoint=5,binary=True, binary_weight = np.array([7.0, 1.0]), multiclass_weight = np.array([7.0, 1.0])):
    
    #binary_weight = np.array([1.1, 1.0])
    if binary:
        bce_weight = torch.tensor(
                                binary_weight,
                                 device=device, dtype=torch.float
                                )
    else:
        bce_weight = torch.tensor(
                                multiclass_weight,
                                 device=device, dtype=torch.float
                                )
                                
                                
    if device == torch.device("cuda:0"):
        print('Model moved to GPU')
        model.to(device)

    train_data = df.data.DFDataset(datapath, 'train')
    val_data = df.data.DFDataset(datapath, 'val')

    train_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(train_data.data, train_data.label),
                                                    batchsize,
                                                    shuffle=True
                                                    )
    val_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(val_data.data, val_data.label),
                                                    batchsize,
                                                    shuffle=True
                                                    )

    # track training
    train_loss = {}
    train_accuracy = {}
    val_accuracy_per_epoch = {}

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight = bce_weight, reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


    for ep in range(epochs):

        running_loss = 0.0
        train_loss.update({ep: []})
        train_accuracy.update({ep: []})
        val_accuracy_per_epoch.update({ep: []})

        ## random permutation of train data
        #n_rand = np.random.permutation(train_data.shape[0])
        #train_data = train_data[n_rand, :, :]
        #train_labels = train_labels[n_rand]
        t1 = time.perf_counter_ns()
        
        nbatch = 1
        for batch, labels in train_dataloader:
            #batch_labels = torch.split(train_labels, batchsize, dim=0)[i]
            
            if device == torch.device("cuda:0"):
                batch = batch.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            output = model(batch)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss[ep].append(loss.item())

            batch_accuracy = CalculateAccuracy(output, labels)
            train_accuracy[ep].append(batch_accuracy)

            # monitor training
            print('| %d | %d | %.3f | %.3f |' % ((ep + 1), nbatch, loss.item(), batch_accuracy))
            nbatch += 1

        t2 = time.perf_counter_ns()
        print(f'Epoch time: {np.round((t2-t1)*1e-9, 2)} s')
        # compute validation accuracy after 1 epoch
        batch_lengths = []
        for batch, labels in val_dataloader:
            #batch_labels = torch.split(val_labels, batchsize, dim=0)[i]

            if device == torch.device("cuda:0"):
                batch = batch.to(device)
                labels = labels.to(device)
            
            val_out = model(batch)
            
            val_acc = CalculateAccuracy(val_out, labels)
            val_accuracy_per_epoch[ep].append(val_acc)

        print('Completed epoch %d. Mean validation accuracy: %.3f' % ((ep + 1), torch.mean(torch.as_tensor(val_accuracy_per_epoch[ep]))))

        # save model checkpoint
        if ep % epochs_per_checkpoint == (epochs_per_checkpoint - 1):
            torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch{ep + 1}.pth'))


        info = {'loss': train_loss, 'acc': train_accuracy, 'val': val_accuracy_per_epoch}

        with open(os.path.join(model_save_path, 'info.pkl'), 'wb') as outfile:
            pkl.dump(info, outfile)


def EvaluateModel(model, datapath, device, batchsize, threshold=0.50, nclass=2):

    if device == torch.device("cuda:0"):
        print('Model moved to GPU')
        model.to(device)
        
        
    train_data = df.data.DFDataset(datapath, 'train')
    val_data = df.data.DFDataset(datapath, 'val')
    test_data = df.data.DFDataset(datapath, 'test')

    train_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(train_data.data, train_data.label),
                                                    batchsize,
                                                    shuffle=True
                                                    )
    val_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(val_data.data, val_data.label),
                                                    batchsize,
                                                    shuffle=True
                                                    )
    test_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(test_data.data, test_data.label),
                                                    batchsize,
                                                    shuffle=True
                                                    )
    confusionmatrix_list = []
    for i, dataloader in enumerate([train_dataloader, val_dataloader, test_dataloader]):
    
        confusionmatrix = np.zeros((nclass, nclass))
        for batch, labels in dataloader:
            
            if device == torch.device("cuda:0"):
                batch = batch.to(device)
            
            out = F.softmax(model(batch), dim=1)
            
            for j in range(labels.shape[0]):
                predict_ind = torch.where(out[j, :] >= threshold)[0].to('cpu')
            
                predicted_classes = np.zeros(nclass)
                predicted_classes[predict_ind] = 1
                #print(predicted_classes)
                for n in range(len(predicted_classes)):
                    if predicted_classes[n] == 1:
                        confusionmatrix[labels[j], n] += 1
                        
        confusionmatrix_list.append(confusionmatrix)
    
    return confusionmatrix_list

def LoadDataSetAndLabels(path):

    with open(path, 'rb') as infile:
        data_set = pkl.load(infile)
    ##print(data_set['meta']['train_pa'][-1], len(data_set['meta']['train_pa']))
    #train_labels = torch.cat(
    #                (
    #                torch.ones(len(data_set['meta']['train_pa']), dtype=torch.long), 
    #                torch.zeros(len(data_set['train']) -  len(data_set['meta']['train_pa']), dtype=torch.long)
    #                )
    #                )
    #test_labels = torch.cat(
    #                (
    #                torch.ones(len(data_set['meta']['test_pa']), dtype=torch.long),
    #                torch.zeros(len(data_set['test']) -  len(data_set['meta']['test_pa']), dtype=torch.long)
    #                )
    #                )
    #data_sets = {}
    #labels = {}
    #for dset in ['train', 'test', 'val']:
    #    data_sets.update({dset: data_set[dset]})
    #    if dset == 'train':
    #        labels.update({dset: train_labels})
    #    else:
    #        labels.update({dset: test_labels})
    #data_sets.update(data_set['meta'])
    return data_set

def CalculateAccuracy(output, labels):

	output_prob = F.softmax(output)

	most_likely_class = torch.argmax(output_prob, dim=1)

	most_likely_class_matches_label = torch.as_tensor(most_likely_class == labels, dtype=torch.float)

	return torch.mean(most_likely_class_matches_label)


def NormalizeData(X):

    norm_tensor = 1 / torch.max(torch.abs(X), 2).values.unsqueeze(2).repeat_interleave(8192, dim=2)
    #print(norm_tensor.shape, X.shape)
    return norm_tensor * X

def BestEpoch(info):

	epochs = list(info['loss'].keys())
	n_epochs = len(epochs)

	#loss_all_epochs = []
	#acc_all_epochs = []
	val = []
	for ep in epochs:
		#loss_all_epochs.extend(info['loss'][ep])
		#acc_all_epochs.extend(info['acc'][ep])
		val.extend(info['val'][ep])
	#print(len(val), len(loss_all_epochs))
	number_of_batch_per_epoch = torch.as_tensor(val).shape[0] // n_epochs

	return (torch.argmax(torch.as_tensor(val)) // number_of_batch_per_epoch + 1).item()
	

def ConfusionMatrix(model, X, y, device, batchsize = 1000, threshold=0.50):
    
    if device == torch.device("cuda:0"):
        #print('Model moved to GPU.')
        model.to(device)

    X = NormalizeData(X)
    X_batch = torch.split(X, batchsize, dim=0)
    y_batch = torch.split(y, batchsize, dim=0)

    unique_class = y.unique()
    N_class = len(unique_class)
    

    
    confusion_matrix = np.zeros((N_class, N_class))

    for i, batch in enumerate(X_batch):     

        #X_batch = X[batch_inds[b], :, :]
        #y_batch = y[batch_inds[b]]

        if device == torch.device("cuda:0"):
            batch = batch.to(device)
        batch_labels = y_batch[i]

        out = F.softmax(model(batch), dim=1)
        #print(out, batch_labels)
        
        for j in range(batch_labels.shape[0]):
            predict_ind = torch.where(out[j, :] >= threshold)[0].to('cpu')
            
            predicted_classes = np.zeros(N_class)
            predicted_classes[predict_ind] = 1
            #print(predicted_classes)
            for n in range(len(predicted_classes)):
                if predicted_classes[n] == 1:
                    confusion_matrix[batch_labels[j], n] += 1
        #for n, class_index in enumerate(iclass):
        #    for m, itrue in enumerate(iclass):
        #        print(out)
                #predict_ind  = torch.where(out[:, ipredicted] >= threshold)[0].to('cpu')
                #true_ind = torch.where(batch_labels == itrue)[0].to('cpu')
                
                #N_ = np.isin(predict_ind, true_ind).sum()
                #print(pred_ind.shape, true_ind.shape, n, m)
                #N_false_neg = len(true_signal_ind) - N_true_pos
                
                #confusion_matrix[m, n] += N_predict_in_true
        
         # predicted signals
        #pred_noise_ind = torch.where(out[:, 0] > 1 - threshold)[0].to('cpu') # predicted noise
        
        #true_noise_ind = torch.where(batch_labels == 0)[0].to('cpu') # true noise
        
        #N_true_pos = np.isin(pred_signal_ind, true_signal_ind).sum()
        #N_false_neg = len(true_signal_ind) - N_true_pos
        
        #N_true_neg = np.isin(pred_noise_ind, true_noise_ind).sum()
        #N_false_pos = len(true_noise_ind) - N_true_neg
        
        #temp_matrix = np.array(
        #                    [[N_true_pos, N_false_neg], 
        #                    [N_false_pos, N_true_neg]]
        #                    )
        #confusion_matrix += temp_matrix
        

        #correct_predictions = torch.where(batch_labels == torch.argmax(out, dim=1))
        #incorrect_predictions = torch.where(batch_labels != torch.argmax(out, dim=1)) 
        #confusion_matrix[0, 0] += torch.where(batch_labels[correct_predictions] == 1)[0].shape[0]
        #confusion_matrix[1, 1] += torch.where(batch_labels[correct_predictions] == 0)[0].shape[0]
        #confusion_matrix[0, 1] += torch.where(batch_labels[incorrect_predictions] == 1)[0].shape[0]
        #confusion_matrix[1, 0] += torch.where(batch_labels[incorrect_predictions] == 0)[0].shape[0]
    #print(np.flip(np.flip(confusion_matrix, axis=0), axis=1))
    return confusion_matrix

def ROC(model, X, y, device, batchsize=1000):

    tpr = []
    fpr = []
    threshold = np.linspace(0.0, 1.0, 31)
    #threshold = [0.0, 0.5, 1.0]
    
    class_labels = y.unique().to('cpu')
    true_class_pop = np.zeros(len(class_labels))
    
    for n in range(len(class_labels)):
        true_class_pop[n] = len(torch.where(y == class_labels[n])[0])
    
    # compute the 1 vs all ROC curves for each class
    N_roc_curves = len(class_labels)
    
    roc_data = {'class_ind':class_labels, 'threshold': threshold, 'tpr_list': [], 'fpr_list': []}
    
    #print(N_roc_curves)
    for m in range(N_roc_curves):
        #print(n)
        tpr_n = []
        fpr_n = []
        for T in threshold:
            print(m, np.round(T, 4))
            matrix = ConfusionMatrix(model, X, y, device, batchsize = batchsize, threshold = T)
            
            N_matrix = matrix.shape[0]
            diag = np.diagonal(matrix)

            true_pos_n = diag[m] # roc curve class true positives
            
            total_neg_n = np.concatenate((true_class_pop[0:m], true_class_pop[m+1:])).sum() # total number of 'neg'
            
            false_pos_n = matrix[:, m].sum() - true_pos_n # fp for class n
            
            true_pos_rate_class_n = true_pos_n / true_class_pop[m]
            false_pos_rate_class_n = false_pos_n / total_neg_n
            
            
            tpr_n.append(true_pos_rate_class_n)
            fpr_n.append(false_pos_rate_class_n)
            

        roc_data['tpr_list'].append(tpr_n)
        roc_data['fpr_list'].append(fpr_n)
            # expresion is wierd since I want to ignore the noise for this calculation
            #tpr.append((np.trace(matrix) - matrix[-1, -1]) / (matrix[0:(matrix.shape[0]-1), :].sum()))
            #fpr.append((matrix[:, 0:(matrix.shape[1]-1)].sum() - np.trace(matrix) + matrix[-1, -1]) / matrix[:, 0:(matrix.shape[1]-1)].sum())
        
    #print(roc_data)
    return roc_data
    
def ROCFromMF(X):

    tpr = []
    fpr = []
    threshold = np.linspace(-10, 100, 51)
    
    for T in threshold:
       tpr.append(len(np.where(X['mf_scores'] >= T)[0]) / len(X['mf_scores']))
       fpr.append(len(np.where(X['mf_scores_noise'] >= T)[0]) / len(X['mf_scores_noise']))
    return tpr, fpr
    
