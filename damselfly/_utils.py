# -*- coding: utf-8 -*-

#import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#import pickle as pkl
#import os
#import time
#import damselfly as df
#import matplotlib.pyplot as plt

'''
def TrainModelAutoencoder(model, datapath, device, epochs, batchsize, learning_rate, model_save_path, epochs_per_checkpoint=1):

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
    val_loss_per_epoch = {}

    # define loss function and optimizer
    criterion = torch.nn.MSELoss()


    #optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=15, threshold=0.0001)

    for ep in range(epochs):

        running_loss = 0.0
        train_loss.update({ep: []})
        val_loss_per_epoch.update({ep: []})

        t1 = time.perf_counter_ns()
        
        nbatch = 1
        for batch, labels in train_dataloader:
            if device == torch.device("cuda:0"):
                batch = batch.to(device)
                #labels = labels.to(device)

            optimizer.zero_grad()

            output = model(batch)

            loss = criterion(output, batch) # loss computed using input as output
            loss.backward()

            optimizer.step()

            train_loss[ep].append(loss.item())

            # monitor training
            print('| %d | %d | %.3f |' % ((ep + 1), nbatch, loss.item()))
            nbatch += 1

        t2 = time.perf_counter_ns()
        print(f'Epoch time: {np.round((t2-t1)*1e-9, 2)} s')

        
        #torch.cuda.empty_cache()
        # compute validation accuracy after 1 epoch
        with torch.no_grad():
            for batch, labels in val_dataloader:
                #batch_labels = torch.split(val_labels, batchsize, dim=0)[i]

                if device == torch.device("cuda:0"):
                    batch = batch.to(device)
                    #labels = labels.to(device)
                
                val_out = model(batch)
                val_loss = criterion(val_out, batch)
                val_loss_per_epoch[ep].append(val_loss)

        #scheduler.step(torch.mean(torch.as_tensor(val_loss_per_epoch[ep])))

        print('Completed epoch %d. Mean validation loss: %.3f' % ((ep + 1), torch.mean(torch.as_tensor(val_loss_per_epoch[ep]))))

        # save model checkpoint
        if ep % epochs_per_checkpoint == (epochs_per_checkpoint - 1):
            torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch{ep + 1}.pth'))


        info = {'loss': train_loss, 'val': val_loss_per_epoch}

        with open(os.path.join(model_save_path, 'info.pkl'), 'wb') as outfile:
            pkl.dump(info, outfile)

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

    print('Loading data')
    # use norm_mode = 'flat' and flat_norm_val = 3.8e-8 (current default value) for pca projected data
    train_data = df.data.DFDataset(datapath, 'train', norm=True, )
    val_data = df.data.DFDataset(datapath, 'val', norm=True, )

    print(train_data.data.shape)
    

    train_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(train_data.data, train_data.label),
                                                    batchsize,
                                                    shuffle=True, 
                                                    )
    val_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(val_data.data, val_data.label),
                                                    batchsize,
                                                    shuffle=True,
                                                    )

    # track training
    train_loss = {}
    train_accuracy = {}
    val_accuracy_per_epoch = {}

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight = bce_weight, reduction = 'mean')
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor = 0.1, patience=40, threshold=0.01)

    
    print('Training starting')

    for ep in range(epochs):

        running_loss = 0.0
        train_loss.update({ep: []})
        train_accuracy.update({ep: []})
        val_accuracy_per_epoch.update({ep: []})

        t1 = time.perf_counter_ns()
        
        nbatch = 1
        for batch, labels in train_dataloader:
            #batch_labels = torch.split(train_labels, batchsize, dim=0)[i]
            
            if device == torch.device("cuda:0"):
                batch = batch.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            #print(labels)
            
            # plot training data for debugging
            '''
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)

            #print(labels.to('cpu').numpy())
            plot_signal = False
            plot_noise = False
            for i, l in enumerate(labels.cpu().numpy()):

                if l == 1 and not plot_signal:
                    ax.plot(batch[i, 0, :].cpu(), label=l)
                    plot_signal = True
                    print(batch[i, 0, :].cpu().numpy().sum())
                if l == 0 and not plot_noise:

                    ax.plot(batch[i, 0, :].cpu(), label=l)
                    plot_noise = True
                    print(batch[i, 0, :].cpu().numpy().sum())

            ax.legend()
            plt.savefig('test_pca_data')

            input('fig')
            

            labels = labels.to("cuda:0")
            '''


            output = model(batch)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss[ep].append(loss.cpu().item())

            batch_accuracy = CalculateAccuracy(output, labels)
            train_accuracy[ep].append(batch_accuracy.cpu())

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
            val_accuracy_per_epoch[ep].append(val_acc.cpu())

        scheduler.step(torch.mean(torch.as_tensor(val_accuracy_per_epoch[ep])))
        
        print('Completed epoch %d. Mean validation accuracy: %.3f. lr = %3f' % ((ep + 1), torch.mean(torch.as_tensor(val_accuracy_per_epoch[ep])), optimizer.param_groups[0]['lr']))

        # save model checkpoint
        if ep % epochs_per_checkpoint == (epochs_per_checkpoint - 1):
            torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch{ep + 1}.pth'))


        info = {'loss': train_loss, 'acc': train_accuracy, 'val': val_accuracy_per_epoch}

        with open(os.path.join(model_save_path, 'info.pkl'), 'wb') as outfile:
            pkl.dump(info, outfile)

def TrainRegressionModel(model, datapath, device, epochs, batchsize, learning_rate, model_save_path, 
                epochs_per_checkpoint=5,):
                                
    if device == torch.device("cuda:0"):
        print('Model moved to GPU')
        model.to(device)

    print('Loading data')
    # use norm_mode = 'flat' and flat_norm_val = 3.8e-8 (current default value) for pca projected data
    train_data = df.data.DFDataset(datapath, 'train', norm=True, label_type='float')
    val_data = df.data.DFDataset(datapath, 'val', norm=True, label_type='float')

    

    train_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(train_data.data, train_data.label),
                                                    batchsize,
                                                    shuffle=True, 
                                                    )
    val_dataloader = torch.utils.data.DataLoader(
                                                    torch.utils.data.TensorDataset(val_data.data, val_data.label),
                                                    batchsize,
                                                    shuffle=True,
                                                    )

    # track training
    train_loss = {}
    val_loss_per_epoch = {}

    # define loss function and optimizer

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=100, threshold=0.01)

    
    print('Training starting')

    for ep in range(epochs):

        running_loss = 0.0
        train_loss.update({ep: []})
        val_loss_per_epoch.update({ep: []})

        t1 = time.perf_counter_ns()
        
        nbatch = 1
        for batch, labels in train_dataloader:
            #batch_labels = torch.split(train_labels, batchsize, dim=0)[i]
            
            if device == torch.device("cuda:0"):
                batch = batch.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            #print(labels)
            
            # plot training data for debugging
            '''
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(1,1,1)

            #print(labels.to('cpu').numpy())
            plot_signal = False
            plot_noise = False
            for i, l in enumerate(labels.cpu().numpy()):

                if l == 1 and not plot_signal:
                    ax.plot(batch[i, 0, :].cpu(), label=l)
                    plot_signal = True
                    print(batch[i, 0, :].cpu().numpy().sum())
                if l == 0 and not plot_noise:

                    ax.plot(batch[i, 0, :].cpu(), label=l)
                    plot_noise = True
                    print(batch[i, 0, :].cpu().numpy().sum())

            ax.legend()
            plt.savefig('test_pca_data')

            input('fig')
            

            labels = labels.to("cuda:0")
            '''
            #print(batch.shape, labels.shape)


            output = model(batch)

            #print(output.squeeze().shape)
            loss = criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss[ep].append(loss.cpu().item())

            #batch_accuracy = CalculateAccuracy(output, labels)
            #train_accuracy[ep].append(batch_accuracy.cpu())

            # monitor training
            print('| %d | %d | %.3f |' % ((ep + 1), nbatch, loss.item()))
            nbatch += 1

        t2 = time.perf_counter_ns()

        print(f'Epoch time: {np.round((t2-t1)*1e-9, 2)} s')
        # compute validation loss after 1 epoch


        with torch.no_grad():
            for batch, labels in val_dataloader:
                #batch_labels = torch.split(val_labels, batchsize, dim=0)[i]

                if device == torch.device("cuda:0"):
                    batch = batch.to(device)
                    labels = labels.to(device)
                
                val_out = model(batch)
                val_loss = criterion(val_out.squeeze(), labels)

                val_loss_per_epoch[ep].append(val_loss.cpu())

            #scheduler.step(torch.mean(torch.as_tensor(val_loss_per_epoch[ep])))
            
            print('Completed epoch %d. Mean validation loss: %.3f. lr = %3f' % ((ep + 1), torch.mean(torch.as_tensor(val_loss_per_epoch[ep])), optimizer.param_groups[0]['lr']))

        # save model checkpoint
        if ep % epochs_per_checkpoint == (epochs_per_checkpoint - 1):
            torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch{ep + 1}.pth'))


        info = {'loss': train_loss, 'val': val_loss_per_epoch}

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

def CalculateAccuracy(output, labels):

	output_prob = F.softmax(output, dim=1)

	most_likely_class = torch.argmax(output_prob, dim=1)

	most_likely_class_matches_label = torch.as_tensor(most_likely_class == labels, dtype=torch.float)

	return torch.mean(most_likely_class_matches_label)

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

def ConfusionMatrix(model, X, y, device, batchsize = 1000, threshold=0.50):
    
    if device == torch.device("cuda:0"):
        #print('Model moved to GPU.')
        model.to(device)

    dataloader = torch.utils.data.DataLoader(
                                            torch.utils.data.TensorDataset(X, y),
                                            batchsize,
                                            shuffle=True
                                            )
    unique_class = y.unique()
    N_class = len(unique_class)
    
    confusion_matrix = np.zeros((N_class, N_class))

    for batch, labels in dataloader:
            
            if device == torch.device("cuda:0"):
                batch = batch.to(device)
            
            out = F.softmax(model(batch), dim=1)
            
            for j in range(labels.shape[0]):
                predict_ind = torch.where(out[j, :] >= threshold)[0].to('cpu')
            
                predicted_classes = np.zeros(N_class)
                predicted_classes[predict_ind] = 1
                #print(predicted_classes)
                for n in range(len(predicted_classes)):
                    if predicted_classes[n] == 1:
                        confusion_matrix[labels[j], n] += 1
                        

    return confusion_matrix

def ROC(model, X, y, device, batchsize=1000):

    tpr = []
    fpr = []
    threshold = np.linspace(0.0, 1.0, 41)
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

    return roc_data
    
def ROCFromMF(X):

    tpr = []
    fpr = []
    threshold = np.linspace(-10, 100, 51)
    
    for T in threshold:
       tpr.append(len(np.where(X['mf_scores'] >= T)[0]) / len(X['mf_scores']))
       fpr.append(len(np.where(X['mf_scores_noise'] >= T)[0]) / len(X['mf_scores_noise']))
    return tpr, fpr
    
'''