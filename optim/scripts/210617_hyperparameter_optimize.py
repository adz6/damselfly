import optuna 
import torch
import numpy as np
import damselfly as df
import gc
import os

EPOCHS = 20
BATCHSIZE = 2000
DEVICE = torch.device("cuda")
DATASET = '210617_df2.h5'
NCH = 3
NCLASS = 2
NINPUT = 8192
DAMSELPATH = '/home/az396/project/damselfly'
STUDYNAME = '210619_optim_cnn_model'
#MODEL = df.models.df_conv6_fc2_nclass_nch(NCLASS, NCH)



def Objective(trial):

    model = DefineModel(trial)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(DEVICE)
    
    #w0 = trial.suggest_float(name='w0', low=1.0, high=10.0, step=0.5)
    #w1 = trial.suggest_float(name='w1', low=1.0, high=10.0, step=1.)
    
    w0 = 5.0
    w1 = 1.0
    
    bce_weight = torch.tensor(
                                np.array([w0, w1]),
                                 device=DEVICE, dtype=torch.float
                                )
    
    train_dataloader, val_dataloader = GetDataLoaders()
    
    criterion = torch.nn.CrossEntropyLoss(weight = bce_weight, reduction = 'mean')
    
    #learning_rate = trial.suggest_float(name='lr', low=1e-4, high=1e-2, log=True)
    learning_rate = 5e-3
    #print(learning_rate)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    
    for ep in range(EPOCHS):
        
        model.train()

        nbatch = 1
        for batch, labels in train_dataloader:

            batch, labels = batch.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            output = model(batch)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_acc_epoch = []
        for batch, labels in val_dataloader:

            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            
            val_out = model(batch)
            
            val_acc = df.utils.CalculateAccuracy(val_out, labels)
            val_acc_epoch.append(val_acc)
            
        trial.report(torch.mean(torch.as_tensor(val_acc_epoch)), ep)
        if ep % 5 == 4:
            
            print(ep + 1, torch.mean(torch.as_tensor(val_acc_epoch)).item())
        if trial.should_prune():
            #gc.collect()
            #torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
                

    return torch.mean(torch.as_tensor(val_acc_epoch))
    
def DefineModel(trial):
    
    
    # generate the number of convolutional and max pool layers
    nconv = trial.suggest_int(name='nconv', low=3, high=12, step=1)
    nmax = trial.suggest_int(name='nmax', low=2, high=3, step=1)
    
    #split the convolutional layers with the max pool layers
    conv_split = np.array_split(np.arange(0, nconv, 1), nmax)
    
    # generate the convolutional layer filter numbers
    nconv_filters = []
    conv_kernel_sizes = []
    conv_dilations = []
    maxpool_kernel_sizes = []
    
    for n in range(nmax):
        if n == 0:
            nconv_filters.append(trial.suggest_int(name=f'conv_f_{n}', low =  8, high = 32 , step=4))
            
            conv_kernel_sizes.append(trial.suggest_int(name=f'conv_kernel_size{n}', low = 8, high = 16, step=4))
            
            conv_dilations.append(trial.suggest_int(name=f'dilation{n}', low=1, high=5, step=2))
            #conv_dilations.append(1)
            
            maxpool_kernel_sizes.append(trial.suggest_int(name=f'maxpool_kernel_size{n}', low = 8, high = 16, step=4))
            
        else:
            nconv_filters.append(2 * nconv_filters[n-1]) # filters increase by a factor of 2 from first hyperparameter
            
            conv_kernel_sizes.append(conv_kernel_sizes[n-1]//2)
            
            #conv_dilations.append(trial.suggest_int(name=f'dilation{n}', low=1, high=conv_dilations[n-1] , step=4))
            conv_dilations.append(conv_dilations[n-1])
            
            maxpool_kernel_sizes.append(maxpool_kernel_sizes[n-1]//2)
    

    # generate the number of linear and dropout layers
    nlinear = trial.suggest_int(name='nlin', low=1, high=3, step=1)
    
    # generate the linear layer sizes and dropout rates
    linear_sizes = []
    dropout_rates = [0.5, 0.5, 0.5]
    for n in range(nlinear):
        if n == 0:
            linear_sizes.append(trial.suggest_int(name=f'linsize{n}', low=64, high=512, step=32))
        else:
            linear_sizes.append(linear_sizes[n-1] // 2) # force the linear size to decrease
            
        #dropout_rates.append(trial.suggest_float(name=f'pdrop{n}', low=0.2, high=0.6, step=0.2))
        
    # build the convolutional layer parameter list

    conv_list = []
    
    for i, convmax_block in enumerate(conv_split): # for each conv-maxpool block
        conv_list.append([[], [], [], []])
        for j, conv_layer in enumerate(convmax_block):
            #print(i,j, conv_list[i])
            
            # handle the weirdness that happens for the input convolutional filter sizes
            if i == 0 and j == 0:
                conv_list[i][0].append(NCH)
            elif j == 0 and i != 0:
                conv_list[i][0].append(nconv_filters[i-1]) # the number of inputs for the first layer of the next block needs to match the number of filters of the previous block
            else:
                conv_list[i][0].append(nconv_filters[i])
                
            conv_list[i][1].append(nconv_filters[i])
            conv_list[i][2].append(conv_kernel_sizes[i])
            conv_list[i][3].append(conv_dilations[i])
        conv_list[i].append(maxpool_kernel_sizes[i])
    #print(conv_list)
   

                
    # build the linear layer parameter list
    linear_list = [[], [], []]
    for ilinear in range(nlinear):
        if ilinear == 0:
            linear_list[0].append(df.models.CalcConvMaxpoolOutputSize(conv_list, NCH, NINPUT))
        else:
            linear_list[0].append(linear_sizes[ilinear - 1])
        linear_list[1].append(linear_sizes[ilinear])
        linear_list[2].append(dropout_rates[ilinear])
    

    
    model = df.models.DFCNN(NCLASS, NCH, conv_list, linear_list)
    #print(model)
    return model
    
    #return df.models.df_conv6_fc2_nclass_nch(NCLASS, NCH)
    
    
def GetDataLoaders():

    path2DF2 = f'/home/az396/project/damselfly/data/datasets/{DATASET}'
    traindata = df.data.DFDataset(path2DF2, 'train')
    valdata = df.data.DFDataset(path2DF2, 'val')
    
    train_dataloader = torch.utils.data.DataLoader(
                                            torch.utils.data.TensorDataset(traindata.data, traindata.label),
                                            BATCHSIZE,
                                            shuffle=True
                                            )
    val_dataloader = torch.utils.data.DataLoader(
                                            torch.utils.data.TensorDataset(valdata.data, valdata.label),
                                            BATCHSIZE,
                                            shuffle=True
                                            )
    return train_dataloader, val_dataloader
    
    
if __name__ == "__main__":

    storage_name = f'sqlite:///{STUDYNAME}.db'
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials = 9, n_warmup_steps=10)
    
    study = optuna.create_study(direction="maximize", storage=storage_name, study_name=STUDYNAME, load_if_exists=True, pruner=pruner)  # 'maximize' because objective function is returning accuracy
    #study = optuna.create_study(direction="minimize")  # 'minimize' because objective function is returning loss
    study.optimize(Objective, n_trials=128, gc_after_trial=False)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    #with open(os.path.join(DAMSELPATH, f'optim/results/{STUDYNAME}.pkl'), 'wb') as outfile:
    #    pkl.dump(study, outfile)
    
