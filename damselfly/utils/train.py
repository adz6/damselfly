import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import math 
from pathlib import Path
import random
from damselfly.utils import prune
from damselfly.data import augmentation


def _batch_acc(output, labels):
    torch_max = torch.max(output.cpu(), dim=-1)
    accuracy = (torch_max[1] == labels.cpu()).sum() / len(labels.cpu())
    return accuracy

def _save_checkpoint(config, model_state, optimizer_state, epoch, loss, acc, val_acc):

    save_dict = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'model_args': config['model_args'],
        'epochs': epoch,
        'loss': loss,
        'acc': acc,
        'val_acc': val_acc
    }

    torch.save(save_dict, config['checkpoint'])

def LoadCheckpoint(model_class, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=torch.device('cpu')
        )

    model = model_class(*checkpoint['model_args'])
    model.load_state_dict(checkpoint['model_state_dict'])

    #optimizer = checkpoint['optimizer_class'](*checkpoint['opt_args'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epochs']
    loss = checkpoint['loss']
    acc = checkpoint['acc']
    val_acc = checkpoint['val_acc']

    return (model, epoch, loss, acc, val_acc)

'''
Train Model 

config = {
    'batchsize': 2,
    'epochs': 1,
    'checkpoint_epochs': 25,
    'checkpoint': path,
    'initial_epoch': 0,
    'loss': [],
    'acc': [],
    'val_acc': [],
}
'''

def TrainModel(rank, model, optimizer, loss_fcn, train_data, val_data, config, **kwargs):

    torch.cuda.set_device(rank)
    print(rank)

    model = model.cuda(rank)

    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])
    val_dataset = torch.utils.data.TensorDataset(val_data[0], val_data[1])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        config['batchsize'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        config['batchsize'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    objective = loss_fcn.cuda(rank)
    ep_count = 0
    for ep in range(config['epochs']):
        batch_count = 0
        ep_acc = []
        ep_loss = []
        for batch, labels in train_loader:
            
            if 'noise_gen' in kwargs:
                batch = kwargs['noise_gen'](batch, *kwargs['noise_gen_args'])

            if 'data_norm' in kwargs:
                batch = kwargs['data_norm'](batch)

            if 'circular_shift' in config:
                shift = random.choice(config['circular_shift'])
                if abs(shift) > 0:
                    batch = augmentation.CircularShift(batch, shift)

            optimizer.zero_grad()
            output = model(batch.cuda(rank))
            loss = loss_fcn(output, labels.cuda(rank))
            loss.backward()

            optimizer.step()
            batch_acc = _batch_acc(output, labels)

            ep_acc.append(batch_acc.item())
            ep_loss.append(loss.item())

            config['loss'].append([ep_count, batch_count, loss.item()])
            config['acc'].append([ep_count, batch_count, batch_acc.item()])
            batch_count += 1

        # validation check
        with torch.no_grad():
            batch_count = 0
            val_acc = []
            for batch, labels in val_loader:
                if 'noise_gen' in kwargs:
                    batch = kwargs['noise_gen'](batch, *kwargs['noise_gen_args'])
                if 'data_norm' in kwargs:
                    batch = kwargs['data_norm'](batch)
                output = model(batch.cuda(rank))
                val_acc.append(_batch_acc(output, labels))
                config['val_acc'].append([ep_count, batch_count, val_acc])
                batch_count += 1

        print(f'|  {ep + 1}  |  loss = {round(float(np.mean(ep_loss)), 5)}  |  acc = {round(float(np.mean(ep_acc)), 5)}  | val. acc = {round(float(np.mean(val_acc)), 5)}', flush=True)
        ep_count += 1

        
        if ep_count % config['checkpoint_epochs'] == config['checkpoint_epochs'] - 1:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            _save_checkpoint(
                config,
                model_state_dict,
                optimizer_state_dict,
                config['initial_epoch'] + ep_count,
                config['loss'],
                config['acc'],
                config['val_acc']
                )
    
    return model.cpu()

def IterativePruneModel(rank, model, optimizer, loss_fcn, train_data, val_data, config, **kwargs):

    torch.cuda.set_device(rank)
    print(rank)

    model = model.cuda(rank)

    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])
    val_dataset = torch.utils.data.TensorDataset(val_data[0], val_data[1])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        config['batchsize'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        config['batchsize'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    for iteration in range(config['prunes']):
    
        objective = loss_fcn.cuda(rank)
        ep_count = 0
        for ep in range(config['epochs']):
            batch_count = 0
            ep_acc = []
            ep_loss = []
            for batch, labels in train_loader:
                
                if 'noise_gen' in kwargs:
                    batch = kwargs['noise_gen'](batch, *kwargs['noise_gen_args'])

                if 'data_norm' in kwargs:
                    batch = kwargs['data_norm'](batch)

                optimizer.zero_grad()
                output = model(batch.cuda(rank))
                loss = loss_fcn(output, labels.cuda(rank))
                loss.backward()

                optimizer.step()
                batch_acc = _batch_acc(output, labels)

                ep_acc.append(batch_acc.item())
                ep_loss.append(loss.item())

                config['loss'].append([ep_count, batch_count, loss.item()])
                config['acc'].append([ep_count, batch_count, batch_acc.item()])
                batch_count += 1

            # validation check
            with torch.no_grad():
                batch_count = 0
                val_acc = []
                for batch, labels in val_loader:
                    if 'noise_gen' in kwargs:
                        batch = kwargs['noise_gen'](batch, *kwargs['noise_gen_args'])
                    if 'data_norm' in kwargs:
                        batch = kwargs['data_norm'](batch)
                    output = model(batch.cuda(rank))
                    val_acc.append(_batch_acc(output, labels))
                    config['val_acc'].append([ep_count, batch_count, val_acc])
                    batch_count += 1

            print(f'|  {ep + 1}  |  loss = {round(float(np.mean(ep_loss)), 5)}  |  acc = {round(float(np.mean(ep_acc)), 5)}  | val. acc = {round(float(np.mean(val_acc)), 5)}')
            ep_count += 1

            
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        # checkpoint name format: name_prune0.tar
        
        config['checkpoint'] = config['checkpoint'].split('_')[0] + f'_prune{iteration}.tar'
        _save_checkpoint(
            config,
            model_state_dict,
            optimizer_state_dict,
            config['initial_epoch'] + ep_count,
            config['loss'],
            config['acc'],
            config['val_acc']
            )
        
        prune_fraction = 1 - (1 - 0.2) ** (iteration + 1)
        print(f'Iteration: {iteration + 1}, Global Sparsity: {prune_fraction}')
        model = prune.PruneModel(model, 0.2)
        model = model.cuda(rank)

        #sparsity_percent = []
        #for module in model.modules():
        #    if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
        #        sparsity_percent.append(100 * float(torch.sum(module.weight==0))/float(module.weight.nelement()))
        #        print(type(module), 100 * float(torch.sum(module.weight==0))/float(module.weight.nelement()))
    
    return model.cpu()

def EvalModel(rank, model_class, checkpoint, data, config, **kwargs):

    torch.cuda.set_device(rank)
    print(rank)

    model, _, _, _, _ = LoadCheckpoint(model_class, checkpoint)
    model = model.eval()
    model = model.cuda(rank)

    #if 'circular_shift' in kwargs:
    #    data[0] = augmentation.CircularShift(data[0], kwargs['circular_shift'])

    dataset = torch.utils.data.TensorDataset(data[0], data[1])

    loader = torch.utils.data.DataLoader(
        dataset, 
        config['batchsize'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    outputs = {}
    with torch.no_grad():
        for ep in range(config['epochs']):
            batch_count = 0
            ep_out = []
            ep_acc = []
            for batch, labels in loader:
                
                if 'noise_gen' in kwargs:
                    batch = kwargs['noise_gen'](batch, *kwargs['noise_gen_args'])
                if 'data_norm' in kwargs:
                    batch = kwargs['data_norm'](batch)
                if 'circular_shift' in config:
                    shift = random.choice(config['circular_shift'])
                    if abs(shift) > 0:
                        batch = augmentation.CircularShift(batch, shift)

                output = model(batch.cuda(rank))
                ep_out.extend(output.cpu().numpy())
                batch_acc = _batch_acc(output, labels)
                ep_acc.append(batch_acc.item())

            ep_out = np.array(ep_out)
            outputs[str(ep)] = ep_out

            print(f'|  {ep + 1}  |  acc = {round(float(np.mean(ep_acc)), 5)}  |', flush=True)

    np.savez(config['output'], **outputs)
    