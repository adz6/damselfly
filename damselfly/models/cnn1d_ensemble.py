import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl

'''
    SparseCNNEnsemble Config

    config = {
        'nclass':
        'lin':
        'model_list': 
    }
'''

def EnsembleModelList(model_list):

    ensemble_models = []

    for model in model_list:
        model.linear_out = torch.nn.Identity()
        #for param in model.parameters():
        #    param.requires_grad_(False)
        ensemble_models.append(model)

    return torch.nn.ModuleList(ensemble_models)

class SparseCNNEnsemble(torch.nn.Module):
    def __init__(self, config):
        super(SparseCNNEnsemble, self).__init__()

        self.model_list = EnsembleModelList(config['model_list'])

        self.lin_list = config['lin']
        self.linear = StackLinear(*self.lin_list)
        self.linear_out = torch.nn.Linear(self.lin_list[1][-1], config['nclass'])

    def forward(self, x):

        x = x.clone()
        out_list = []
        for model in self.model_list:
            out_list.append(model(x))

        x = torch.cat((*out_list,), dim=1)
        x = self.linear(x)
        x = self.linear_out(x)

        return x