import torch
import torch.nn.utils.prune as prune
import statistics

def PruneModel(model, proportion):

    sparsity_percent = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
            sparsity_percent.append(100 * float(torch.sum(module.weight==0))/float(module.weight.nelement()))
            print(type(module), 100 * float(torch.sum(module.weight==0))/float(module.weight.nelement()))

    print(f'Pruning starting. {round(statistics.mean(sparsity_percent), 1)}% Average Sparsity')

    module_tups = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    #for module, _ in module_tups:
    #    prune.remove(module, 'weight')


    sparsity_percent = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.Linear):
            sparsity_percent.append(100 * float(torch.sum(module.weight==0))/float(module.weight.nelement()))
            print(type(module), 100 * float(torch.sum(module.weight==0))/float(module.weight.nelement()))

    print(f'Pruning complete. {round(statistics.mean(sparsity_percent), 1)}% Average Sparsity')
    return model
