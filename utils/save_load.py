import torch
from torch.nn import DataParallel


def save(model, optimizer, scheduler, path):
    if isinstance(model, DataParallel):
        modelDict = model.module.state_dict()
    else:
        modelDict = model.state_dict()
    saveDict = {'model': modelDict, 'optimizer': optimizer.state_dict(
    ), 'scheduler': scheduler.state_dict()}
    torch.save(saveDict, path)


def load(model, optimizer, scheduler, path):
    loadDict = torch.load(path, map_location=model.device)
    if isinstance(model, DataParallel):
        model.module.load_state_dict(loadDict['model'])
    else:
        model.load_state_dict(loadDict['model'])
    optimizer.load_state_dict(loadDict['optimizer'])
    scheduler.load_state_dict(loadDict['scheduler'])
