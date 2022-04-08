import torch
from skimage.metrics import peak_signal_noise_ratio
import hydra
from hydra.utils import get_original_cwd
from shutil import make_archive
from os.path import join
from utils.gif_process import gifDecodeArr
from utils.dataset import gifDataset
from torch.utils.data.dataloader import DataLoader
from model.ffm import BasicFFM
from model.mlp import MLP
import model.gif_nerf as gifNerf
from torch.nn import DataParallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.save_load import save, load
from tqdm import tqdm, trange


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    make_archive('project.zip', 'zip', get_original_cwd())

    # Load data
    XFull, yFull, XTrain, yTrain, opTrain, fps = gifDecodeArr(
        cfg.gif.data_path, cfg.train.train_split, cfg.gif.ofargs)
    trainLoader = DataLoader(gifDataset(
        XTrain, yTrain, opTrain), batch_size=cfg.train.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valLoader = DataLoader(gifDataset(
        XFull, yFull, None), batch_size=cfg.train.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Load model
    ffm = BasicFFM(cfg.model.ffm.mode, cfg.model.ffm.L,
                   cfg.model.ffm.num_input)
    mlp = MLP(cfg.model.mlp.num_inputs,
              cfg.model.mlp.num_outputs, cfg.model.mlp.num_neurons,
              cfg.model.mlp.depth, cfg.model.mlp.skip_list,
              cfg.model.mlp.body_acti, cfg.model.mlp.last_acti)
    mainDevice = f'cuda:{cfg.train.GPUIndex[0]}'
    model = getattr(gifNerf, cfg.model.nerf.type)(ffm, mlp).to(mainDevice)
    if len(cfg.train.GPUIndex) > 1:
        model = DataParallel(model, device_ids=cfg.train.GPUIndex)

    # optimizer and scheduler
    optimizer = getattr(optim, cfg.train.optim.type)(
        model.parameters(), cfg.train.optim.kwargs)
    scheduler = getattr(optimizer, cfg.train.lr_sche.type)(
        cfg.train.lr_sche.kwargs)

    # load ckpt
    if cfg.train.ckpt is not None:
        load(model, optimizer, scheduler, cfg.train.ckpt)

    # train
    for e in (bar := tqdm(range(cfg.train.num_epoches))):
        model.train()
        for b, batch in enumerate(trainLoader):
            # train
            pass

        model.eval()
        for b, batch in enumerate(valLoader):
            # eval
            pass
