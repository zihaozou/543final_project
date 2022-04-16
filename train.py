from utils.ssim import MS_SSIM
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.save_load import save, load
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torch.nn import DataParallel
import model.gif_nerf as gifNerf
from model.mlp import MLP
from model.ffm import BasicFFM
from torch.utils.data.dataloader import DataLoader
from utils.dataset import gifDataset
from utils.gif_process import gifDecodeArr
from os.path import join
from shutil import make_archive
from hydra.utils import get_original_cwd
import hydra
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from os import mkdir
import os
from model.optic import backWarp, UNet
import random
import numpy as np
from itertools import product
import torch.nn as nn
from torch.nn.functional import mse_loss
#from model.gridencoder.grid import GridEncoder
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


@hydra.main(config_path="conf", config_name="config")
def main(cfg):

    make_archive('project', 'zip', get_original_cwd())

    # Load data
    gt, trainImage, XFull, yFull, XTrain, yTrain, fps, numFrames, H, W = gifDecodeArr(
        join(get_original_cwd(), *cfg.gif.data_path.split('/')), cfg.train.train_split, *cfg.gif.ofargs)
    trainLoader = DataLoader(gifDataset(
        XTrain, yTrain, None), batch_size=cfg.train.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valLoader = DataLoader(gifDataset(
        XFull, yFull, None), batch_size=cfg.train.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if cfg.train.target_fps_factor is not None:
        XSlow = torch.from_numpy(np.asarray(list(product(range(numFrames*cfg.train.target_fps_factor), range(
            H), range(W))))).float()
        XSlow[:, 0] /= float(numFrames*cfg.train.target_fps_factor)
        XSlow[:, 1] /= float(H)
        XSlow[:, 2] /= float(W)
    # Load model
    ffm = BasicFFM(cfg.model.ffm.mode, cfg.model.ffm.L,
                   cfg.model.ffm.num_input)
    mlp = MLP(cfg.model.mlp.num_inputs,
              cfg.model.mlp.num_outputs, cfg.model.mlp.num_neurons,
              cfg.model.mlp.depth, cfg.model.mlp.skip_list,
              cfg.model.mlp.body_acti, cfg.model.mlp.last_acti)
    mainDevice = f'cuda:{cfg.train.GPUIndex[0]}'
    model = getattr(gifNerf, cfg.model.nerf.type)(ffm, mlp).to(mainDevice)
    flowComp = UNet(6, 4).to(mainDevice)
    flowComp.load_state_dict(torch.load(
        join(get_original_cwd(), 'data', 'SuperSloMo.ckpt'), map_location='cpu')['state_dictFC'])
    flowComp.eval()
    FlowBackWarp = backWarp(H, W).to(mainDevice)
    alpha = torch.tensor([-0.1], requires_grad=True,
                         dtype=torch.float32, device=mainDevice)
    if len(cfg.train.GPUIndex) > 1:
        model = DataParallel(model, device_ids=cfg.train.GPUIndex)
    ssim = MS_SSIM(data_range=1.).to(mainDevice)
    # optimizer and scheduler
    optimizer = getattr(optim, cfg.train.optim.type)(
        [{'params': alpha}, {'params': model.parameters()}], **cfg.train.optim.kwargs)
    scheduler = getattr(lr_scheduler, cfg.train.lr_sche.type)(optimizer,
                                                              **cfg.train.lr_sche.kwargs)

    # logger
    mkdir('logs')
    mkdir('ckpts')
    logger = SummaryWriter('logs')

    # load ckpt
    if cfg.train.ckpt is not None:
        load(model, optimizer, scheduler, alpha, cfg.train.ckpt)
    logger.add_video('gt', torch.from_numpy(
        gt).permute((0, 3, 1, 2)).unsqueeze(0), None, fps)
    logger.add_video('trainset', torch.from_numpy(
        gt[::cfg.train.train_split, ...]).permute((0, 3, 1, 2)).unsqueeze(0), None, int(fps/cfg.train.train_split))
    # train
    for e in (bar := tqdm(range(cfg.train.num_epoches))):
        model.train()
        epochLoss = 0
        for b, batch in enumerate(trainLoader):
            # train
            optimizer.zero_grad()
            if isinstance(model, DataParallel):
                loss = model.module.trainingStep(model, batch, mainDevice)
            else:
                loss = model.trainingStep(model, batch, mainDevice)
            loss.backward()
            optimizer.step()
            logger.add_scalar('train/loss', loss.item(), b+e*len(trainLoader))
            bar.set_description(f'{b}/{len(trainLoader)}:{loss.item():.3e}')
            epochLoss += loss.item()

        epochLoss /= len(trainLoader)
        logger.add_scalar('train/epoch loss', epochLoss, e)
        #if e >= int(cfg.train.num_epoches*0.1):
        if True:
            selectStep = 1./float(trainImage.shape[0])
            tIndex = 0
            for ind in np.arange(0, 1-2*selectStep, selectStep):
                stratify = np.linspace(
                    ind, ind+selectStep, cfg.train.stra_size)
                for blInd in range(len(stratify)-1):
                    bl = stratify[blInd]
                    bh = stratify[blInd+1]
                    t = random.uniform(bl, bh)
                    X = torch.from_numpy(np.asarray(list(product([t], range(
                        H), range(W))))).float()
                    X[:, 1] /= float(H)
                    X[:, 2] /= float(W)
                    y = torch.from_numpy(
                        trainImage[tIndex:tIndex+2, ...]).float().permute((0, 3, 1, 2))
                    batch = (X, y)

                    if isinstance(model, DataParallel):
                        midGrad = model.module.interframeTrainStep(
                            model, flowComp, FlowBackWarp, batch, (t-ind)/selectStep, alpha, cfg.train.batch_size, ssim, mainDevice)
                    else:
                        midGrad = model.interframeTrainStep(
                            model, flowComp, FlowBackWarp, batch, (t-ind)/selectStep, alpha, cfg.train.batch_size, ssim, mainDevice)
                    midGrad = midGrad.permute((0, 2, 3, 1)).reshape(-1, 3)
                    #y = y.to(mainDevice).permute((0, 2, 3, 1)).reshape(-1, 3)
                    XChunk = torch.split(X, cfg.train.batch_size)
                    #yChunk = torch.split(y, cfg.train.batch_size)
                    gradChunk = torch.split(midGrad, cfg.train.batch_size)
                    optimizer.zero_grad()
                    for XPiece, gradPiece in zip(XChunk, gradChunk):
                        XPiece = XPiece.to(mainDevice)
                        predPiece = model(XPiece)
                        #loss = mse_loss(predPiece, yPiece, reduction='none')
                        predPiece.backward(gradPiece)
                    optimizer.step()
                tIndex += 1
        model.eval()
        reconLst = []
        for b, batch in enumerate(valLoader):
            # eval
            if isinstance(model, DataParallel):
                pred = model.module.valStep(model, batch[0], mainDevice)
            else:
                pred = model.valStep(model, batch[0], mainDevice)
            reconLst.append(pred)
        recon = torch.cat(reconLst, dim=0).reshape((numFrames, H, W, 3))
        logger.add_video(
            'val/recon', recon.permute((0, 3, 1, 2)).unsqueeze(0), e, fps)
        reconPSNR = 0
        for i in range(recon.shape[0]):
            reconPSNR += psnr(gt[i, ...], recon[i, ...].numpy(), data_range=1)
        reconPSNR /= recon.shape[0]
        logger.add_scalar('val/psnr', reconPSNR, e)
        if cfg.train.target_fps_factor is not None:
            reconLst = []
            XChunk = torch.split(XSlow, cfg.train.batch_size)
            for b, batch in enumerate(XChunk):
                # eval
                if isinstance(model, DataParallel):
                    pred = model.module.valStep(model, batch, mainDevice)
                else:
                    pred = model.valStep(model, batch, mainDevice)
                reconLst.append(pred)
            recon = torch.cat(reconLst, dim=0).reshape(
                (numFrames*cfg.train.target_fps_factor, H, W, 3))
            logger.add_video(
                'val/slow motion', recon.permute((0, 3, 1, 2)).unsqueeze(0), e, fps)
        scheduler.step()
        save(model, optimizer, scheduler, alpha, f'ckpts/epoch{e}.pt')


if __name__ == '__main__':
    main()
