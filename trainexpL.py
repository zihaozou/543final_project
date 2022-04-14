from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
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
import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


@hydra.main(config_path="conf", config_name="configexpL")
def main(cfg):
    make_archive('project', 'zip', get_original_cwd())

    # Load data
    for i in range(len(cfg.model.ffm.L)):
      gt, XFull, yFull, XTrain, yTrain, opTrain, fps, numFrames, H, W = gifDecodeArr(
          join(get_original_cwd(), *cfg.gif.data_path.split('/')), cfg.train.train_split, *cfg.gif.ofargs)
      trainLoader = DataLoader(gifDataset(
          XTrain, yTrain, None), batch_size=cfg.train.batch_size, shuffle=True, num_workers=2, pin_memory=True)
      valLoader = DataLoader(gifDataset(
          XFull, yFull, None), batch_size=cfg.train.batch_size, shuffle=False, num_workers=2, pin_memory=True)

      # Load model
      ffm = BasicFFM(cfg.model.ffm.mode, cfg.model.ffm.L[i],
                    cfg.model.ffm.num_input)
      mlp = MLP(cfg.model.mlp.num_inputs[i],
                cfg.model.mlp.num_outputs, cfg.model.mlp.num_neurons,
                cfg.model.mlp.depth, cfg.model.mlp.skip_list,
                cfg.model.mlp.body_acti, cfg.model.mlp.last_acti)
      mainDevice = f'cuda:{cfg.train.GPUIndex[0]}'
      model = getattr(gifNerf, cfg.model.nerf.type)(ffm, mlp).to(mainDevice)
      if len(cfg.train.GPUIndex) > 1:
          model = DataParallel(model, device_ids=cfg.train.GPUIndex)

      # optimizer and scheduler
      optimizer = getattr(optim, cfg.train.optim.type)(
          model.parameters(), **cfg.train.optim.kwargs)
      scheduler = getattr(lr_scheduler, cfg.train.lr_sche.type)(optimizer,
                                                                **cfg.train.lr_sche.kwargs)

      # logger
      mkdir(f'logs_L={cfg.model.ffm.L[i]}')
      mkdir(f'ckpts_L={cfg.model.ffm.L[i]}')
      logger = SummaryWriter(f'logs_L={cfg.model.ffm.L[i]}')

      # load ckpt
      if cfg.train.ckpt is not None:
          load(model, optimizer, scheduler, cfg.train.ckpt)
      logger.add_video('gt', torch.from_numpy(
          gt).permute((0, 3, 1, 2)).unsqueeze(0), None, fps)
      logger.add_video('trainset', torch.from_numpy(
          gt[::cfg.train.train_split, ...]).permute((0, 3, 1, 2)).unsqueeze(0), None, int(fps/cfg.train.train_split))
      # train
      bar = tqdm(range(cfg.train.num_epoches))
      tracker = {'train_loss':[], 'train_epoch_loss':[], 'val_psnr':[]}
      for e in bar:
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
              tracker['train_loss'].append(loss.item())
              bar.set_description(f'{b}/{len(trainLoader)}:{loss.item():.3e}')
              epochLoss += loss.item()

          epochLoss /= len(trainLoader)
          logger.add_scalar('train/epoch loss', epochLoss, e)
          tracker['train_epoch_loss'].append(epochLoss)

          model.eval()
          reconLst = []
          for b, batch in enumerate(valLoader):
              # eval
              if isinstance(model, DataParallel):
                  pred = model.module.valStep(model, batch, mainDevice)
              else:
                  pred = model.valStep(model, batch, mainDevice)
              reconLst.append(pred)
          recon = torch.cat(reconLst, dim=0).reshape((numFrames, H, W, 3))
          logger.add_video(
              'val/recon', recon.permute((0, 3, 1, 2)).unsqueeze(0), e, fps)
          reconPSNR = 0
          for j in range(recon.shape[0]):
              reconPSNR += psnr(gt[j, ...], recon[j, ...].numpy(), data_range=1)
          reconPSNR /= recon.shape[0]
          logger.add_scalar('val/psnr', reconPSNR, e)
          tracker['val_psnr'].append(reconPSNR)
          scheduler.step()
          save(model, optimizer, scheduler, f'ckpts_L={cfg.model.ffm.L[i]}/epoch{e}.pt')

      with open(f'tracker_exp_L={cfg.model.ffm.L[i]}.pickle','wb') as f:
        pickle.dump(tracker,f)

if __name__ == '__main__':
    main()

