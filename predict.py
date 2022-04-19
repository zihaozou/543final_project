import torch
from torch.nn import DataParallel
from model.mlp import MLP
from model.ffm import BasicFFM
import model.gif_nerf as gifNerf
from utils.gif_process import gifDecodeArr
from hydra.utils import get_original_cwd
from os.path import join
from torch.utils.data.dataloader import DataLoader
from utils.dataset import gifDataset
import hydra
from utils.save_load import load_model_only
import numpy as np
from moviepy.editor import ImageSequenceClip
from itertools import product
@hydra.main(config_path="conf", config_name="predict_config")
def predict(cfg):
    gt,y, _, _, _, _, fps, numFrames, H, W = gifDecodeArr(
        join(get_original_cwd(), *cfg.gif.data_path.split('/')), cfg.gif.split, *cfg.gif.ofargs)
    Xinput=torch.tensor(list(product(range(cfg.gif.slowmo_factor*numFrames),range(H),range(W)))).to(float)
    Xinput[:,0]=Xinput[:,0]/float(cfg.gif.slowmo_factor*numFrames)
    Xinput[:,1]=Xinput[:,1]/float(H)
    Xinput[:,2]=Xinput[:,2]/float(W)
    ffm = BasicFFM(cfg.model.ffm.mode, cfg.model.ffm.L,
                   cfg.model.ffm.num_input)
    mlp = MLP(cfg.model.mlp.num_inputs,
              cfg.model.mlp.num_outputs, cfg.model.mlp.num_neurons,
              cfg.model.mlp.depth, cfg.model.mlp.skip_list,
              cfg.model.mlp.body_acti, cfg.model.mlp.last_acti)
    mainDevice = f'cuda:{cfg.model.GPUIndex[0]}'
    model = getattr(gifNerf, cfg.model.nerf.type)(ffm, mlp).to(mainDevice)
    path = cfg.model.path   #Must replace the path based on your choice
    load_model_only(model, path)
    XChunk=torch.split(Xinput,cfg.model.batch_size)
    reconLst = []
    for b, batch in enumerate(XChunk):
            pred = model.valStep(model, batch, mainDevice)
            reconLst.append(pred)
    recon = torch.cat(reconLst, dim=0).reshape((numFrames, H, W, 3))
    recon = (recon.numpy()*255).astype(np.uint8)
    clip = ImageSequenceClip(list(recon), fps)
    clip.write_gif(f'{cfg.model.name}_recon.gif', fps)
    clip = ImageSequenceClip(list((gt*255).astype(np.uint8)), fps)
    clip.write_gif(f'{cfg.model.name}_gt.gif', fps)
    clip = ImageSequenceClip(list((y*255).astype(np.uint8)), fps/cfg.gif.split)
    clip.write_gif(f'{cfg.model.name}_reduced.gif', fps/cfg.gif.split)
if __name__=='__main__':
    predict()