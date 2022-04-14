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

@hydra.main(config_path="conf", config_name="predict_config")
def predict(cfg):
    _, XFull, yFull, _, _, _, fps, numFrames, H, W = gifDecodeArr(
        join(get_original_cwd(), *cfg.gif.data_path.split('/')), cfg.gif.split, *cfg.gif.ofargs)
    valLoader = DataLoader(gifDataset(
        XFull, yFull, None), batch_size=cfg.model.batch_size, shuffle=False, num_workers=8, pin_memory=True)    

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

    reconLst = []
    for b, batch in enumerate(valLoader):
            pred = model.valStep(model, batch, mainDevice)
            reconLst.append(pred)
    recon = torch.cat(reconLst, dim=0).reshape((numFrames, H, W, 3))
    recon = (recon.numpy()*255).astype(np.uint8)
    clip = ImageSequenceClip(list(recon), fps)
    clip.write_gif('test.gif', fps)


if __name__=='__main__':
    predict()