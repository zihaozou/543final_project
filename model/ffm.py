import torch
import torch.nn as nn
from math import pi


class BasicFFM(nn.Module):
    def __init__(self, mode, L, numInput):
        super(BasicFFM, self).__init__()
        self.mode = mode
        self.L = L
        stepTensor = torch.arange(end=L, start=0).repeat_interleave(2*numInput)
        index = torch.arange(start=0, end=L*numInput*2, step=numInput*2)
        ffm = torch.empty(L*2*numInput)
        if mode == 'linear':
            stepTensor += 1
            for i in range(numInput):
                ffm[index+i] = .5*stepTensor[index]*pi
                ffm[index+numInput+i] = .5*stepTensor[index]*pi
        elif mode == 'loglinear':
            for i in range(numInput):
                ffm[index+i] = 2**stepTensor[index]*pi
                ffm[index+numInput+i] = 2**stepTensor[index]*pi
        ffm = ffm.unsqueeze(0)
        self.register_buffer('ffmModule', ffm)

    def forward(self, x):
        repeatX = x.repeat(1, self.L*2)
        repeatX = repeatX*self.ffmModule
        ret = repeatX.clone()
        ret[:, ::4] = torch.sin(repeatX[:, ::4])
        ret[:, 1::4] = torch.sin(repeatX[:, 1::4])
        ret[:, 2::4] = torch.cos(repeatX[:, 2::4])
        ret[:, 3::4] = torch.cos(repeatX[:, 3::4])
        return repeatX
