import torch
import torch.nn as nn
from torch.autograd import grad


class GIFNERF(nn.Module):
    def __init__(self, ffm, mlp) -> None:
        super(GIFNERF, self).__init__()

        self.ffm = ffm
        self.mlp = mlp

    def forward(self, X):
        return (self.mlp(self.ffm(X)))


class GIFNERFJacob(nn.Module):
    def __init__(self, ffm, mlp) -> None:
        super(GIFNERFJacob, self).__init__()
        self.ffm = ffm
        self.mlp = mlp

    def forward(self, X):
        output = self.mlp(self.ffm(X))
        RGB = output[:, 0:3]
        midValue = output[:, 3]
        optics = grad(midValue, X[:, 1:],
                      grad_outputs=torch.ones_like(midValue), retain_graph=True)
        return torch.cat((RGB, optics), dim=1)
