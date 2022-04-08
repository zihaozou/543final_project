import torch
import torch.nn as nn
from torch.autograd import grad
from torch.nn.functional import mse_loss


class GIFNERF(nn.Module):
    def __init__(self, ffm, mlp) -> None:
        super(GIFNERF, self).__init__()

        self.ffm = ffm
        self.mlp = mlp

    def forward(self, X):
        return (self.mlp(self.ffm(X)))

    @staticmethod
    def trainingStep(model, batch, device):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = mse_loss(pred, y)
        return loss

    @staticmethod
    def valStep(model, batch, device):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        pred = model(X).detach().cpu()
        return pred


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
