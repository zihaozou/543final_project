import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.nn.functional import mse_loss, sigmoid


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
        RGB = sigmoid(output[:, :3])
        if self.training:
            midValue = output[:, 3].sum()
            # torch.autograd.set_detect_anomaly(True)
            optics = grad(midValue, X,
                          None, retain_graph=True)[0][:, 1:]
            return torch.cat((RGB, optics), dim=1)
        else:
            return RGB

    @staticmethod
    def trainingStep(model, batch, device):
        X, y, op = batch
        X = X.to(device)
        y = y.to(device)
        op = op.to(device)
        X.requires_grad = True
        pred = model(X)
        loss = mse_loss(torch.cat((y, op), dim=1), pred)
        return loss

    @staticmethod
    def valStep(model, batch, device):
        X, y = batch
        X = X.to(device)
        y = y.to(device)
        pred = model(X).detach().cpu()
        return pred
