import torch
import torch.nn as nn
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.nn.functional import mse_loss, sigmoid, l1_loss, pad
import model.softsplat as fw
from torch.autograd import grad
from math import ceil, log, pow


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
    def interframeTrainStep(model, flowComp, backWrap, batch, co, alpha, bs, ssim, device):
        X, y = batch
        y = y.to(device)
        XChunk = torch.split(X, bs)

        with torch.no_grad():
            predChunk = []
            for XPiece in XChunk:
                XPiece = XPiece.to(device)
                predPiece = model(XPiece)
                predChunk.append(predPiece)
            pred = torch.cat(predChunk, dim=0)
            del predChunk, XChunk
            pred = pred.view(
                (1, y.shape[2], y.shape[3], 3)).permute((0, 3, 1, 2))

            flowOut = flowComp(
                pad(y.reshape((1, 6, y.shape[2], y.shape[3])), (0, int(pow(2, ceil(log(y.shape[2], 2)))-y.shape[2]), 0, int(pow(2, ceil(log(y.shape[3], 2)))-y.shape[3])), mode='constant'))[:, :, :y.shape[2], :y.shape[3]]
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]
            #F_t_0 = -(1-co)*co * F_0_1 + co**2 * F_1_0
            #F_t_1 = (1-co)**2 * F_0_1 - co*(1-co) * F_1_0
            tenMetric1 = l1_loss(input=y[0, ...].unsqueeze(0), target=backWrap(
                y[1, ...].unsqueeze(0), F_0_1), reduction='none').mean(1, True)
            tenMetric2 = l1_loss(input=y[1, ...].unsqueeze(0), target=backWrap(
                y[0, ...].unsqueeze(0), F_1_0), reduction='none').mean(1, True)
        pred.requires_grad_()
        recon1 = co*fw.softsplat(
            y[0, ...], tenFlow=F_0_1*co, tenMetric=(-1 * tenMetric1).clip(-1, 1), strMode='soft')

        #midLoss1 = mse_loss(recon1, y[1, ...].unsqueeze(0))

        recon2 = (1-co)*fw.softsplat(
            y[1, ...], tenFlow=F_1_0*(1-co), tenMetric=(-1 * tenMetric2).clip(-1, 1), strMode='soft')
        loss = mse_loss(recon1+recon2, pred)
        #midLoss2 = mse_loss(recon2, y[1, ...].unsqueeze(0))
        midGrad = grad(loss, pred, torch.tensor(
            1, dtype=torch.float32, device=device))[0]
        return midGrad

    @staticmethod
    def valStep(model, batch, device):
        X = batch
        X = X.to(device)
        #y = y.to(device)
        pred = model(X).detach().cpu()
        return pred
