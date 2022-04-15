import cv2
import numpy as np
import torch
from itertools import product


def gifDecodeArr(path, split, *args):
    cap = cv2.VideoCapture(path)
    frameLst = []

    while(1):
        ret, frame = cap.read()
        if not ret:
            break
        frameLst.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frames = np.stack(frameLst, axis=0)
    XFull = torch.from_numpy(np.asarray(list(product(range(frames.shape[0]), range(
        frames.shape[1]), range(frames.shape[2]))))).float()
    XFull[:, 0] = XFull[:, 0]/float(frames.shape[0])
    XFull[:, 1] = XFull[:, 1]/float(frames.shape[1])
    XFull[:, 2] = XFull[:, 2]/float(frames.shape[2])
    yFull = torch.from_numpy(frames.reshape(
        (-1, frames.shape[-1]))).float()/255.
    trainFrameList = range(0, frames.shape[0], split)
    Xtrain = torch.from_numpy(np.asarray(list(product(trainFrameList, range(
        frames.shape[1]), range(frames.shape[2]))))).float()
    Xtrain[:, 0] = Xtrain[:, 0]/float(frames.shape[0])
    Xtrain[:, 1] = Xtrain[:, 1]/float(frames.shape[1])
    Xtrain[:, 2] = Xtrain[:, 2]/float(frames.shape[2])
    yTrainImage = frames[::split, ...]
    yTrain = torch.from_numpy(yTrainImage.reshape(
        (-1, yTrainImage.shape[-1]))).float()/255.
    frames = frames.astype(float)/255.
    yTrainImage = frames[::split, ...]
    return frames, yTrainImage, XFull, yFull, Xtrain, yTrain, cap.get(5), frames.shape[0], frames.shape[1], frames.shape[2]
