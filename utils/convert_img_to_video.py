import torch
import cv2
import numpy as np


def convert_image_video(path):
    image = torch.load(path)
    image = image.numpy()

    height = image.shape[1]
    width = image.shape[2]
    numFrames = image.shape[0]
    fps = 30

    for frame in range(image.shape[0]):
        image[frame, ...] = cv2.cvtColor(image[frame, ...], cv2.COLOR_BGR2RGB)

    fourcc = cv2.VideoWriter_fourcc(*'MP42') 
    video = cv2.VideoWriter('test.mp4', fourcc, float(fps), (height, width))
    
    for frame in range(numFrames):
        video.write((image[frame,...]*255).astype(np.uint8))
    
    video.release()