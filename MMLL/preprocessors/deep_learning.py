# -*- coding: utf-8 -*-
'''
Preprocessing object for deep learning image transformation 
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np
from torchvision import models  # pip install torchvision
from torchvision import transforms
from tqdm import tqdm   # pip install tqdm
from PIL import Image as PIL_Image
import torch

class deep_learning_model():

    def __init__(self, data_description):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features
        """
        self.data_description = data_description

        NI = self.data_description['NI']

        self.name = 'deep_learning'
        self.vision_model = models.alexnet(pretrained=True)
        # Grayscale
        self.transf = transforms.Compose([            #[1]
            transforms.Resize(256),                   #[2]
            transforms.CenterCrop(224),               #[3]
            transforms.ToTensor(),                    #[4]
            transforms.Normalize(                     #[5]
                mean=[0.485],           #[6]
                std=[0.229]             #[7]
            )])

    def transform(self, X):
        """
        Transform data with a Deep Learning preprocessing

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        self.vision_model.eval()
        N = X.shape[0]

        #pixels = (X[0, 1:] * 256.0).astype(np.uint8)
        #pixels = pixels.reshape((28, 28))
        pixels = X[0, 0, :]
        image = PIL_Image.fromarray(pixels)
        img_t = self.transf(image)
        # The input of the DL network needs RGB components
        img_3t = torch.cat((img_t, img_t, img_t), 0)
        batch_t = torch.unsqueeze(img_3t, 0)
        out = self.vision_model(batch_t)
        out_np = out.detach().numpy()
        newX = out_np

        for i in tqdm(range(1, N)):
            #pixels = (X[i, 1:] * 256.0).astype(np.uint8)
            #pixels = pixels.reshape((28, 28))
            pixels = X[i, 0, :]
            image = PIL_Image.fromarray(pixels)
            img_t = self.transf(image)
            img_3t = torch.cat((img_t, img_t, img_t), 0)
            batch_t = torch.unsqueeze(img_3t, 0)
            out = self.vision_model(batch_t)
            out_np = out.detach().numpy()
            newX = np.vstack((newX, out_np))

        # adding column of ones
        #NP = newX.shape[0]
        #newX_b = np.hstack((np.ones((NP, 1)), newX))

        print(newX.shape)

        return newX
