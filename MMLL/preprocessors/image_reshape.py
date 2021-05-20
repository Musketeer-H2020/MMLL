# -*- coding: utf-8 -*-
'''
Preprocessing object for image reshaping 
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


class image_reshape_model():
    """
    This class represents the main object for reshaping images.
    """

    def __init__(self, data_description, M, N):
        """
        Create an image_reshape_model instance.

        Parameters
        ----------
        data_description: dict
            Description of the input features

        M: integer
            Target number of rows

        N: integer
            Target number of columns
        """
        self.data_description = data_description
        self.name = 'image_reshape'
        NI = self.data_description['NI']
        self.M = M
        self.N = N

        self.transf = transforms.Resize((self.M, self.N))


    def transform(self, X):
        """
        Transform image by reshaping.

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray
        """
        N = X.shape[0]
        NC = X.shape[1]
        X_t = np.zeros((N, NC, self.M, self.N))
        for k in tqdm(range(N)):
            for kc in range(NC):
                pixels = X[k, kc, :]
                image = PIL_Image.fromarray(pixels)
                img_t = self.transf(image)
                pixels_t = np.array(img_t)
                X_t[k, kc, :, :] = pixels_t

        #print(X.shape)
        #print(X_t.shape)
        return X_t
