# -*- coding: utf-8 -*-
'''
Preprocessing object for image reshaping into a vector
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

class image_to_vector_model():
    """
    Parameters
    ----------
    data_description: dict
        Description of the input features
    """

    def __init__(self, data_description):
        self.data_description = data_description
        self.name = 'image_to_vector'

    def transform(self, X):
        """
        Transform image by transforming into 1D vector

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        NP = X.shape[0]
        NC = X.shape[1] # No. channels
        M = X.shape[2] # No. rows
        N = X.shape[3] # No. cols

        D = NC * M * N
        X_t = np.zeros((N, D))

        new_input_data_description = {
                            "NI": D, 
                            "input_types": [
                            {"type": "num", "name": "pixel value"}] * D
                            }

        X_t = np.zeros((NP, D))

        for k in tqdm(range(NP)):
            pixels = X[k, 0, :].reshape((1, D))
            X_t[k, :] = pixels

        return X_t, new_input_data_description
