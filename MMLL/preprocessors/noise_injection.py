# -*- coding: utf-8 -*-
'''
Preprocessing object for noise injection in the training data 
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np

class noise_injection_model():

    def __init__(self, input_data_description, alfa):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features

        alfa: float
            Noise factor wrt training data standard deviation
        """
        self.input_data_description = input_data_description
        self.alfa = alfa
        self.name = 'noise_injection'

    def transform(self, X):
        """
        Transform data with a noise injection

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        try:
            X_transf = []

            X = np.array(X)

            for kinput in range(self.input_data_description['NI']):

                # We apply noise_injection to numeric values
                if self.input_data_description['input_types'][kinput]['type'] in ['num', 'bin']:
                    newX = X[:, kinput].astype(float).reshape((-1, 1))
                    std = np.std(newX)

                    noise = np.random.normal(0, self.alfa * std, newX.shape)
                    newX += noise

                    X_transf.append(newX)

                if self.input_data_description['input_types'][kinput]['type'] == 'cat':
                    newX = X[:, kinput].reshape((-1, 1))
                    X_transf.append(newX)

            X_transf = np.hstack(X_transf)

        except:
            print('ERROR AT noise_injection_model')
            raise
            '''
            import code
            code.interact(local=locals())
            '''

        return X_transf