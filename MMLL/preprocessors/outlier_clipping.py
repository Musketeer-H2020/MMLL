# -*- coding: utf-8 -*-
'''
Preprocessing object for outlier_clipping 
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np

class outlier_clipping_model():

    def __init__(self, input_data_description, times_sigma):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features

        times_sigma: float
            Maximal allowed variation with respect to data standard deviation
        """
        self.input_data_description = input_data_description
        self.times_sigma = times_sigma
        self.name = 'outlier_clipping'

    def transform(self, X):
        """
        Transform data by removing outliers

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

                # We apply outlier_clipping to numeric values
                if self.input_data_description['input_types'][kinput]['type'] == 'num':
                    newX = X[:, kinput].astype(float).reshape((-1, 1))
                    mean = np.mean(newX)
                    std = np.std(newX)
                    upper = mean + self.times_sigma * std
                    lower = mean - self.times_sigma * std

                    for kk in range(newX.shape[0]):
                        x = newX[kk, 0]
                        if x > upper:
                            newX[kk, 0] = upper
                            #print('Clipping UP')
                        if x < lower:
                            newX[kk, 0] = lower
                            #print('Clipping DOWN')

                    X_transf.append(newX)

                if self.input_data_description['input_types'][kinput]['type'] == 'bin':
                    newX = X[:, kinput].astype(float).reshape((-1, 1))
                    X_transf.append(newX)

                if self.input_data_description['input_types'][kinput]['type'] == 'cat':
                    newX = X[:, kinput].reshape((-1, 1))
                    X_transf.append(newX)

            X_transf = np.hstack(X_transf)

        except:
            print('ERROR AT outlier_clipping_model')
            raise
            '''
            import code
            code.interact(local=locals())
            '''

        return X_transf