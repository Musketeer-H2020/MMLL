# -*- coding: utf-8 -*-
'''
Preprocessing object for log scale data transformation 
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np

class logscale_model():

    def __init__(self, input_data_description=None):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features
        """
        self.input_data_description = input_data_description
        self.name = 'logscale'

    def transform(self, X):
        """
        Transform data with a log scale

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        X_transf = None
        if self.input_data_description is not None:
            try:
                X_transf = []

                X = np.array(X)

                for kinput in range(self.input_data_description['NI']):

                    # We apply logscale to numeric values
                    if self.input_data_description['input_types'][kinput]['type'] == 'num':
                        newX = X[:, kinput].astype(float).reshape((-1, 1))
                        try:
                            for kk in range(newX.shape[0]):
                                x = newX[kk, 0]
                                if x > 0:
                                    newX[kk, 0] = np.log(1 + x)
                                else: 
                                    newX[kk, 0] = -np.log(1 - x)
                        except:
                            raise
                            '''
                            print('ERROR HERE')
                            import code
                            code.interact(local=locals())
                            '''
                        X_transf.append(newX)

                    if self.input_data_description['input_types'][kinput]['type'] == 'bin':
                        newX = X[:, kinput].astype(float).reshape((-1, 1))
                        X_transf.append(newX)

                    if self.input_data_description['input_types'][kinput]['type'] == 'cat':
                        newX = X[:, kinput].reshape((-1, 1))
                        X_transf.append(newX)

                X_transf = np.hstack(X_transf)
            
            except:
                print('ERROR AT logscale')
                raise
                '''
                import code
                code.interact(local=locals())
                '''

        return X_transf