# -*- coding: utf-8 -*-
'''
Preprocessing object for data normalization
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np

class normalize_model():

    def __init__(self, input_data_description = None, method='global_mean_std'):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features

        method: string
            Seected method for normalization

        """
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.input_data_description = input_data_description 
        self.which_variables = None
        self.name = 'normalization'

    def transform(self, X):
        """
        Normalizing data given mean and std

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        X_transf = []

        ### Probando
        #x = ['?', 'Federal-gov', 'Private']
        #x_int = label_encoder.transform(x).reshape((-1, 1))
        # ohe = onehotencoder.transform(x_int)
        X = np.array(X)

        '''
        if self.which_variables is not None:
            print('stop at normalizer')
            import code
            code.interact(local=locals())

        '''

        for kinput in range(self.input_data_description['NI']):
            if self.input_data_description['input_types'][kinput]['type'] == 'num' or self.input_data_description['input_types'][kinput]['type'] == 'bin':
                try:
                    newX = X[:, kinput].astype(float).reshape((-1, 1))
                except:
                    raise
                    '''
                    print('ERROR HERE')
                    import code
                    code.interact(local=locals())
                    '''

                if self.method == 'global_mean_std' and self.mean is not None:
                    if self.mean[0, kinput] is not None and self.std[0, kinput] is not None and self.std[0, kinput] != 0:
                        newX = (newX - self.mean[0, kinput] ) / self.std[0, kinput]
                elif self.method == 'global_min_max':
                    if self.min[0, kinput] is not None and self.max[0, kinput] is not None:
                        newX = (newX - self.min[0, kinput] ) 
                        if (self.max[0, kinput]-self.min[0, kinput]) > 0:
                            newX = newX / (self.max[0, kinput]-self.min[0, kinput]) 
                newX = newX.reshape((-1, 1))
                X_transf.append(newX)

            #if self.input_data_description['input_types'][kinput]['type'] == 'bin' and self.which_variables!='all':
            #    newX = X[:, kinput].astype(float).reshape((-1, 1))
            #    X_transf.append(newX)

            if self.input_data_description['input_types'][kinput]['type'] == 'cat':
                print('ERROR AT normalizer: convert first categorical inputs to numeric using data2num')
                return None
        try:
            X_transf = np.hstack(X_transf)
        except:
            print('ERROR AT masternode model transform')
            raise
            '''
            import code
            code.interact(local=locals())
            '''

        return X_transf
