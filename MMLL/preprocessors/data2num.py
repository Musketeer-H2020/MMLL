# -*- coding: utf-8 -*-
'''
Preprocessing object for data transformation to numeric
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class data2num_model():

    def __init__(self, input_data_description = None):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features
        """
        self.input_data_description = input_data_description 
        self.name = 'data2num'

        self.onehot_encodings = {}
        self.label_encodings = {}
        new_input_types = []

        for k in range(self.input_data_description['NI']):
            if self.input_data_description['input_types'][k]['type'] == 'cat':
                categories = self.input_data_description['input_types'][k]['values']
                Nbin= len(categories)
                label_encoder = LabelEncoder()
                label_encoder.fit(categories)
                self.label_encodings.update({k: label_encoder})
                integer_encoded = label_encoder.transform(categories).reshape((-1, 1))
                # Creating one-hot-encoding object
                onehotencoder = OneHotEncoder(sparse=False)
                onehotencoder.fit(integer_encoded)
                self.onehot_encodings.update({k: onehotencoder})
                #onehot_encoded = onehotencoder.transform(np.array([0, 1]).reshape(-1, 1))
                aux = [{'type': 'bin', 'name': 'onehot transformed'}] * Nbin
                new_input_types += aux
            else: # dejamos lo que hay
                new_input_types.append(input_data_description['input_types'][k])

        self.new_input_data_description = {
                    "NI": len(new_input_types), 
                    "input_types": new_input_types
                    }


    def transform(self, X):
        """
        Transform data into numeric

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        X_transf = []
        X = np.array(X)

        for kinput in range(self.input_data_description['NI']):
            if self.input_data_description['input_types'][kinput]['type'] == 'num' or (self.input_data_description['input_types'][kinput]['type'] == 'bin'):
                try:
                    values = list(X[:, kinput])
                    
                    if '' in values: # mising data
                        N = X[:, kinput].shape[0]
                        newX = []
                        for k in range(N):
                            value = X[k, kinput]
                            
                            if value != '':
                                newX.append(value.astype(float))
                            else:
                                newX.append(np.nan)
                        newX = np.array(newX).reshape((-1, 1))
                    else:
                        newX = X[:, kinput].astype(float).reshape((-1, 1))
                except:
                    print('ERROR at datanum, not a numeric value in numeric feature.')
                    raise
                    '''
                    import code
                    code.interact(local=locals())
                    '''

                X_transf.append(newX)

            if self.input_data_description['input_types'][kinput]['type'] == 'cat':

                Xcat = X[:, kinput]
                if '' in list(Xcat):  # missing values
                    print('Missing values in feature %s' % self.input_data_description['input_types'][kinput]['name'])
                    # getting onehotencoding size
                    sample = self.input_data_description['input_types'][kinput]['values'][0]
                    sample_int = self.label_encodings[kinput].transform(np.array([sample]))
                    sample_ohe = self.onehot_encodings[kinput].transform(sample_int.reshape(1, -1))[0]
                    Nohe = sample_ohe.shape[0]
                    xnan = np.empty(Nohe)
                    xnan[:] = np.nan

                    N = Xcat.shape[0]
                    Xohe = []

                    for k in range(N):
                        sample = Xcat[k]
                        if sample != '':
                            sample_int = self.label_encodings[kinput].transform(np.array([sample]))
                            sample_ohe = self.onehot_encodings[kinput].transform(sample_int.reshape(1, -1))[0]
                            Xohe.append(sample_ohe)
                        else:
                            Xohe.append(xnan)

                    Xohe = np.array(Xohe)
                else: # no missing values
                    # This block version does not allow nan values, but it is much faster
                    x_int = self.label_encodings[kinput].transform(Xcat).reshape((-1, 1))
                    Xohe = self.onehot_encodings[kinput].transform(x_int)

                X_transf.append(Xohe)
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