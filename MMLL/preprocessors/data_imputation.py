# -*- coding: utf-8 -*-
'''
Preprocessing object for missing data imputation (mean)
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np

class imputation_model_V():

    def __init__(self, input_data_description=None):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features
        """
        self.input_data_description = input_data_description
        self.name = 'missing_data_imputation_V'

    def transform(self, X):
        """
        Transform for missing data imputation in vertical partition

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

                self.means = np.nanmean(X, axis=0)
                NF = X.shape[1]
                for kfeat in range(NF):
                    x = X[:, kfeat]
                    if self.input_data_description['input_types'][kfeat]['type'] in ['num', 'bin']:
                        which = np.isnan(x)
                        x[which] = self.means[kfeat]
                        X_transf.append(x.reshape(-1, 1))
                    else:
                        X_transf.append(x.reshape(-1, 1))
                X_transf = np.hstack(X_transf)

            except:
                raise
                print('ERROR AT imputation_iobject')
                '''
                import code
                code.interact(local=locals())
                '''
        return X_transf