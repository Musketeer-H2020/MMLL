# -*- coding: utf-8 -*-
'''
Preprocessing object for PCA extraction
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import numpy as np

class PCA_model():

    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        self.P = None
        self.name = 'PCA'

    def fit(self, Rxx, NF):
        """
        Parameters
        ----------
        Rxx: matrix
            Cross-correlation matrix

        NF: int
            Number of features to extract
        """
        self.NF = NF
        #self.eig_vals, self.eig_vecs = np.linalg.eig(Rxx)
        #self.P = self.eig_vecs[:, 0:NF]
        # More stable, equivalent
        U, self.s, V = np.linalg.svd(Rxx)
        self.P = V[0:NF, :].T

    def transform(self, X):

        """
        Transform data reducing its dimensionality using PCA

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        try:
            X_transf = np.dot(X, self.P)
        except:
            print('ERROR AT PCA model transform')
            raise
            '''
            import code
            code.interact(local=locals())
            '''
        return X_transf