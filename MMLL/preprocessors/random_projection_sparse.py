# -*- coding: utf-8 -*-
'''
Preprocessing object for random sparse projection
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np

class random_projection_sparse_model():

    def __init__(self, input_data_description, NF):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features

        NF: int
            Number of features to extract

        """
        self.input_data_description = input_data_description
        self.name = 'random_sparse_projection'
        self.new_input_data_description = {}
        self.input_data_description = input_data_description

        NI = input_data_description['NI']
        Ntimes = int(float(NI)/float(NF))
        P = np.eye(NF)
        for k in range(Ntimes):
            P = np.hstack((P, np.eye(NF)))

        np.random.shuffle(P)
        P = P.T
        np.random.shuffle(P)
        P = P.T
        self.P = P[:, 0:NI].T
        self.new_input_data_description.update({'NI': NF})
        new_input_types = [{"type": "num", "name": "projected_tfidf"}] * NF
        self.new_input_data_description.update({'input_types': new_input_types})

    def transform(self, X):
        """
        Transform sparse data reducing its dimensionality using random projection

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        if self.input_data_description['NI'] == X.shape[1]:
            return np.dot(X, self.P)
        else:
            raise Exception('Wrong input dimension: received %d, expected %d' % (X.shape[1], self.input_data_description['NI']))
            return None
