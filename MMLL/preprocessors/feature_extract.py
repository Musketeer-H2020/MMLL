# -*- coding: utf-8 -*-
'''
Preprocessing object for extracting given features 
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np
import scipy

class feature_extract_model():

    def __init__(self, selected_features, input_data_description):
        """
        Parameters
        ----------
        selected_features: list of indices
            Features to be retained

        input_data_description: dict
            Description of the input features
        """
        self.selected_features = selected_features
        self.input_data_description = input_data_description
        self.name = 'feature_extract'
        self.new_input_data_description = {}
        self.new_input_data_description.update({'NI': len(self.selected_features)})

        old_input_types = self.input_data_description['input_types']
        new_input_types = []
        for k in self.selected_features:
            new_input_types.append(old_input_types[k])

        self.new_input_data_description.update({'input_types': new_input_types})

    def transform(self, X):
        """
        Transform data by extracting features

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        try:
            # Transform X
            X_transf = X[:, self.selected_features] 
            # Converting to dense if sparse

            sparse = False
            try:
                Xtype = X_transf.getformat()
                if Xtype == 'csr' or Xtype == 'csc':
                    sparse = True
            except:
                pass

            # If the matrix is sparse, we skip this processing
            #if scipy.sparse.issparse(X_transf):
            if sparse:
                X_transf = np.array(X_transf.todense())

        except:
            print('ERROR AT feature_extract_model transform')
            raise
            '''
            import code
            code.interact(local=locals())
            '''

        return X_transf