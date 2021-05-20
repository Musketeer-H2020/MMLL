# -*- coding: utf-8 -*-
'''
Preprocessing object for random sparse projection
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np
import scipy

class random_projection_model():

    def __init__(self, input_data_description, NF, projection_type='full'):
        """
        Parameters
        ----------
        input_data_description: dict
            Description of the input features

        NF: int
            Number of features to obtain

        projection_type: string
            Selected type of projection
        """
        self.input_data_description = input_data_description
        self.name = 'random_projection_%s' % projection_type
        self.new_input_data_description = {}
        self.input_data_description = input_data_description
        self.projection_type = projection_type
        if self.projection_type == 'sparse' or self.projection_type == 'full':
            self.seed = int(np.random.uniform(0, 1000))
        else:
            raise Exception('Unknown type of projection %s, valid values are "full" or "sparse".' % projection_type)
        self.NI = input_data_description['NI']
        self.NF = NF

        self.new_input_data_description.update({'NI': NF})
        new_input_types = [{"type": "num", "name": "random projected"}] * NF
        self.new_input_data_description.update({'input_types': new_input_types})


    def transform(self, X):
        """
        Transform data reducing its dimensionality using random projection

        Parameters
        ----------
        X: ndarray4
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        try:
            np.random.seed(seed=self.seed)
            
            if self.projection_type == 'sparse':

                # Note, to avoid the transmission of large projection matrices, we simply store and send the seed, and generate 
                # the projection matrices just in time for the projection
                P = scipy.sparse.random(self.NI, self.NF, density=1.0/np.sqrt(self.NF), format='csr', random_state=self.seed)
                '''
                #print(P[0:4, 0:4])
                print(P[20,:])
                print('Stop at random projection')
                import code
                code.interact(local=locals())
                Ntimes = int(float(self.NI)/float(self.NF))
                P = np.eye(self.NF)
                for k in range(Ntimes):
                    P = np.hstack((P, np.eye(self.NF)))

                np.random.shuffle(P)
                P = P.T
                np.random.shuffle(P)
                P = P.T
                P = P[:, 0:NI].T
                '''

            elif self.projection_type == 'full':
                P = np.random.normal(0, 1/np.sqrt(self.NF), (self.NI, self.NF))
                #print(P[0:4, 0:4])
            else:
                raise Exception('Unknown type of projection %s, valid values are "full" or "sparse".' % projection_type)

            if self.input_data_description['NI'] == X.shape[1]:
                #if scipy.sparse.issparse(X):
                #    X = np.array(X.todense())

                # Warning when X is full nparray and P is scipy sparse...
                sparse_X = False
                try:
                    Xtype = X.getformat()
                    if Xtype == 'csr' or Xtype == 'csc':
                        sparse_X = True
                except:
                    pass

                sparse_P = False
                try:
                    Ptype = P.getformat()
                    if Ptype == 'csr' or Ptype == 'csc':
                        sparse_P = True
                except:
                    pass

                # Both full
                if not sparse_X and not sparse_P:
                    X_P = np.dot(X, P)

                if not sparse_X and sparse_P:
                    P = np.array(P.todense())
                    X_P = np.dot(X, P)

                if sparse_X and not sparse_P:
                    X = np.array(X.todense())
                    X_P = np.dot(X, P)

                if sparse_X and sparse_P:
                    X_P = np.dot(X, P)

                '''
                X_P = np.zeros(N, self.NF)
                for k in range(N):
                    x = X[k, :].reshape(1, -1)
                    xp = np.dot(x, P)
                '''
                del P                     
                return X_P
            else:
                raise Exception('Wrong input dimension: received %d, expected %d' % (X.shape[1], self.input_data_description['NI']))
                return None
        except:
            print('ERROR at RP transform')
            raise
            '''
            import code
            code.interact(local=locals())
            '''

