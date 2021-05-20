# -*- coding: utf-8 -*-
'''
Preprocessing object for obtaining a tfidf numerical matrix
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
import numpy as np
from tqdm import tqdm   # pip install tqdm

class tfidf_matrix_model():

    def __init__(self, vocab, df_dict, input_data_description):
        """
        Parameters
        ----------

        vocab: list of string
            Vocabulary to ne used

        df_dict: dict
            Document frequency counts.

        input_data_description: dict
            Description of the input features
        """
        self.input_data_description = input_data_description
        self.name = 'tfidf_matrix'
        self.new_input_data_description = {}
        self.new_input_data_description.update({'NI': len(vocab)})
        new_input_types = [{"type": "num", "name": "tfidf"}] * len(vocab)
        self.new_input_data_description.update({'input_types': new_input_types})
        self.vocab = vocab
        self.df_dict = df_dict

    def transform(self, X):
        """
        Transform data into a tfidf matrix

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed values: ndarray

        """
        Ndocs = X.shape[0]
        Xtfidf = np.zeros((Ndocs, self.new_input_data_description['NI']))

        inverse_index = {}
        for k in range(len(self.vocab)):
            inverse_index.update({self.vocab[k]: k})

        for kdoc in tqdm(range(Ndocs)):
            tf_dict = X[kdoc, 0]
            for word in list(tf_dict.keys()):
                if word in self.vocab: 
                    Xtfidf[kdoc, inverse_index[word]] = tf_dict[word]/self.df_dict[word]

        return Xtfidf