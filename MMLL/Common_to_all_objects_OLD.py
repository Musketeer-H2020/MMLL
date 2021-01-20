# -*- coding: utf-8 -*-
'''
Collection of methods common to all objects, to be inherited by other classes'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Mar 2020"


#import requests
#import json
#import pickle
#import base64
import numpy as np
#import dill


class Common_to_all_objects():
    """
    This class implements some basic methods andcommon to all objects.
    """

    def __init__(self):
        """
        Create a :class:`Common_to_all_objects` instance.

        Parameters
        ----------
        None

        """
        self.verbose = True
        return

    def display(self, message, verbose=True):
        """
        Write message to log file and display on screen if verbose=True

        :param message: string message to be shown/logged
        :type message: str
        """
        if verbose:
            if self.verbose:
                print(message)
        try:
            self.logger.info(message)
        except:
            pass
            
    def add_bias(self, X):
        # We add first column of ones, to deal with bias
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def process_kwargs(self, kwargs):
        """
        Process the variable arguments and assign the new variables

        Parameters
        ----------
        kwargs: dictionary
            Arbitrary keyword arguments.
            
        """
        for key, value in kwargs.items():
            if key == 'cr':
                self.cr = value
            if key == 'cryptonode_address':
                self.cryptonode_address = value
            if key == 'master_address':
                self.master_address = value
            if key == 'platform':
                self.platform = value
            if key == 'Nmaxiter':
                self.Nmaxiter = value
            if key == 'NC':
                self.NC = value
            if key == 'learning_rate':
                self.learning_rate = value
            if key == 'model_architecture':
                self.model_architecture = value
            if key == 'regularization':
                self.regularization = value
            if key == 'classes':
                self.classes = value
            if key == 'balance_classes':
                self.balance_classes = value
            if key == 'C':
                self.C = value
            if key == 'ni':
                self.ni = value
            if key == 'nf':
                self.nf = value
            if key == 'nbmu':
                self.nbmu = value
            if key == 'NI':
                self.NI = value
            if key == 'Xval_b':
                self.Xval_b = value
            if key == 'yval':
                self.yval = value
            if key == 'fsigma':
                self.fsigma = value
            if key == 'mu':
                self.mu = value
            if key == 'mu_min':
                self.mu_min = value
            if key == 'mu_max':
                self.mu_max = value
            if key == 'mu_step':
                self.mu_step = value
            if key == 'alpha_fixed':
                self.alpha_fixed = value
            if key == 'alpha_min':
                self.alpha_min = value
            if key == 'alpha_max':
                self.alpha_max = value
            if key == 'alpha_step':
                self.alpha_step = value
            if key == 'tolerance':
                self.tolerance = value
            if key == 'N':
                self.N = value
            if key == 'normalize_data':
                self.normalize_data = value
            if key == 'Csvm':
                self.Csvm = value
            if key == 'task_definition':
                self.task_definition = value
            if key == 'workers_addresses':
                self.workers_addresses = value
            if key == 'key_size':
                self.key_size = value
            if key == 'crypt_library':
                self.crypt_library = value
            if key == 'conv_stop':
                self.conv_stop = value
            if key == 'data_description':
                self.data_description = value
            if key == 'input_data_description':
                self.input_data_description = value
            if key == 'target_data_description':
                self.target_data_description = value


