# -*- coding: utf-8 -*-
'''
Collection of methods common to all objects, to be inherited by other classes
'''

__author__ = "Angel Navia-Vázquez and Marcos Fernández Díaz"
__date__ = "Mar 2020"


#import requests
#import json
#import pickle
#import base64
import numpy as np
#import dill
from tqdm import tqdm # pip install tqdm
import hashlib # pip install hashlib
import os, sys
#sys.path.append("..")
#sys.path.append("../..")
from MMLL.preprocessors.normalizer import normalize_model
from MMLL.preprocessors.data2num import data2num_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.stats import kurtosis, skew

class Common_to_all_objects():
    """
    This class implements some basic methods and common to all objects.
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

    def display(self, message, verbose=None):
        """
        Write message to log file and display on screen if verbose=True.

        Parameters
        ----------
        message: string
            Message to be displayed.

        verbose: boolean
            Indicates whether to print messages on screen nor not.
        """
        if verbose is not None:
            if verbose:
                print(message)
        else:
            if self.verbose:
                print(message)
        try:
            self.logger.info(message)
        except:
            pass
            
    def add_bias(self, X):
        """
        Adds a first column of value 1.

        Parameters
        ----------
        X: ndarray
            Matrix with the input values.

        Returns
        -------
        X_b: ndarray
            Matrix with the bias added.
        """

        # We add first column of ones, to deal with bias
        X_b = np.hstack((np.ones((X.shape[0], 1)), X.astype(float)))
        return X_b

    def compute_stats(self, X, y, stats_list):
        """
        Compute statistics on data.

        Parameters
        ----------
        X: ndarray
            Matrix with the input values.

        X: ndarray
            Vector with the target values.

        stats_list: list of strings
            Statistics to be computed.

        Returns
        -------
        stats_dict: dict
            Computed statistics.
        """
        stats_dict = {}

        X = X.astype(float)
        y = np.array(y).astype(float)

        if 'rxy' in stats_list:
            rxy = np.dot(X.T, y)
            # normalize norm
            if np.linalg.norm(rxy) > 0:
                rxy = rxy / np.linalg.norm(rxy)
            stats_dict.update({'rxy': rxy})

        if 'meanx' in stats_list:
            meanx = np.mean(X, axis=0)
            # normalize norm
            if np.linalg.norm(meanx) > 0:
                meanx = meanx / np.linalg.norm(meanx)
            stats_dict.update({'meanx': meanx})

        if 'medianx' in stats_list:
            medianx = np.median(X, axis=0)
            # normalize norm
            if np.linalg.norm(medianx) > 0:
                medianx = medianx / np.linalg.norm(medianx)
            stats_dict.update({'medianx': medianx})

        if 'npatterns' in stats_list:
            npatterns = X.shape[0]
            stats_dict.update({'npatterns': npatterns})

        if 'stdx' in stats_list:
            stdx = np.stdx(X, axis=0)
            # normalize norm
            if np.linalg.norm(stdx) > 0:
                stdx = stdx / np.linalg.norm(stdx)
            stats_dict.update({'stdx': stdx})

        if 'skewx' in stats_list:
            skewx = skew(X)
            # normalize norm
            if np.linalg.norm(skewx) > 0:
                skewx = skewx / np.linalg.norm(skewx)
            stats_dict.update({'skewx': skewx})

        if 'kurx' in stats_list:
            kurx = kurtosis(X)
            # normalize norm
            if np.linalg.norm(kurx) > 0:
                kurx = kurx / np.linalg.norm(kurx)
            stats_dict.update({'kurx': kurx})

        if 'perc25' in stats_list:
            perc25 = np.percentile(X, 25, axis=0)
            # normalize norm
            if np.linalg.norm(perc25) > 0:
                perc25 = perc25 / np.linalg.norm(perc25)
            stats_dict.update({'perc25': perc25})

        if 'perc75' in stats_list:
            perc75 = np.percentile(X, 75, axis=0)
            # normalize norm
            if np.linalg.norm(perc75) > 0:
                perc75 = perc75 / np.linalg.norm(perc75)
            stats_dict.update({'perc75': perc75})

        if 'staty' in stats_list:
            staty = np.array([np.mean(y), np.std(y), skew(y), kurtosis(y)])
            # normalize norm
            if np.linalg.norm(staty) > 0:
                staty = staty / np.linalg.norm(staty)
            stats_dict.update({'staty': staty})

        return stats_dict

    def compute_gfs(self, Rxy_b, rxy_b, Xval, yval, NF=None, stop_incr=None, regularization=0.0001):
        '''
        Feature Selection by a brute force greedy approach using linear models.

        Parameters
        ----------
        Rxy_b: ndarray
            Self Correlation Matrix.

        rxy_b: ndarray
            Cross-correlation Vector.

        Xval: ndarray
            Validation input data .

        yval: ndarray
            Validation target data.

        NF: integer
            Number of features to extract.

        stop_incr: float
            Threshold to stop extracting features.

        regularization: float
            Regularization value to be used in the linear model.

        Returns
        -------
        pos_selected: list of int
            Positions of the selected features.

        perf_val: list of float
            Performance on the validation set.
        '''
        try: 
            NI = Rxy_b.shape[0]
            pos_remain = list(range(0, NI))
            pos_selected = []
            Xval_b = self.add_bias(Xval)

            if NF is None: 
                NF = NI  # We rank all
            else:
                NF = NF + 1 # bias will be removed
            
            yval = np.array(yval).ravel().astype(float)

            self.display('Ranking input variables...', verbose=True)

            for i in tqdm(range(NF)):
                get_more_features = False
                if len(pos_selected) < NF + 1:
                    get_more_features = True

                if stop_incr is not None and i > 1:
                    try: 
                        if (perf_val[i - 2] - perf_val[i - 1]) / perf_val[i - 2] > stop_incr:
                            get_more_features = True
                        else:
                            get_more_features = False
                    except:
                        get_more_features = False

                if get_more_features:
                    if len(pos_selected) < NF + 1:
                        if len(pos_selected) == 0:
                            pos_selected = [0]
                            cuales_select = pos_selected
                            R_tmp = Rxy_b[cuales_select, :]
                            R_tmp = R_tmp[:, cuales_select]

                            r_tmp = rxy_b[cuales_select]
                            NI_tmp = R_tmp.shape[0]
                            w = np.dot(np.linalg.inv(R_tmp + regularization * np.eye(NI_tmp)), r_tmp)

                            Xval_tmp = Xval_b[:, cuales_select]
                            preds_val = np.dot(Xval_tmp, w.ravel())
                            error = (preds_val.ravel() - yval.ravel()) ** 2
                            perf_val = [np.mean(error)]  # store here the final values
                            pos_remain = list(set(pos_remain) - set([0]))
                        else:
                            perf_val_tmp = []
                            for pos_eval in pos_remain:
                                cuales_select = pos_selected + [pos_eval]
                                R_tmp = Rxy_b[cuales_select, :]
                                R_tmp = R_tmp[:, cuales_select]
                                r_tmp = rxy_b[cuales_select]
                                NI_tmp = R_tmp.shape[0]
                                Xval_tmp = Xval_b[:, cuales_select]
                                w = np.dot(np.linalg.inv(R_tmp + regularization * np.eye(NI_tmp)), r_tmp)
                                preds_val = np.dot(Xval_tmp, w.ravel())
                                error = (preds_val.ravel() - yval.ravel()) ** 2
                                perf = np.mean(error)
                                perf_val_tmp.append(perf)

                            # select the best
                            which_min = np.argmin(np.array(perf_val_tmp))
                            new_index = pos_remain[which_min]
                            pos_selected.append(new_index)
                            pos_remain = list(set(pos_remain) - set([new_index]))
                            perf_val.append(perf_val_tmp[which_min])
        except:
            pass
        # remove bias (first position)
        pos_selected = pos_selected[1:]
        pos_selected = [x-1 for x in pos_selected]
        perf_val = perf_val[1:]
        
        return pos_selected, perf_val

    def hash_md5(self, x):
        '''
        Compute hash value using MD5.

        Parameters
        ----------
        x: string
            Input value.

        Returns
        -------
        z: string
            Hash value.
        '''
        y = hashlib.md5(str(x).encode()).hexdigest()
        return y

    def hash_sha256(self, x):
        '''
        Compute hash value using SHA256.

        Parameters
        ----------
        x: string
            input value.

        Returns
        -------
        z: string
            Hash value.
        '''
        z = hashlib.sha256(str(x).encode()).hexdigest()
        return z

    def hash_sha3_512(self, x):
        '''
        Compute hash value using SHA512.

        Parameters
        ----------
        x: string
            Input value.

        Returns
        -------
        z: string
            Hash value.
        '''
        z = hashlib.sha3_512(str(x).encode()).hexdigest()
        return z

    def get_data2num_model(self, input_data_description):
        '''
        Obtain model to convert data to numeric.

        Parameters
        ----------
        input_data_description: dictionary
            Description of the input data.

        Returns
        -------
        model: :class:`datanum_model`
            Model to transform the data.
        '''
        types = [x['type'] for x in input_data_description['input_types']]
        model = None
        if 'cat' in types:  # there is something to transform
            model = data2num_model(input_data_description)
            '''
            model.input_data_description = input_data_description
            model.mean = None
            model.std = None

            onehot_encodings = {}
            label_encodings = {}
            #new_data_description = dict(data_description)
            new_input_types = []

            for k in range(input_data_description['NI']):
                if input_data_description['input_types'][k]['type'] == 'cat':
                    categories = input_data_description['input_types'][k]['values']
                    Nbin= len(categories)
                    label_encoder = LabelEncoder()
                    label_encoder.fit(categories)
                    label_encodings.update({k: label_encoder})
                    integer_encoded = label_encoder.transform(categories).reshape((-1, 1))
                    # Creating one-hot-encoding object
                    onehotencoder = OneHotEncoder(sparse=False)
                    onehotencoder.fit(integer_encoded)
                    onehot_encodings.update({k: onehotencoder})
                    onehot_encoded = onehotencoder.transform(np.array([1, 2]).reshape(-1, 1))
                    aux = [{'type': 'bin', 'name': 'onehot transformed'}] * Nbin
                    new_input_types += aux
                else: # dejamos lo que hay
                    new_input_types.append(input_data_description['input_types'][k])

            new_input_data_description = {
                        "NI": len(new_input_types), 
                        "input_types": new_input_types
                        }

            model.label_encodings = label_encodings
            model.onehot_encodings = onehot_encodings
            model.name = 'data2num'
            model.new_input_data_description = new_input_data_description
            '''
        return model

    def process_kwargs(self, kwargs):
        """
        Process the variable arguments and assign the new variables.

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
            if key == 'sigma':
                self.sigma = value
            if key == 'NmaxiterGD':
                self.NmaxiterGD = value
            if key == 'eta':
                self.eta = value
            if key == 'learning_rate':
                self.learning_rate = value
            if key == 'model_architecture':
                self.model_architecture = value
            if key == 'optimizer':
                self.optimizer = value
            if key == 'loss':
                self.loss = value
            if key == 'metric':
                self.metric = value
            if key == 'batch_size':
                self.batch_size = value
            if key == 'num_epochs':
                self.num_epochs = value
            if key == 'model_averaging':
                self.model_averaging = value
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
            if key == 'aggregation_type':
                self.aggregation_type = value
            if key == 'use_bias':
                self.use_bias = value
            if key == 'num_epochs_worker':
                self.num_epochs_worker = value
            if key == 'eps':
                self.eps = value
            if key == 'nesterov':
                self.nesterov = value
            if key == 'momentum':
                self.momentum = value
            if key == 'minvalue':
                self.minvalue = value
            if key == 'maxvalue':
                self.maxvalue = value
            if key == 'Tmax':
                self.Tmax = value
            if key == 'landa':
                self.landa = value
            if key == 'Xtr':
                self.Xtr = value
            if key == 'ytr':
                self.ytr = value
            if key == 'minibatch':
                self.minibatch = value

