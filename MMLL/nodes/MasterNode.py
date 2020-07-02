# -*- coding: utf-8 -*-
'''
MasterNode Class
'''

__author__ = "Angel Navia-Vázquez, Marcos Fernández"
__date__ = "May 2020"

import numpy as np
import sys
import pickle
sys.path.append("..")
sys.path.append("../..")
from MMLL.Common_to_all_objects import Common_to_all_objects
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

class normalize_model():

    def __init__(self, method='global_mean_std'):
        self.method = method
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.data_description = None 

    def transform(self, X):
        """
        Transform data given mean and std

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

        for kinput in range(self.data_description['NI']):
            if self.data_description['input_types'][kinput]['type'] == 'num':
                try:
                    newX = X[:, kinput].astype(float).reshape((-1, 1))
                except:
                    print('ERROR HERE')
                    import code
                    code.interact(local=locals())

                if self.method == 'global_mean_std':
                    if self.mean[0, kinput] is not None and self.std[0, kinput] is not None and self.std[0, kinput] != 0:
                        newX = X[:, kinput].astype(float).reshape((-1, 1))
                        newX = (newX - self.mean[0, kinput] ) / self.std[0, kinput]
                        newX = newX.reshape((-1, 1))
                elif self.method == 'global_min_max':
                    if self.min[0, kinput] is not None and self.max[0, kinput] is not None:
                        newX = X[:, kinput].astype(float).reshape((-1, 1))
                        newX = (newX - self.min[0, kinput] ) 
                        if (self.max[0, kinput]-self.min[0, kinput]) > 0:
                            newX = newX / (self.max[0, kinput]-self.min[0, kinput]) 
                        newX = newX.reshape((-1, 1))
                X_transf.append(newX)

            if self.data_description['input_types'][kinput]['type'] == 'bin':
                newX = X[:, kinput].astype(float).reshape((-1, 1))
                X_transf.append(newX)

            if self.data_description['input_types'][kinput]['type'] == 'cat':
                Xcat = X[:, kinput]
                x_int = self.label_encodings[kinput].transform(Xcat).reshape((-1, 1))
                Xohe = self.onehot_encodings[kinput].transform(x_int)
                X_transf.append(Xohe)

        try:
            X_transf = np.hstack(X_transf)
        except:
            print('ERROR AT masternode model transform')
            import code
            code.interact(local=locals())

        return X_transf


class MasterNode(Common_to_all_objects):
    """
    This class represents the main process associated to the Master Node, and serves to 
    coordinate the training procedure under the different POMs
    """
    def __init__(self, pom, comms, logger, verbose=False, **kwargs):

        """
        Creates a :class:`MasterNode` instance.

        Parameters
        ----------
        pom: integer
            the selected POM

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen

        **kwargs: Variable keyword arguments.
       """

        self.pom = pom                                      # Selected POM
        self.comms = comms                                  # comms library
        self.workers_addresses = comms.workers_ids          # Workers addresses + ca
        self.Nworkers = len(self.workers_addresses)
        self.master_address = 'ma' 
        self.logger = logger
        self.verbose = verbose # print on screen when true
        
        #self.normalize_data = False                           
        self.classes = None                           
        self.balance_classes = False
        # Processing kwargs
        self.process_kwargs(kwargs)
        # We assume that the Master may receive a validation and a test set 
        self.Xtst_b = None                                  # Test data (input)
        self.ytst = None                                    # Test data (targets)
        self.Xval_b = None                                  # Validation data (input)
        self.yval = None       # Validation data (targets)
        self.model_is_trained = False
        self.classes = None
        self.display('MasterNode: Initiated')
        #print(self.master_address)
        #print(self.workers_addresses)
        self.data_is_ready = False

    def create_model_Master(self, model_type, model_parameters=None):
        """
        Create the model object to be used for training at the Master side.

        Parameters
        ----------
        model_type: str
            Type of model to be used

        model_parameters: dictionary
            parameters needed by the different models, for example it may contain:

            Nmaxiter: integer
                Maximum number of iterations during learning

            NC: integer
                Number of centroids

            regularization: float
                Regularization parameter

            classes: list of strings
                Possible class values in a multiclass problem

            balance_classes: Boolean
                If True, the algorithm takes into account unbalanced datasets

            C: array of floats
                Centroids matrix

            nf: integer
                Number of bits for the floating part

            N: integer
                Number of

            fsigma: float
                factor to multiply standard sigma value = sqrt(Number of inputs)

            normalize_data: Boolean
                If True, data normalization is applied, irrespectively if it has been previously normalized


        """
        self.model_type = model_type
        self.process_kwargs(model_parameters)  # Warning, this removes other ones...

        if self.pom == 1:
            from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master
            self.MasterCommon = POM1_CommonML_Master(self.workers_addresses, self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'Kmeans':
                from MMLL.models.POM1.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'NN':
                from MMLL.models.POM1.NeuralNetworks.neural_network import NN_Master
                self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))
 
        if self.pom == 2:
            from MMLL.models.POM2.CommonML.POM2_CommonML import POM2_CommonML_Master
            self.MasterCommon = POM2_CommonML_Master(self.workers_addresses, self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'Kmeans':
                from MMLL.models.POM2.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'NN':
                from MMLL.models.POM2.NeuralNetworks.neural_network import NN_Master
                self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

        if self.pom == 3:
            from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master
            self.MasterCommon = POM3_CommonML_Master(self.workers_addresses, self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'Kmeans':
                from MMLL.models.POM3.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'NN':
                from MMLL.models.POM3.NeuralNetworks.neural_network import NN_Master
                self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

        if self.pom == 4:
            from MMLL.models.POM4.CommonML.POM4_CommonML import POM4_CommonML_Master
            self.MasterCommon = POM4_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'LR':
                from MMLL.models.POM4.LR.LR import LR_Master
                self.MasterMLmodel = LR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

            if model_type == 'Kmeans':
                from MMLL.models.POM4.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

        if self.pom == 5:
            from MMLL.models.POM5.CommonML.POM5_CommonML import POM5_CommonML_Master
            self.MasterCommon = POM5_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose, cr=self.cr)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'LR':
                from MMLL.models.POM5.LR.LR import LR_Master
                self.MasterMLmodel = LR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

            if model_type == 'Kmeans':
                from MMLL.models.POM5.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, model_parameters=model_parameters)
                self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

        if self.pom == 6:

            from MMLL.models.POM6.CommonML.POM6_CommonML import POM6_CommonML_Master
            self.MasterCommon = POM6_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'XC':
                from MMLL.models.POM6.XC.XC import XC_Master
                self.MasterMLmodel = XC_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)

            if model_type == 'RR':
                from MMLL.models.POM6.RR.RR import RR_Master
                self.MasterMLmodel = RR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)

            if model_type == 'KR_pm':
                from MMLL.models.POM6.KR_pm.KR_pm import KR_pm_Master
                self.MasterMLmodel = KR_pm_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)

            if model_type == 'LC_pm':
                from MMLL.models.POM6.LC_pm.LC_pm import LC_pm_Master
                self.MasterMLmodel = LC_pm_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)

            if model_type == 'MLC_pm':
                from MMLL.models.POM6.MLC_pm.MLC_pm import MLC_pm_Master
                self.MasterMLmodel = MLC_pm_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)

            if model_type == 'Kmeans_pm':
                from MMLL.models.POM6.Kmeans_pm.Kmeans_pm import Kmeans_pm_Master
                self.MasterMLmodel = Kmeans_pm_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)

            '''
            # We setup the necessary variables before training
            self.display('MasterNode_' + self.model_type + ': Innitiating variables')
            self.MasterCommon.reset(self.NI)
            self.MasterMLmodel.reset(self.NI)
            #passing validation values
            self.MasterMLmodel.Xval_b = self.Xval_b
            self.MasterMLmodel.yval = self.yval
            '''

    def fit(self, Xval=None, yval=None):
        """
        Train the Machine Learning Model

        """
        if Xval is not None:
            self.MasterMLmodel.Xval = np.array(Xval)
        else:
            self.MasterMLmodel.Xval = None
        if yval is not None:
            self.MasterMLmodel.yval = np.array(yval)
        else:
            self.MasterMLmodel.yval = None

        if self.pom == 4:
            # We ask for the encrypter to the cryptonode
            self.mn_ask_encrypter()
            # We ask for the encrypted data to the workers
            self.mn_get_encrypted_data()

            try:
                self.MasterMLmodel.send_to = self.MasterCommon.send_to
                self.MasterMLmodel.receive_from = self.MasterCommon.receive_from
                self.MasterMLmodel.state_dict = self.MasterCommon.state_dict
                self.MasterMLmodel.broadcast_addresses = self.MasterCommon.broadcast_addresses
                self.MasterMLmodel.workers_addresses = self.MasterCommon.workers_addresses
                self.MasterMLmodel.cryptonode_address = self.MasterCommon.cryptonode_address
            except:
                pass

        if self.pom == 5:
            # We send the encrypter to all the workers
            self.MasterCommon.send_encrypter()

            try:
                self.MasterMLmodel.send_to = self.MasterCommon.send_to
                self.MasterMLmodel.worker_names = self.MasterCommon.worker_names
            except:
                pass

        if self.pom == 6:
            # Operations run by MasterCommon
            '''
            if self.normalize_data:
                # We ask every worker to communicate the sum of X and y, and N
                self.MasterCommon.get_sumXy()
                try:
                    self.MasterMLmodel.total_sumX = self.MasterCommon.total_sumX
                    self.MasterMLmodel.total_sumy = self.MasterCommon.total_sumy
                    self.MasterMLmodel.total_NP = self.MasterCommon.total_NP
                except:
                    pass
            '''
            # self.classes
            # balance_classes
            if self.balance_classes:
                # We ask every worker to communicate the number of patterns per class
                # Direct communication by now, improved confidentiality with roundrobin 
                self.MasterCommon.get_Npc()
                try:
                    self.MasterMLmodel.aggregated_Npc_dict = self.MasterCommon.aggregated_Npc_dict
                except:
                    pass

            # Operations run by MasterMLmodel
            try:
                self.MasterMLmodel.Xq_prodpk_dict = self.MasterCommon.Xq_prodpk_dict
                self.MasterMLmodel.yq_prodpk_dict = self.MasterCommon.yq_prodpk_dict
            except:
                pass

            try:
                self.MasterMLmodel.NI_dict = self.MasterCommon.NI_dict
            except:
                pass

            try:
                self.MasterMLmodel.send_to = self.MasterCommon.send_to
                self.MasterMLmodel.worker_names = self.MasterCommon.worker_names
            except:
                pass

        ###################  Common to all POMS  ##################
        try:
            self.MasterMLmodel.train_Master()
            # Set this to True if the model has been sucessfully trained.
            self.model_is_trained = True
            self.display('MasterNode: the model has been trained.')
        except Exception as err:
            self.display('MasterNode: Error during training: ', err)
            raise

    def get_model(self):
        """
        Returns the ML model as an object, if it is trained, returns None otherwise

        Parameters
        ----------
        None
        """
        if not self.model_is_trained:
            self.display('MasterNode: Error - Model not trained yet')
            return None
        else:
            return self.MasterMLmodel.model

    def save_model(self, output_filename_model=None):
        """
        Saves the ML model using pickle if it is trained, prints an error otherwise

        Parameters
        ----------
        None
        """

        if not self.model_is_trained:
            self.display('MasterNode: Error - Model not trained yet, nothing to save.')
        else:
            if self.pom==2 or self.pom==3:
                self.display('MasterNode: Error - In POMs 2 and 3, the model is owner by the workers, not the master. Nothing to save.')
                return
            '''
            if output_filename_model is None:
                output_filename_model = './POM' + str(self.pom) + '_' + self.model_type + '_' + self.dataset_name + '_model.pkl'
            '''
            try:
                with open(output_filename_model, 'wb') as f:
                    pickle.dump(self.MasterMLmodel.model, f)
            except:
                output_filename_model = './POM' + str(self.pom) + '_' + self.model_type + '_' + self.dataset_name + '_model.pkl'
                with open(output_filename_model, 'wb') as f:
                    pickle.dump(self.MasterMLmodel.model, f)

            self.display('MasterNode: Model saved at %s' %output_filename_model)

    def terminate_Workers(self, workers_addresses_terminate=None):
        """
        Terminate selected workers

        Parameters
        ----------
        workers_addresses_terminate: list of strings
            List of addresses of workers that must be terminated. If the list is empty, all the workers will stop.

        """
        if workers_addresses_terminate is None:  # All
            workers_addresses_terminate = self.workers_addresses

        self.workers_addresses = list(set(self.workers_addresses) - set(workers_addresses_terminate))
        self.Nworkers = len(self.workers_addresses)

        self.MasterCommon.terminate_Workers(workers_addresses_terminate)


    def set_validation_data(self, dataset_name, Xval=None, yval=None):
        """
        Set data to be used for validation.

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xval: Input data matrix: row = No. patterns , col = No. features
        yval: target vector: row = No. patterns
        add_bias: boolean. If true, it adds a column of ones to the input data matrix
        """
        self.dataset_name = dataset_name
        try:
            self.NPval = Xval.shape[0]
            self.NI = Xval.shape[1]                # Number of inputs
            self.yval = yval
            self.Xval_b = Xval

            if self.Xval_b.shape[0] != self.yval.shape[0] and yval is not None:
                self.display('ERROR: different number of patterns in Xval and yval (%s vs %s)' % (str(self.Xval_b.shape[0]), str(self.yval.shape[0])))
                self.Xval_b = None
                self.yval = None
                self.NPval = 0
                self.display('MasterNode: ***** Validation data NOT VALID. *****')
                return
            else:
                self.display('MasterNode got validation data: %d patterns, %d features' % (self.NPval, self.NI))
        except:
            self.display('MasterNode: ***** Validation data NOT available. *****')
            pass

    def set_validation_data_raw(self, dataset_name, Xval=None, yval=None):
        """
        Set data to be used for validation.

        *****  List of lists... ****

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xval: Input data: list of lists
        yval: target vector: list of lists 
        """
        self.dataset_name = dataset_name
        try:
            self.Xval_b = np.array(Xval)            
            self.yval = np.array(yval)
            self.NPval, self.NI = self.Xval_b.shape

            if self.Xval_b.shape[0] != self.yval.shape[0] and yval is not None:
                self.display('ERROR: different number of patterns in Xval and yval (%s vs %s)' % (str(self.Xval_b.shape[0]), str(self.yval.shape[0])))
                self.Xval_b = None
                self.yval = None
                self.NPval = 0
                self.display('MasterNode: ***** Validation data NOT VALID. *****')
                return
            else:
                self.display('MasterNode got RAW validation data: %d patterns, %d features' % (self.NPval, self.NI))
                try:   # conversion to numeric
                    self.Xval_b = self.Xval_b.astype(float)                
                    self.yval = self.yval.astype(float)
                    self.data_is_ready = True
                except:
                    self.data_is_ready = False
        except:
            self.display('MasterNode: ***** Validation data NOT available. *****')
            pass

    def set_test_data(self, dataset_name, Xtst=None, ytst=None):
        """
        Set data to be used for testing.

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xtst: Input data: list of lists
        ytst: target vector: list of lists 
        """
        self.dataset_name = dataset_name
        try:
            self.NPtst = Xtst.shape[0]
            self.NI = Xtst.shape[1]                # Number of inputs
            self.ytst = ytst
            self.Xtst_b = Xtst

            if self.Xtst_b.shape[0] != self.ytst.shape[0]  and ytst is not None:
                self.display('ERROR: different number of patterns in Xtst and ytst (%s vs %s)' % (str(self.Xval_b.shape[0]), str(self.yval.shape[0])))
                self.Xtst_b = None
                self.ytst = None
                self.NPtst = 0
                self.display('MasterNode: ***** Test data NOT VALID. *****')
                return
            else:
                self.display('MasterNode got test data: %d patterns, %d features' % (self.NPtst, self.NI))
        except:
            self.display('MasterNode: ***** Test data NOT available. *****')
            pass

    def set_test_data_raw(self, dataset_name, Xtst=None, ytst=None):
        """
        Set data to be used for validation.

        *****  List of lists... ****

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xtst: Input data matrix: row = No. patterns , col = No. features
        ytst: target vector: row = No. patterns
        add_bias: boolean. If true, it adds a column of ones to the input data matrix
        """
        self.dataset_name = dataset_name
        try:
            self.Xtst_b = np.array(Xtst)            
            self.ytst = np.array(ytst)
            self.NPtst, self.NI = self.Xtst_b.shape

            if self.Xtst_b.shape[0] != self.ytst.shape[0] and ytst is not None:
                self.display('ERROR: different number of patterns in Xtst and ytst (%s vs %s)' % (str(self.Xtst_b.shape[0]), str(self.ytst.shape[0])))
                self.Xtst_b = None
                self.ytst = None
                self.NPtst = 0
                self.display('MasterNode: ***** Test data NOT VALID. *****')
                return
            else:
                self.display('MasterNode got RAW test data: %d patterns, %d features' % (self.NPtst, self.NI))
                try:   # try conversion to numeric
                    self.Xtst_b = self.Xtst_b.astype(float)                
                    self.ytst = self.ytst.astype(float)
                    self.data_is_ready = True
                except:
                    self.data_is_ready = False
        except:
            self.display('MasterNode: ***** Test data NOT available. *****')
            pass

    '''
    OLD
    def normalizer_fit(self, type='global_mean_std'):
        """
        Terminate selected workers

        Parameters
        ----------
        workers_addresses_terminate: list of strings
            List of addresses of workers that must be terminated. If the list is empty, all the workers will stop.

        """

        if self.pom == 6:
            if type == 'global_mean_std':
                # We ask every worker to communicate the sum of X and y, and N
                self.MasterCommon.get_sumX()
                # We pass the values to MasterMLmodel
                self.MasterMLmodel.total_sumX = self.MasterCommon.total_sumX
                self.MasterMLmodel.total_NP = self.MasterCommon.total_NP

                mean_values = self.MasterCommon.total_sumX / self.MasterCommon.total_NP
                
                x_m_2 = self.MasterCommon.get_X_minus_mean_squared(mean_values=mean_values) / self.MasterCommon.total_NP
                std = np.sqrt(x_m_2)
                
                model = normalize_model()
                model.mean = mean_values
                model.std = std

            return model

    def normalize_data_workers(self, model=None):
        """
        Normalize data at workers, by sending the preprocessing model

        Parameters
        ----------
       model: object
            Preprocessing object.

        """

        if model is not None:
            self.MasterCommon.send_preprocess(model)
        else:
            print('Error: The preprocessing object is missing.')
    '''

    def normalizer_fit_transform_workers(self, data_description, transform_num='global_mean_std'):
        """
        Adjust the normalizer parameters and transform the training data in the workers

        Parameters
        ----------
        type: string
            Type of normalization of the numerical inputs. Binary inputs are not transformed, and
            categorical inputs are transformed using a one-hot encoding.

        """

        if self.pom == 1:
            if transform_num == 'global_mean_std' or transform_num == 'global_min_max':
                model = normalize_model()
                model.method = transform_num
                model.data_description = data_description

                from MMLL.preprocessors.POM1.Preprocessor import Preprocessor_Master
                preprocessor = Preprocessor_Master(self.comms, self.logger, data_description, self.verbose, normalization_type=transform_num, model=model)
                preprocessor.normalize_Master()
                model = preprocessor.model

                return model


        if self.pom == 6:
            if transform_num == 'global_mean_std':
                # We ask every worker to communicate the sum of X and y, and N
                self.MasterCommon.get_sumX(data_description)
                # We pass the values to MasterMLmodel
                self.MasterMLmodel.total_sumX = self.MasterCommon.total_sumX
                self.MasterMLmodel.total_NP = self.MasterCommon.total_NP

                mean_values = self.MasterCommon.total_sumX / self.MasterCommon.total_NP
                
                x_m_2 = self.MasterCommon.get_X_minus_mean_squared(mean_values, data_description) / self.MasterCommon.total_NP
                std = np.sqrt(x_m_2)
                
                # invert first example
                #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

                onehot_encodings = {}
                label_encodings = {}

                for k in range(data_description['NI']):
                    if data_description['input_types'][k]['type'] == 'cat':
                        categories = data_description['input_types'][k]['values']
                        label_encoder = LabelEncoder()
                        label_encoder.fit(categories)
                        label_encodings.update({k: label_encoder})
                        integer_encoded = label_encoder.transform(categories).reshape((-1, 1))
                        # Creating one-hot-encoding object
                        onehotencoder = OneHotEncoder(sparse=False)
                        onehotencoder.fit(integer_encoded)
                        onehot_encodings.update({k: onehotencoder})
                        onehot_encoded = onehotencoder.transform(np.array([1, 2]).reshape(-1, 1))

                        ### Probando
                        #x = ['?', 'Federal-gov', 'Private']
                        #x_int = label_encoder.transform(x).reshape((-1, 1))
                        # ohe = onehotencoder.transform(x_int) 

                model = normalize_model()
                model.mean = mean_values
                model.std = std
                model.data_description = data_description
                model.label_encodings = label_encodings
                model.onehot_encodings = onehot_encodings

                # Sending the model to the workers, so they preprocess their data
                self.MasterCommon.send_preprocess(model)

            return model

    def mn_ask_encrypter(self):
        """
        Obtain encrypter from cryptonode, under POM 4

        """
        self.MasterCommon.ask_encrypter()
        # We pass the encrypter to the MasterML model
        self.MasterMLmodel.encrypter = self.MasterCommon.encrypter
        print('Masternode WARNING: remove this')
        self.MasterMLmodel.decrypter = self.MasterCommon.decrypter
        return

    def mn_get_encrypted_data(self):
        """
        Obtain Encrypted data from workers, under POM 4

        """
        self.MasterCommon.get_cryptdata()
        
        # We pass the encrypted data  to the MasterML model
        self.MasterMLmodel.X_encr_dict = self.MasterCommon.X_encr_dict
        self.MasterMLmodel.y_encr_dict = self.MasterCommon.y_encr_dict

        self.MasterMLmodel.workers_addresses = self.MasterCommon.workers_addresses
        self.MasterMLmodel.broadcast_addresses = self.MasterCommon.broadcast_addresses
        self.MasterMLmodel.cryptonode_address = self.MasterCommon.cryptonode_address

        self.MasterMLmodel.decrypter = self.MasterCommon.decrypter
        self.MasterMLmodel.encrypter = self.MasterCommon.encrypter

        # We pass the blinding data  to the MasterML model
        self.MasterMLmodel.BX_dict = self.MasterCommon.BX_dict
        self.MasterMLmodel.By_dict = self.MasterCommon.By_dict

        return
