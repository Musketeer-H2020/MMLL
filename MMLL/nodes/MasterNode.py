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
from MMLL.preprocessors.normalizer import normalize_model
from MMLL.preprocessors.data2num import data2num_model
from sklearn.metrics import roc_curve, auc

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
        self.all_workers_addresses = comms.workers_ids      # ALL Workers addresses + ca

        self.Nworkers = len(self.workers_addresses)
        self.master_address = 'ma' 
        self.logger = logger
        self.verbose = verbose # print on screen when true
        
        #self.normalize_data = False                          
        self.classes = None                           
        self.balance_classes = False
        # Processing kwargs
        self.aggregation_type = 'direct'
        self.process_kwargs(kwargs)
        # We assume that the Master may receive a validation and a test set 
        self.Xtst_b = None                                  # Test data (input)
        self.ytst = None                                    # Test data (targets)
        self.Xval_b = None                                  # Validation data (input)
        self.yval = None       # Validation data (targets)
        self.model_is_trained = False
        self.classes = None
        self.display('MasterNode: Initiated')
        self.data_is_ready = False
        self.selected_workers = None

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
            self.MasterCommon = POM1_CommonML_Master(self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'Kmeans':
                from MMLL.models.POM1.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'NN':
                from MMLL.models.POM1.NeuralNetworks.neural_network import NN_Master
                self.MasterMLmodel = NN_Master(self.comms, self.logger, pgd_params=self.pgd_params, verbose=self.verbose,
                                               model_architecture=self.model_architecture,
                                               Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate,
                                               model_averaging=self.model_averaging, optimizer=self.optimizer,
                                               loss=self.loss, metric=self.metric, batch_size=self.batch_size,
                                               num_epochs=self.num_epochs)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'SVM':
                from MMLL.models.POM1.SVM.SVM import SVM_Master
                self.MasterMLmodel = SVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, NmaxiterGD=self.NmaxiterGD, eta=self.eta)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))
 
        if self.pom == 2:
            from MMLL.models.POM2.CommonML.POM2_CommonML import POM2_CommonML_Master
            self.MasterCommon = POM2_CommonML_Master(self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'Kmeans':
                from MMLL.models.POM2.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'NN':
                from MMLL.models.POM2.NeuralNetworks.neural_network import NN_Master
                self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate, model_averaging=self.model_averaging, optimizer=self.optimizer, loss=self.loss, metric=self.metric, batch_size=self.batch_size, num_epochs=self.num_epochs)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'SVM':
                from MMLL.models.POM2.SVM.SVM import SVM_Master
                self.MasterMLmodel = SVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, NmaxiterGD=self.NmaxiterGD, eta=self.eta)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

        if self.pom == 3:
            #from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master
            #self.MasterCommon = POM3_CommonML_Master(self.workers_addresses, self.comms, self.logger, self.verbose)
            #self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'Kmeans':
                from MMLL.models.POM3.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'NN':
                from MMLL.models.POM3.NeuralNetworks.neural_network import NN_Master
                self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate, model_averaging=self.model_averaging, optimizer=self.optimizer, loss=self.loss, metric=self.metric, batch_size=self.batch_size, num_epochs=self.num_epochs)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            elif model_type == 'SVM':
                from MMLL.models.POM3.SVM.SVM import SVM_Master
                self.MasterMLmodel = SVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, NmaxiterGD=self.NmaxiterGD, eta=self.eta)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

            self.MasterCommon = self.MasterMLmodel

        if self.pom == 4:
            from MMLL.models.POM4.CommonML.POM4_CommonML import POM4_CommonML_Master
            self.MasterCommon = POM4_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose, aggregation_type=self.aggregation_type)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)
            self.got_encrypted_data = False

            if model_type == 'LR':
                from MMLL.models.POM4.LR.LR import LR_Master
                self.use_bias = True
                self.MasterMLmodel = LR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

            if model_type == 'Kmeans':
                from MMLL.models.POM4.Kmeans.Kmeans import Kmeans_Master
                self.use_bias = False
                self.MasterMLmodel = Kmeans_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

            # Pinging workers
            self.ping_workers()

        if self.pom == 5:
            from MMLL.models.POM5.CommonML.POM5_CommonML import POM5_CommonML_Master
            self.MasterCommon = POM5_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose, cr=self.cr, aggregation_type=self.aggregation_type)
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
            self.MasterCommon = POM6_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose, aggregation_type=self.aggregation_type)
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

    def fit(self, Xval=None, yval=None, selected_workers=None):
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

        if self.pom in [1, 2, 3]:
            if selected_workers is not None: 
                if set(selected_workers).issubset(self.MasterMLmodel.all_workers_addresses):
                    self.MasterMLmodel.workers_addresses = selected_workers
                    self.MasterMLmodel.Nworkers = len(selected_workers)

        if self.pom == 4:
            if not self.got_encrypted_data:
                # We ask for the encrypter and encrypted data to the cryptonode, but only once...
                self.mn_ask_encrypter()
                # We ask for the encrypted data to the workers
                self.mn_get_encrypted_data(use_bias=self.use_bias)
                self.got_encrypted_data = True

            if selected_workers is not None: 
                self.MasterMLmodel.selected_workers = selected_workers
            else: 
                self.MasterMLmodel.selected_workers = None

            self.MasterMLmodel.send_to = self.MasterCommon.send_to
            self.MasterMLmodel.receive_from = self.MasterCommon.receive_from
            self.MasterMLmodel.state_dict = self.MasterCommon.state_dict
            self.MasterMLmodel.broadcast_addresses = self.MasterCommon.broadcast_addresses
            self.MasterMLmodel.workers_addresses = self.MasterCommon.workers_addresses
            self.MasterMLmodel.cryptonode_address = self.MasterCommon.cryptonode_address

        if self.pom == 5:
            if not self.MasterMLmodel.encrypter_sent:
                # We send the encrypter to all the workers
                self.MasterCommon.send_encrypter()
                self.MasterMLmodel.encrypter_sent = True

            if selected_workers is not None: 
                self.MasterMLmodel.selected_workers = selected_workers
            else: 
                self.MasterMLmodel.selected_workers = None

            try:
                self.MasterMLmodel.send_to = self.MasterCommon.send_to
                self.MasterMLmodel.worker_names = self.MasterCommon.worker_names
            except:
                pass

        if self.pom == 6:

            if selected_workers is not None: 
                self.MasterMLmodel.selected_workers = selected_workers
            else: 
                self.MasterMLmodel.selected_workers = None

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
        try:
            model_is_trained = self.model_is_trained
            if not model_is_trained:
                self.display('MasterNode: Error - Model not trained yet')
                return None
            else:
                return self.MasterMLmodel.model
        except:
            self.display('ERROR: In this POM, the model is not available at MasterNode.')
            return None


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

    '''
    def terminate_Workers(self, workers_addresses_terminate=None):
        """
        Terminate selected workers

        Parameters
        ----------
        workers_addresses_terminate: list of strings
            List of addresses of workers that must be terminated. If the list is empty, all the workers will stop.

        """
        print('STOP AT masternode')
        import code
        code.interact(local=locals())



        if workers_addresses_terminate is None:  # All
            workers_addresses_terminate = self.workers_addresses

        self.workers_addresses = list(set(self.workers_addresses) - set(workers_addresses_terminate))
        self.Nworkers = len(self.workers_addresses)

        self.MasterCommon.terminate_Workers(workers_addresses_terminate)
    '''

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

    def ping_workers(self):
        if self.pom == 4:
            self.MasterCommon.ping_dict = {}
            self.MasterCommon.send_ping_workers()

            true_workers_addresses = []
            self.MasterCommon.cryptonode_address = None

            for key in self.MasterCommon.ping_dict:
                if self.MasterCommon.ping_dict[key]['name'] == 'worker':
                    true_workers_addresses.append(key)
                else:
                    self.MasterCommon.cryptonode_address = key

            # removing cryptonode from workers_addresses
            self.MasterCommon.workers_addresses = true_workers_addresses

            # Maybe not needed
            self.MasterCommon.broadcast_addresses = true_workers_addresses
            
            self.display('Identified workers: ' + str(self.MasterCommon.workers_addresses))
            self.display('Identified cryptonode as worker %s: address %s'% (str(self.MasterCommon.cryptonode_address), str(self.MasterCommon.send_to[self.MasterCommon.cryptonode_address])))

            #we update self.state_dict with the new list of workers_addresses
            self.MasterCommon.state_dict = {}
            for waddr in self.MasterCommon.workers_addresses:
                self.MasterCommon.state_dict.update({waddr: ''})

            # We do not need to modify these...
            #self.MasterCommon.send_to = {}
            #self.MasterCommon.receive_from = {}
            #for k in range(self.Nworkers):
            #    self.send_to.update({str(k): workers_addresses[k]})
            #    self.receive_from.update({workers_addresses[k]: str(k)})

        return                   

    def normalizer_fit_transform_workers(self, input_data_description, transform_num='global_mean_std', which_variables='num'):
        """
        Adjust the normalizer parameters and transform the training data in the workers

        Parameters
        ----------
        type: string
            Type of normalization of the numerical inputs. Binary inputs are not transformed, and
            categorical inputs are transformed using a one-hot encoding.

        """

        if self.pom in [1, 2, 3]:
            if transform_num == 'global_mean_std' or transform_num == 'global_min_max':
                model = normalize_model()
                model.method = transform_num
                model.input_data_description = input_data_description

                from MMLL.preprocessors.POM1.Preprocessor import Preprocessor_Master
                preprocessor = Preprocessor_Master(self.comms, self.logger, input_data_description, self.verbose, normalization_type=transform_num, model=model)
                preprocessor.normalize_Master()
                model = preprocessor.model

                return model

        """
        elif self.pom == 2:
            if transform_num == 'global_mean_std' or transform_num == 'global_min_max':
                
                model = normalize_model()
                model.method = transform_num
                model.input_data_description = input_data_description

                from MMLL.preprocessors.POM2.Preprocessor import Preprocessor_Master
                preprocessor = Preprocessor_Master(self.comms, self.logger, input_data_description, self.verbose, normalization_type=transform_num, model=model)
                preprocessor.normalize_Master()
                # Store the public key in master
                self.MasterMLmodel.public_key = preprocessor.public_key
                model = preprocessor.model

                return model

        elif self.pom == 3:
            if transform_num == 'global_mean_std' or transform_num == 'global_min_max':
                model = normalize_model()
                model.method = transform_num
                model.input_data_description = input_data_description

                self.MasterMLmodel.Preprocessing(model)

                #from MMLL.preprocessors.POM3.Preprocessor import Preprocessor_Master
                #preprocessor = Preprocessor_Master(self.comms, self.logger, data_description, self.verbose, normalization_type=transform_num, model=model)
                #preprocessor.normalize_Master()
                #model = preprocessor.model

                #return model
        """

        if self.pom in [4, 5, 6]:

            model = normalize_model()
            model.input_data_description = input_data_description
            model.mean = None
            model.std = None

            if transform_num == 'global_mean_std' or transform_num == 'global_mean_std_iter2':
                # We ask every worker to communicate the sum of X and y, and N
                self.MasterCommon.sumX_dict = {}
                self.MasterCommon.NP_dict = {}
                self.MasterCommon.get_sumX(input_data_description, which_variables)
                # We pass the values to MasterMLmodel
                self.MasterMLmodel.total_sumX = self.MasterCommon.total_sumX
                self.MasterMLmodel.total_NP = self.MasterCommon.total_NP

                mean_values = self.MasterCommon.total_sumX / self.MasterCommon.total_NP
                
                x_m_2 = self.MasterCommon.get_X_minus_mean_squared(mean_values, input_data_description, which_variables) / self.MasterCommon.total_NP
                std = np.sqrt(x_m_2)
                # 
                model.mean = mean_values
                model.std = std
                model.which_variables = which_variables

            if transform_num == 'global_min_max':
                # We ask every worker to communicate the min and max of X, and N
                self.MasterCommon.get_min_max_X(input_data_description)

                #self.total_minX
                #self.total_maxX
                # We pass the values to MasterMLmodel
                self.MasterMLmodel.total_minX = self.MasterCommon.total_minX
                self.MasterMLmodel.total_maxX = self.MasterCommon.total_maxX
                self.MasterMLmodel.total_NP = self.MasterCommon.total_NP

                model.min = self.MasterCommon.total_minX
                model.max = self.MasterCommon.total_maxX

            '''
            # This part could be unnecessay if we always call data2num beore normalizing...
            onehot_encodings = {}
            label_encodings = {}

            for k in range(input_data_description['NI']):
                if input_data_description['input_types'][k]['type'] == 'cat':
                    categories = input_data_description['input_types'][k]['values']
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

            model.label_encodings = label_encodings
            model.onehot_encodings = onehot_encodings
            '''
            # Sending the model to the workers, so they preprocess their data
            self.MasterCommon.send_preprocess(model)

            return model

    def data2num_transform_workers(self, input_data_description):
        """
        Convert data to numerical vector

        Parameters
        ----------
        data_description, type: string
            Type of normalization of the numerical inputs. Binary inputs are not transformed, and
            categorical inputs are transformed using a one-hot encoding.

        """
        if self.pom in [1, 2, 3]:
            model = None
            new_input_data_description = input_data_description
            errors = None

            try:
                types = [x['type'] for x in input_data_description['input_types']]

                if 'cat' in types:  # there is something to transform
                    model = data2num_model(input_data_description=input_data_description)
                    # Sending the model to the workers, so they preprocess their data
                    worker_errors = self.MasterCommon.send_preprocessor(model)
                    new_input_data_description = model.new_input_data_description
                else:
                    model = None
                    worker_errors = None
            except Exception as err:
                self.display('MasterNode: Error at data2num_transform_workers: ', err)
                raise

            return model, new_input_data_description, worker_errors


        model = None
        new_input_data_description = input_data_description
        errors = None

        if self.pom in [4, 5, 6]:
            try:
                types = [x['type'] for x in input_data_description['input_types']]

                if 'cat' in types:  # there is something to transform
                    model = data2num_model(input_data_description=input_data_description)
                    # Sending the model to the workers, so they preprocess their data

                    worker_errors = self.MasterCommon.send_preprocess(model)
                    new_input_data_description = model.new_input_data_description
                else:
                    model = None
                    worker_errors = None
            except:
                self.display('*** Error at data2num_transform_workers ***')

                print('STOP AT ')
                import code
                code.interact(local=locals())



                model = None
                pass

        return model, new_input_data_description, worker_errors


    def data2num_transform_workers_V(self, input_data_description):
        """
        Convert data to numerical vector

        Parameters
        ----------
        data_description, type: string
            Type of normalization of the numerical inputs. Binary inputs are not transformed, and
            categorical inputs are transformed using a one-hot encoding.

        """
        model = None
        errors = None
        new_input_data_description = input_data_description
        worker_errors = None

        if self.pom in [1, 2, 3, 4, 5, 6]:
            try:
                model = self.get_data2num_model(input_data_description)

                # Sending a local data2num model to the workers, so they preprocess their data
                input_data_description_dict, target_data_description_dict, worker_errors_dict = self.MasterCommon.data2num_at_workers_V()

                #construct new_input_data_description
                new_types = []
                for waddr in self.MasterCommon.workers_addresses:
                    new_types = new_types + input_data_description_dict[waddr]['input_types']

                new_input_data_description = {
                                            "NI": len(new_types), 
                                            "input_types": new_types
                                            }

                worker_errors = list(set(worker_errors_dict.values()))
                if len(worker_errors) == 1 and worker_errors[0] is None:
                    worker_errors = None
                else:
                    worker_errors = worker_errors_dict

            except:
                self.display('*** Error at data2num_transform_workers ***')
                print('STOP HERE ')
                import code
                code.interact(local=locals())
                model = None
                pass

        return model, new_input_data_description, worker_errors


    def preprocess_data_at_workers(self, prep):
        """
        Send preprocessing object to workers

        Parameters
        ----------
        prep, type: object
            preprocessing object

        """
        model = None
        new_data_description = None
        worker_errors = None

        if self.pom in [1, 2, 3]:
            worker_errors = self.MasterCommon.send_preprocessor(prep)
            if len(worker_errors) == 0:
                worker_errors = None

        if self.pom in [4, 5, 6]:
            worker_errors = self.MasterCommon.send_preprocess(prep)
            if len(worker_errors) == 0:
                worker_errors = None

        return worker_errors


    def preprocess_data_at_workers_V(self, prep):
        """
        Send preprocessing object to workers

        Parameters
        ----------
        prep, type: object
            preprocessing object

        """
        model = None
        new_data_description = None
        worker_errors = None

        if self.pom in [1, 2, 3, 4, 5, 6]:
            worker_errors = self.MasterCommon.send_preprocess_V(prep)
            if len(worker_errors) == 0:
                worker_errors = None

            # Update normalizer with mean/std values
            if prep.name == 'normalization':
                mean = []
                std = []
                for waddr in self.MasterCommon.workers_addresses:
                    mean.append(self.MasterCommon.mean_dict[waddr])
                    std.append(self.MasterCommon.std_dict[waddr])

                prep.mean = np.hstack(mean) 
                prep.std = np.hstack(std) 
                
            if prep.name == 'missing_data_imputation_V':               
                mean = []
                for waddr in self.MasterCommon.workers_addresses:
                    mean.append(self.MasterCommon.mean_dict[waddr])
                prep.mean = np.hstack(mean) 
                prep.std = None 

            if prep.name in ['normalization', 'missing_data_imputation_V']:
                return worker_errors, prep
            else:
                return worker_errors


    def rank_features_gfs(self, Xval, yval, input_data_description, method, NF=None, stop_incr=None):
        # Compute a greedy feature selection based on workers data

        ranked_inputs = None
        performance_evolution = None

        if self.pom in [1, 2, 3]:
            Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb()

        if self.pom in [4, 5, 6]:

            # We get the Rxx_b and rxy_b
            if method == 'direct':
                Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb_direct()

            if method == 'roundrobin':
                NF = input_data_description['NI'] + 1 # account for bias
                Rxyb_ini = np.random.uniform(-9e5, 9e5, (NF, NF))
                rxyb_ini = np.random.uniform(-9e5, 9e5, (NF, 1))
                Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb_roundrobin(Rxyb_ini, rxyb_ini)

        ranked_inputs, performance_evolution = self.compute_gfs(Rxy_b, rxy_b, Xval, yval, NF=NF, stop_incr=stop_incr)       

        return ranked_inputs, performance_evolution     


    def pca_fit_transform_workers(self, input_data_description, method, NF):
        # Compute a PCA based on workers data
        if self.pom in [1, 2, 3]:
            Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb()

        if self.pom in [4, 5, 6]:

            # We get the Rxx_b and rxy_b
            if method == 'direct':
                Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb_direct()

            if method == 'roundrobin':
                NF = input_data_description['NI'] + 1 # account for bias
                Rxyb_ini = np.random.uniform(-9e5, 9e5, (NF, NF))
                rxyb_ini = np.random.uniform(-9e5, 9e5, (NF, 1))
                Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb_roundrobin(Rxyb_ini, rxyb_ini)

        Rxx = Rxy_b[1:,1:] / Rxy_b[0,0]

        from MMLL.preprocessors.pca import PCA_model
        pca_model = PCA_model()
        pca_model.fit(Rxx, NF)

        workers_errors = self.preprocess_data_at_workers(pca_model)
        if workers_errors is not None:
            if len(workers_errors) == 0:
                workers_errors = None

        new_input_data_description = {'NI': NF, 'input_types': [{'type': 'num', 'name': 'pca component'}] * NF}

        return pca_model, new_input_data_description, workers_errors     


    def compute_statistics(self, X, y, stats_list):
        # Compute statistics of local data

        stats_dict = None
        if self.pom in [1, 2, 3, 4, 5, 6]:
            stats_dict = self.MasterCommon.compute_stats(X, y, stats_list)

        return stats_dict 


    def get_statistics_workers(self, stats_list):
        """
        Get the statistics from the workers

        Parameters
        ----------

        """
        stats_dict_workers = None

        if self.pom in [1, 2, 3]:    
            stats_dict_workers = self.MasterCommon.get_stats(stats_list)

        if self.pom in [4, 5, 6]:    
            self.MasterCommon.get_stats(stats_list)
            stats_dict_workers = self.MasterCommon.stats_dict
        return stats_dict_workers



    def get_task_alignment(self, Xval, yval):
        """
        Compute the task alignment of the workers

        Parameters
        ----------

        """
        ta_dict = None

        if self.pom in [1, 2, 3]:
            ta_dict = {}
            stat = 'rxy'
            stats_list = [stat]
            stats_dict_workers = self.MasterCommon.get_stats(stats_list)
            stats_dict_val = self.MasterCommon.compute_stats(Xval, yval, stats_list)
            rxy_val = stats_dict_val[stat]

            for worker in self.MasterCommon.workers_addresses:
                rxy = stats_dict_workers[worker][stat]
                ta_dict[worker] =  np.dot(rxy_val.T, rxy).ravel()[0]

        if self.pom in [4, 5, 6]:
            ta_dict = {}
            stats_list = ['rxy']
            self.MasterCommon.get_stats(stats_list)

            rxy_val = np.dot(Xval.T, np.array(yval).astype(float))
            #rxy_val_ = rxy_val - np.mean(rxy_val)
            rxy_val_ = rxy_val
            rxy_val_ = rxy_val_ / np.linalg.norm(rxy_val_)
            #np.dot(rxy_val_.T, rxy_val_)[0][0]

            for worker in self.MasterCommon.workers_addresses:
                rxy = self.MasterCommon.stats_dict[worker]['rxy']
                #rxy_ = rxy - np.mean(rxy)
                rxy_ = rxy
                rxy_ = rxy_ / np.linalg.norm(rxy_)
                ta_dict.update({worker: np.dot(rxy_val_.T, rxy_).ravel()[0]})
        return ta_dict


    def check_data_at_workers(self, input_data_description, target_data_description):
        """
        Checking data at workers. Returns None if everything is OK 

        Parameters
        ----------
        input_data_description: dict describing input data
        target_data_description: dict describing target data

        """
        err = 'Method not available for this POM'
        bad_workers = None

        if self.pom in [1, 2, 3, 4, 5, 6]:
            err = None
            worker_errors_dict = self.MasterCommon.send_check(input_data_description, target_data_description)
            err = ''
            bad_workers = []
            for worker in self.MasterCommon.workers_addresses:
                if len(worker_errors_dict[worker]) > 0:
                    err += 'Error at worker %s: %s\n'% (worker, worker_errors_dict[worker])
                    bad_workers.append(worker)
            if len(err) == 0:
                err = None

        return err, bad_workers


    def deep_learning_transform_workers(self, data_description):
        """
        Convert images to numerical vector using Deep Learning

        Parameters
        ----------
        data_description, type: string
            

        """
        model = None
        new_data_description = None
        errors = None

        if self.pom in [1, 2, 3]:
            try:
                if data_description['input_types'][0]['type'] == 'matrix' and len(data_description['input_types']) == 1:
                    from MMLL.preprocessors.deep_learning import deep_learning_model
                    model = deep_learning_model(data_description)
                    self.display('Masternode: Deep learning model is ready')

                    new_input_types = [{'type': 'num', 'name': 'deep learning feature'}] * 1000
                    new_input_data_description = {
                                "NI": len(new_input_types), 
                                "input_types": new_input_types
                                }
                    model.name = 'deep_learning'

                    # Sending the model to the workers, so they preprocess their data
                    worker_errors = self.MasterCommon.send_preprocessor(model)
            except Exception as err:
                self.display('MasterNode: Error at data2num_transform_workers: ', err)
                raise

        if self.pom in [4, 5, 6]:
            try:
                if data_description['input_types'][0]['type'] == 'matrix' and len(data_description['input_types']) == 1:
                    from MMLL.preprocessors.deep_learning import deep_learning_model
                    model = deep_learning_model(data_description)
                    self.display('Deep learning model is ready')

                    new_input_types = [{'type': 'num', 'name': 'deep learning feature'}] * 1000
                    new_input_data_description = {
                                "NI": len(new_input_types), 
                                "input_types": new_input_types
                                }
                    model.name = 'deep_learning'

                    # Sending the model to the workers, so they preprocess their data
                    worker_errors = self.MasterCommon.send_preprocess(model)
            except:
                self.display('*** Error at deep_learning_transform_workers ***')
                model = None
                pass

        return model, new_input_data_description, worker_errors

    def get_vocabulary_workers(self, data_description, init_vocab_dict=None):
        """
        Get vocabulary from all workers

        Parameters
        ----------
        type: string
            Type of normalization of the numerical inputs. Binary inputs are not transformed, and
            categorical inputs are transformed using a one-hot encoding.

        """
        if self.pom in [1, 2, 3]:
            vocab = self.MasterCommon.get_vocabulary()
            vocab, global_df_dict_filtered = self.MasterCommon.get_df(vocab)
 
            return vocab, global_df_dict_filtered

        if self.pom == 6 or self.pom == 5 or self.pom == 4:

            if self.aggregation_type == 'roundrobin':
                self.init_vocab_dict = init_vocab_dict

            self.MasterCommon.get_vocabulary()   # for both direct and roundrobin


            if self.aggregation_type == 'roundrobin':
                print('STOP AT MN get_vocabulary_workers, roundrobin')
                import code
                code.interact(local=locals())

            # We obtain the tf and df
            self.MasterCommon.get_df(self.MasterCommon.vocab)   # for both direct and roundrobin

            return self.MasterCommon.vocab, self.MasterCommon.global_df_dict_filtered 


    def get_feat_freq_transformer(self, data_description, Max_freq, NF):
        """
        Get features freq from all workers, generate transformer and transform data at workers

        Parameters
        ----------
        type: string
            Type of normalization of the numerical inputs. Binary inputs are not transformed, and
            categorical inputs are transformed using a one-hot encoding.

        """

        if self.pom in [1, 2, 3, 4, 5, 6]:
            #self.Max_freq = Max_freq
            count, NP = self.MasterCommon.get_feat_count()   # for both direct and roundrobin
            
            freq = count/NP

            # Removing values > Max_freq
            freq[freq > Max_freq] = 0

            ind = np.argsort(-1 * freq)
            selected_features = ind[0: NF]

            from MMLL.preprocessors.feature_extract import feature_extract_model
            feature_extractor = feature_extract_model(selected_features, data_description)
            new_input_data_description = feature_extractor.new_input_data_description

            return feature_extractor, new_input_data_description 

    def record_linkage_transform_workers(self, linkage_type='full'):
        """
        Transform data at workers such that features are aligned

        Parameters
        ----------

        """

        if self.pom in [1, 2, 3, 4, 5, 6]:

            hashids = self.MasterCommon.get_hashids(linkage_type)   # for both direct and roundrobin

            # record-linkage transform data at workers
            input_data_description_dict, target_data_description_dict = self.MasterCommon.linkage_hashids_transform_workers(hashids, linkage_type)
            return input_data_description_dict, target_data_description_dict 


    def mn_ask_encrypter(self):
        """
        Obtain encrypter from cryptonode, under POM 4

        """
        self.MasterCommon.ask_encrypter()
        # We pass the encrypter to the MasterML model
        self.MasterMLmodel.encrypter = self.MasterCommon.encrypter
        #print('Masternode WARNING: remove this')
        #self.MasterMLmodel.decrypter = self.MasterCommon.decrypter
        return

    def mn_get_encrypted_data(self, use_bias=False):
        """
        Obtain Encrypted data from workers, under POM 4

        """
        self.MasterCommon.get_cryptdata(use_bias)
        
        # We pass the encrypted data  to the MasterML model
        self.MasterMLmodel.X_encr_dict = self.MasterCommon.X_encr_dict
        self.MasterMLmodel.y_encr_dict = self.MasterCommon.y_encr_dict

        self.MasterMLmodel.workers_addresses = self.MasterCommon.workers_addresses
        self.MasterMLmodel.broadcast_addresses = self.MasterCommon.broadcast_addresses
        self.MasterMLmodel.cryptonode_address = self.MasterCommon.cryptonode_address

        # Masternode should not have the decrypter, just for debugging... 
        #self.MasterMLmodel.decrypter = self.MasterCommon.decrypter
        self.MasterMLmodel.encrypter = self.MasterCommon.encrypter

        # We pass the blinding data  to the MasterML model
        self.MasterMLmodel.BX_dict = self.MasterCommon.BX_dict
        self.MasterMLmodel.By_dict = self.MasterCommon.By_dict

        return


    def stop_workers(self):
        """
        Stop workers and start a new training

        Parameters
        ----------
        None
        """
        if self.pom in [4, 5, 6]:
            self.MasterCommon.stop_workers_()


    def terminate_workers(self, workers_addresses_terminate=None):
        """
        Terminate selected workers

        Parameters
        ----------
        workers_addresses_terminate: list of strings
            List of addresses of workers that must be terminated. If the list is empty, all the workers will stop.

        """
        if self.pom in [1, 2, 3]:
            self.MasterCommon.terminate_workers_(workers_addresses_terminate) # Updates worker_addresses, Nworkers and state_dict in MasterCommon
            self.MasterMLmodel.workers_addresses = self.MasterCommon.workers_addresses
            self.MasterMLmodel.Nworkers = self.MasterCommon.Nworkers
            self.MasterMLmodel.state_dict = self.MasterCommon.state_dict
        
        elif self.pom in [4, 5, 6]:
            '''
            if workers_addresses_terminate is None:  # All
                workers_addresses_terminate = self.workers_addresses

            self.workers_addresses = list(set(self.workers_addresses) - set(workers_addresses_terminate))
            self.Nworkers = len(self.workers_addresses)
            '''
            self.MasterCommon.terminate_workers_(workers_addresses_terminate)

        # We update the list of workers
        #self.MasterCommon.workers_addresses = self.workers_addresses
        #self.MasterMLmodel.workers_addresses = self.workers_addresses
        self.MasterMLmodel.workers_addresses = self.MasterCommon.workers_addresses

    def get_data_value_apriori(self, Xval, yval, stats_list):
        """
        Obtain "A priori" Data Value estimation

        Parameters
        ----------
        Xval
        yval
        stats_list

        """
        if self.pom in [1, 2, 3]:
            ref_stats_dict = self.compute_statistics(Xval, yval, stats_list)
            stats_dict_workers = self.get_statistics_workers(stats_list + ['npatterns'])

            workers = self.MasterCommon.workers_addresses
            Nworkers = len(workers)
            NP = np.zeros((Nworkers, 1))
            dv = np.zeros((Nworkers, 1))

            for kworker in range(Nworkers):
                worker = workers[kworker]
                das = []
                for stat in stats_list:
                    das.append(np.dot(ref_stats_dict[stat].ravel(), stats_dict_workers[worker][stat].ravel()))        
                dv[kworker] = np.mean(np.array(das))
                NP[kworker] = stats_dict_workers[worker]['npatterns']

            dv = dv * NP
            dv = dv / np.sum(dv)

            data_value_dict = {}
            for index, worker in enumerate(self.MasterCommon.workers_addresses):
                data_value_dict[worker] = dv[index]

            return data_value_dict


        if self.pom in [4, 5, 6]:

            ref_stats_dict = self.compute_statistics(Xval, yval, stats_list)
            stats_dict_workers = self.get_statistics_workers(stats_list + ['npatterns'])

            #workers = ['0', '1', '2', '3', '4']
            workers = self.MasterCommon.workers_addresses

            Nworkers = len(workers)
            NP = np.zeros((Nworkers, 1))
            dv = np.zeros((Nworkers, 1))

            for kworker in range(Nworkers):
                worker = workers[kworker]
                das = []
                for stat in stats_list:
                    das.append(np.dot(ref_stats_dict[stat].ravel(), stats_dict_workers[worker][stat].ravel()))        
                dv[kworker] = np.mean(np.array(das))
                NP[kworker] = stats_dict_workers[worker]['npatterns']

            dv = dv * NP
            dv = dv /np.sum(dv)

            return dv.ravel()


    def get_data_value_aposteriori(self, Xval, yval, baseline_auc=0):
        """
        Obtain "A posterior" Data Value estimation

        Parameters
        ----------
        Xval
        yval

        """
        if self.pom in [2, 3]:
            self.display('ERROR: In this POM, the model is not available at MasterNode.')
            return None, None


        if self.pom==1:

            # Estimation with an increasing number of workers
            workers = self.MasterCommon.workers_addresses 
            Nworkers = len(workers)

            candidates = workers[:]
            best_workers = []
            best_aucs = []

            for kround in range(Nworkers):

                Ncandidates = len(candidates)
                aucs_candidates = np.zeros(Ncandidates)

                for kcandidate in range(Ncandidates):
                    worker = candidates[kcandidate]
                    selected_workers = best_workers + [worker]
                    self.display('==========>   Evaluating performance with workers %s' %str(selected_workers))
                    self.fit(Xval=Xval, yval=yval, selected_workers=selected_workers)
                    model = self.get_model()
                    preds_val = model.predict(Xval).ravel()
                    fpr_val, tpr_val, thresholds_val = roc_curve(list(yval), preds_val)
                    roc_auc_val = auc(fpr_val, tpr_val)
                    aucs_candidates[kcandidate] = roc_auc_val

                self.display('-------------------------') 
                self.display('Round %d' %kround)
                self.display('AUC candidates: %s' %str(aucs_candidates) )

                which = np.argmax(aucs_candidates)
                winner_worker = candidates[which]
                best_workers += [winner_worker]
                best_aucs += [aucs_candidates[which]]
                candidates = list(set(candidates) - set([winner_worker]))
                self.display('Winner worker: %s' %(winner_worker))
                self.display('Best workers: %s' %str(best_workers))
                self.display('Best AUCs: %s' %str(best_aucs))
                self.display('-------------------------') 

            inc_performance = np.zeros(Nworkers)
            # first increment with respect to baseline
            inc0 = best_aucs[0] - baseline_auc
            if inc0 > 0:
                inc_performance[0] = inc0

            for kworker in range(1, Nworkers):
                inc = best_aucs[kworker] - best_aucs[kworker - 1]
                if inc > 0:
                    inc_performance[kworker] = inc

            totalsum = np.sum(inc_performance)
            if totalsum == 0:
                dv = np.zeros(Nworkers)
            else:
                dv = inc_performance / totalsum

            return dv.ravel(), best_workers


        if self.pom in [4, 5, 6]:

            # Estimation with an increasing number of workers
            workers = self.MasterCommon.workers_addresses 
            Nworkers = len(workers)

            candidates = workers[:]
            best_workers = []
            best_aucs = []

            for kround in range(Nworkers):

                Ncandidates = len(candidates)
                aucs_candidates = np.zeros(Ncandidates)

                for kcandidate in range(Ncandidates):
                    try:
                        worker = candidates[kcandidate]
                        selected_workers = best_workers + [worker]
                        print('==========>   Evaluating performance with workers ', str(selected_workers))
                        self.fit(Xval=Xval, yval=yval, selected_workers=selected_workers)
                        model = self.get_model()
                        Xval_b = self.add_bias(Xval)
                        preds_val = model.predict(Xval_b).ravel()
                        fpr_val, tpr_val, thresholds_val = roc_curve(list(yval), preds_val)
                        roc_auc_val = auc(fpr_val, tpr_val)
                        aucs_candidates[kcandidate] = roc_auc_val
                    except:
                        print('ERROR AT get_data_value_aposteriori')
                        import code
                        code.interact(local=locals())
                        pass
                        
                print('-------------------------')
                print(kround)
                print(aucs_candidates)

                which = np.argmax(aucs_candidates)
                winner_worker = candidates[which]
                best_workers += [winner_worker]
                best_aucs += [aucs_candidates[which]]
                candidates = list(set(candidates) - set([winner_worker]))
                print(winner_worker)
                print(best_workers)
                print(best_aucs)
                print('-------------------------')

            inc_performance = np.zeros(Nworkers)
            # first increment with respect to baseline
            inc0 = best_aucs[0] - baseline_auc
            if inc0 > 0:
                inc_performance[0] = inc0

            for kworker in range(1, Nworkers):
                inc = best_aucs[kworker] - best_aucs[kworker - 1]
                if inc > 0:
                    inc_performance[kworker] = inc

            totalsum = np.sum(inc_performance)
            if totalsum == 0:
                dv = np.zeros(Nworkers)
            else:
                dv = inc_performance / totalsum

            return dv.ravel(), best_workers

