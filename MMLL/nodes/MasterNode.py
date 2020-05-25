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
        self.workers_addresses = comms.workers_ids          # Workers addresses
        self.Nworkers = len(self.workers_addresses)
        self.master_address = 'ma' 
        self.logger = logger
        self.verbose = verbose                              # print on screen when true
        self.normalize_data = False                           
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
                self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate, Xval_b=self.Xval_b, yval=self.yval, Xtest_b=self.Xtst_b, ytest=self.ytst)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))
 
        if self.pom == 2:
            from MMLL.models.POM2.CommonML.POM2_CommonML import POM2_CommonML_Master
            self.MasterCommon = POM2_CommonML_Master(self.workers_addresses, self.platform, self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'Kmeans':
                from MMLL.models.POM2.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.master_address, self.workers_addresses, self.platform, self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

        if self.pom == 3:
            from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master
            self.MasterCommon = POM3_CommonML_Master(self.workers_addresses, self.platform, self.comms, self.logger, self.verbose)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

            if model_type == 'Kmeans':
                from MMLL.models.POM3.Kmeans.Kmeans import Kmeans_Master
                self.MasterMLmodel = Kmeans_Master(self.master_address, self.workers_addresses, self.platform, self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                self.display('MasterNode: Created %s model, POM = %d' % (model_type, self.pom))

        if self.pom == 5:
            from MMLL.models.POM5.CommonML.POM5_CommonML import POM5_CommonML_Master
            self.MasterCommon = POM5_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose, cr=self.cr)
            self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

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
                self.MasterMLmodel = LC_pm_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Nmaxiter=self.Nmaxiter, regularization=self.regularization, classes=self.classes, balance_classes=self.balance_classes, Xval_b = self.Xval_b, yval = self.yval)

            if model_type == 'Kmeans_pm':
                from MMLL.models.POM6.Kmeans_pm.Kmeans_pm import Kmeans_pm_Master
                self.MasterMLmodel = Kmeans_pm_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)


            # We setup the necessary variables before training
            self.display('MasterNode_' + self.model_type + ': Innitiating variables')
            self.MasterCommon.reset(self.NI)
            self.MasterMLmodel.reset(self.NI)
            #passing validation values
            self.MasterMLmodel.Xval_b = self.Xval_b
            self.MasterMLmodel.yval = self.yval


    def fit(self):
        """
        Train the Machine Learning Model

        """
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
            if self.normalize_data:
                # We ask every worker to communicate the sum of X and y, and N
                self.MasterCommon.get_sumXy()
                try:
                    self.MasterMLmodel.total_sumX = self.MasterCommon.total_sumX
                    self.MasterMLmodel.total_sumy = self.MasterCommon.total_sumy
                    self.MasterMLmodel.total_NP = self.MasterCommon.total_NP
                except:
                    pass

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
        self.MasterMLmodel.train_Master()
        # Set this to True if the model has been sucessfully trained.
        self.model_is_trained = True

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


    def set_test_data(self, dataset_name, Xtst=None, ytst=None):
        """
        Set data to be used for testing.

        Parameters
        ----------
        dataset_name: (string): dataset name
        Xtst: Input data matrix: row = No. patterns , col = No. features
        ytst: target vector: row = No. patterns
        add_bias: boolean. If true, it adds a column of ones to the input data matrix
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
