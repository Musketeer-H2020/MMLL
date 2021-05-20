# -*- coding: utf-8 -*-
'''
MasterNode Class.
'''

__author__ = "Angel Navia-Vázquez, Marcos Fernández"
__date__ = "May 2021"


import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..")
from sklearn.metrics import roc_curve, auc

from MMLL.Common_to_all_objects import Common_to_all_objects
from MMLL.preprocessors.normalizer import normalize_model
from MMLL.preprocessors.data2num import data2num_model



class MasterNode(Common_to_all_objects):
    """
    This class represents the main process associated to the Master Node, and serves to 
    coordinate the training procedure under the different POMs.
    """
    def __init__(self, pom, comms, logger, verbose=False, **kwargs):
        """
        Creates a :class:`MasterNode` instance.

        Parameters
        ----------
        pom: int
            The selected Privacy Operation Mode.

        comms: comms object instance
            Object providing communications.

        logger: class:`logging.Logger`
            Logging object instance.

        verbose: boolean
            Indicates if messages are print or not on screen.

        **kwargs: Variable keyword arguments.
        """
        try:
            self.pom = pom                                      # Selected POM
            self.comms = comms                                  # comms library
            self.workers_addresses = comms.workers_ids          # Workers addresses + ca
            self.all_workers_addresses = comms.workers_ids      # ALL Workers addresses + ca
            self.Nworkers = len(self.workers_addresses)
            self.master_address = 'ma' 
            self.logger = logger
            self.verbose = verbose # print on screen when true
            self.classes = None                           
            self.balance_classes = False
            self.aggregation_type = 'direct'
            self.process_kwargs(kwargs)
            self.Xtst_b = None                                  # Test data (input)
            self.ytst = None                                    # Test data (targets)
            self.Xval_b = None                                  # Validation data (input)
            self.yval = None                                    # Validation data (targets)
            self.model_is_trained = False
            self.classes = None
            self.display('MasterNode: Initiated')
            self.data_is_ready = False
            self.selected_workers = None
            self.Tmax = None
        except Exception as err:
            self.display('MasterNode: Error at __init__: ', err)
            raise


    def create_model_Master(self, model_type, model_parameters=None):
        """
        Create the model object to be used for training at the Master side.

        Parameters
        ----------
        model_type: str
            Type of model to be used.

        model_parameters: dictionary
            Parameters needed by the different models, for example it may contain:

            Nmaxiter: int
                Maximum number of iterations during learning.

            NC: int
                Number of centroids.

            regularization: float
                Regularization parameter.

            C: array of floats
                Centroids matrix.

            nf: int
                Number of bits for the floating part.

            fsigma: float
                Factor to multiply standard sigma value = sqrt(Number of inputs).

            normalize_data: Boolean
                If True, data normalization is applied, irrespectively if it has been previously normalized.
        """
        try:
            self.model_type = model_type
            self.process_kwargs(model_parameters)  # Warning, this removes other ones...

            if self.pom == 1:
                from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master
                self.MasterCommon = POM1_CommonML_Master(self.comms, self.logger, self.verbose)
                self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

                if model_type == 'Kmeans':
                    from MMLL.models.POM1.Kmeans.Kmeans import Kmeans_Master
                    self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'NN':
                    from MMLL.models.POM1.NeuralNetworks.neural_network import NN_Master
                    self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate, model_averaging=self.model_averaging, optimizer=self.optimizer, loss=self.loss, metric=self.metric, batch_size=self.batch_size, num_epochs=self.num_epochs, momentum=self.momentum, nesterov=self.nesterov, Tmax=self.Tmax)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'SVM':
                    from MMLL.models.POM1.SVM.SVM import SVM_Master
                    self.MasterMLmodel = SVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, NmaxiterGD=self.NmaxiterGD, eta=self.eta)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'FBSVM':
                    from MMLL.models.POM1.FBSVM.FBSVM import FBSVM_Master
                    self.MasterMLmodel = FBSVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, num_epochs_worker=self.num_epochs_worker, eps=self.eps, mu=self.mu, NI=self.NI, minvalue=self.minvalue, maxvalue=self.maxvalue)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'DSVM':
                    from MMLL.models.POM1.DSVM.DSVM import DSVM_Master
                    self.MasterMLmodel = DSVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, eps=self.eps, NI=self.NI, minvalue=self.minvalue, maxvalue=self.maxvalue)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))
 
            elif self.pom == 2:
                from MMLL.models.POM2.CommonML.POM2_CommonML import POM2_CommonML_Master
                self.MasterCommon = POM2_CommonML_Master(self.comms, self.logger, self.verbose)
                self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

                if model_type == 'Kmeans':
                    from MMLL.models.POM2.Kmeans.Kmeans import Kmeans_Master
                    self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'NN':
                    from MMLL.models.POM2.NeuralNetworks.neural_network import NN_Master
                    self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate, model_averaging=self.model_averaging, optimizer=self.optimizer, loss=self.loss, metric=self.metric, batch_size=self.batch_size, num_epochs=self.num_epochs)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'SVM':
                    from MMLL.models.POM2.SVM.SVM import SVM_Master
                    self.MasterMLmodel = SVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, NmaxiterGD=self.NmaxiterGD, eta=self.eta)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'FBSVM':
                    from MMLL.models.POM2.FBSVM.FBSVM import FBSVM_Master
                    self.MasterMLmodel = FBSVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, num_epochs_worker=self.num_epochs_worker, eps=self.eps, mu=self.mu, NI=self.NI, minvalue=self.minvalue, maxvalue=self.maxvalue)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

            elif self.pom == 3:

                if model_type == 'Kmeans':
                    from MMLL.models.POM3.Kmeans.Kmeans import Kmeans_Master
                    self.MasterMLmodel = Kmeans_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'NN':
                    from MMLL.models.POM3.NeuralNetworks.neural_network import NN_Master
                    self.MasterMLmodel = NN_Master(self.comms, self.logger, self.verbose, model_architecture=self.model_architecture, Nmaxiter=self.Nmaxiter, learning_rate=self.learning_rate, model_averaging=self.model_averaging, optimizer=self.optimizer, loss=self.loss, metric=self.metric, batch_size=self.batch_size, num_epochs=self.num_epochs)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'SVM':
                    from MMLL.models.POM3.SVM.SVM import SVM_Master
                    self.MasterMLmodel = SVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, NmaxiterGD=self.NmaxiterGD, eta=self.eta)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'FBSVM':
                    from MMLL.models.POM3.FBSVM.FBSVM import FBSVM_Master
                    self.MasterMLmodel = FBSVM_Master(self.comms, self.logger, self.verbose, NC=self.NC, Nmaxiter=self.Nmaxiter, tolerance=self.tolerance, sigma=self.sigma, C=self.C, num_epochs_worker=self.num_epochs_worker, eps=self.eps, mu=self.mu, NI=self.NI, minvalue=self.minvalue, maxvalue=self.maxvalue)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

                self.MasterCommon = self.MasterMLmodel

            elif self.pom == 4:
                from MMLL.models.POM4.CommonML.POM4_CommonML import POM4_CommonML_Master
                self.MasterCommon = POM4_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose, aggregation_type=self.aggregation_type)
                self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)
                self.got_encrypted_data = False

                if model_type == 'LR':
                    from MMLL.models.POM4.LR.LR import LR_Master
                    self.use_bias = True
                    self.MasterMLmodel = LR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)

                elif model_type == 'LC':
                    from MMLL.models.POM4.LC.LC import LC_Master
                    self.use_bias = True
                    self.MasterMLmodel = LC_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'MLC':
                    from MMLL.models.POM4.MLC.MLC import MLC_Master
                    self.use_bias = True
                    self.MasterMLmodel = MLC_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'Kmeans':
                    from MMLL.models.POM4.Kmeans.Kmeans import Kmeans_Master
                    self.use_bias = False
                    self.MasterMLmodel = Kmeans_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'KR':
                    from MMLL.models.POM4.KR.KR import KR_Master
                    self.use_bias = False
                    self.MasterMLmodel = KR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'BSVM':
                    from MMLL.models.POM4.BSVM.BSVM import BSVM_Master
                    self.use_bias = False
                    self.MasterMLmodel = BSVM_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

                # Pinging workers to know the cryptonode address
                self.ping_workers()

            elif self.pom == 5:
                from MMLL.models.POM5.CommonML.POM5_CommonML import POM5_CommonML_Master
                self.MasterCommon = POM5_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose, cr=self.cr, aggregation_type=self.aggregation_type)
                self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

                if model_type == 'LR':
                    from MMLL.models.POM5.LR.LR import LR_Master
                    self.MasterMLmodel = LR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'KR':
                    from MMLL.models.POM5.KR.KR import KR_Master
                    self.MasterMLmodel = KR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'LC':
                    from MMLL.models.POM5.LC.LC import LC_Master
                    self.MasterMLmodel = LC_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'MLC':
                    from MMLL.models.POM5.MLC.MLC import MLC_Master
                    self.MasterMLmodel = MLC_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'Kmeans':
                    from MMLL.models.POM5.Kmeans.Kmeans import Kmeans_Master
                    self.MasterMLmodel = Kmeans_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'BSVM':
                    from MMLL.models.POM5.BSVM.BSVM import BSVM_Master
                    self.MasterMLmodel = BSVM_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'MBSVM':
                    from MMLL.models.POM5.MBSVM.MBSVM import MBSVM_Master
                    self.MasterMLmodel = MBSVM_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

            elif self.pom == 6:

                from MMLL.models.POM6.CommonML.POM6_CommonML import POM6_CommonML_Master
                self.MasterCommon = POM6_CommonML_Master(self.master_address, self.workers_addresses, self.comms, self.logger, self.verbose, aggregation_type=self.aggregation_type)
                self.display('MasterNode: Created CommonML_Master, POM = %d' % self.pom)

                if model_type == 'RR':
                    from MMLL.models.POM6.RR.RR import RR_Master
                    self.MasterMLmodel = RR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'KR':
                    from MMLL.models.POM6.KR.KR import KR_Master
                    self.MasterMLmodel = KR_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'LC':
                    from MMLL.models.POM6.LC.LC import LC_Master
                    self.MasterMLmodel = LC_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'MLC':
                    from MMLL.models.POM6.MLC.MLC import MLC_Master
                    self.MasterMLmodel = MLC_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'Kmeans':
                    from MMLL.models.POM6.Kmeans.Kmeans import Kmeans_Master
                    self.MasterMLmodel = Kmeans_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'BSVM':
                    from MMLL.models.POM6.BSVM.BSVM import BSVM_Master
                    self.MasterMLmodel = BSVM_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                elif model_type == 'MBSVM':
                    from MMLL.models.POM6.MBSVM.MBSVM import MBSVM_Master
                    self.MasterMLmodel = MBSVM_Master(self.master_address, self.workers_addresses, self.model_type, self.comms, self.logger, self.verbose, Xval_b = self.Xval_b, yval = self.yval, model_parameters=model_parameters)
                    self.display('MasterNode: Created %s model, POM = %d' % (self.model_type, self.pom))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at create_model_Master: ', err)
            raise


    def fit(self, Xval=None, yval=None, selected_workers=None):
        """
        Train the Machine Learning Model

        Parameters
        ----------
        Xval: list of lists or numpy array 
            Validation data, one pattern per row.

        yval:  list of lists or numpy array 
            Validation targets, one target per row.

        selected_workers: list of ids
            List of selected workers to operate with.
        """
        try:
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
    
                if self.model_type == 'MLC': # We need the classes
                    self.classes = self.target_data_description['output_type'][0]['values']

                if not self.got_encrypted_data:
                    # We ask for the encrypter and encrypted data to the cryptonode, but only once...
                    self.mn_ask_encrypter()
                    # We ask for the encrypted data to the workers
                    self.mn_get_encrypted_data(use_bias=self.use_bias, classes = self.classes)
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

        except Exception as err:
            self.display('MasterNode: Error at fit: ', err)
            raise


    def get_model(self):
        """
        Returns the ML model as an object, if it is trained, returns None otherwise.

        Parameters
        ----------
        None

        Returns
        -------
        model: ML model
            Machine learning model if it has been trained, None otherwise.
        """
        model = None
        try:
            model_is_trained = self.model_is_trained
            if not model_is_trained:
                self.display('MasterNode: Error - Model not trained yet')
                return model
            else:
                model = self.MasterMLmodel.model
                return model
        except:
            self.display('ERROR: In this POM, the model is not available at MasterNode.')
            return model               


    def set_validation_data(self, dataset_name, Xval=None, yval=None):
        """
        Set data to be used for validation.

        Parameters
        ----------
        dataset_name: string 
            Dataset name.

        Xval: list of lists or numpy array 
            Validation data, one pattern per row.

        yval:  list of lists or numpy array 
            Validation targets, one target per row.
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

        Parameters
        ----------
        dataset_name: string
            Dataset name.

        Xval: list of lists
            Validation data, one pattern per row.

        yval:  list of lists
            Validation targets, one target per row.
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
        dataset_name: string
            Dataset name.

        Xtst: list of lists or numpy array 
            Validation data, one pattern per row.

        ytst:  list of lists or numpy array 
            Validation targets, one target per row.
        """
        self.dataset_name = dataset_name
        try:
            self.NPtst = Xtst.shape[0]
            self.NI = Xtst.shape[1]                # Number of inputs
            self.ytst = ytst
            self.Xtst_b = Xtst

            if self.Xtst_b.shape[0] != self.ytst.shape[0] and ytst is not None:
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

        Parameters
        ----------
        dataset_name: string
            Dataset name.

        Xtst: list of lists
            Validation data, one pattern per row.

        ytst:  list of lists
            Validation targets, one target per row.
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


    def ping_workers(self):
        """
        Send ping message to workers to get address info.

        Parameters
        ----------
        None
        """
        try: 
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

        except Exception as err:
            self.display('MasterNode: Error at ping_workers: ', err)
            raise
                  

    def normalizer_fit_transform_workers(self, input_data_description, transform_num='global_mean_std', which_variables='num'):
        """
        Adjust the normalizer parameters and transform the training data in the workers.

        Parameters
        ----------
        input_data_description: dict
            Description of the input features.

        transform_num: string
            Type of normalization of the numerical inputs. Binary inputs are not transformed, and
            categorical inputs are transformed using a one-hot encoding.

        which_variables: string
            Indicates to which type of features we have to apply the normalization.

        Returns
        -------
        model: normalizer model
            Object to normalize new data.
        """
        try:
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

                else:
                    raise ValueError('MasterNode: Transformation method "%s" not available' %str(transform_num))

            elif self.pom in [4, 5, 6]:
    
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

                    # We pass the values to MasterMLmodel
                    self.MasterMLmodel.total_minX = self.MasterCommon.total_minX
                    self.MasterMLmodel.total_maxX = self.MasterCommon.total_maxX
                    self.MasterMLmodel.total_NP = self.MasterCommon.total_NP

                    model.min = self.MasterCommon.total_minX
                    model.max = self.MasterCommon.total_maxX

                # Sending the model to the workers, so they preprocess their data
                self.MasterCommon.send_preprocess(model)
    
                return model

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at normalizer_fit_transform_workers: ', err)
            raise


    def data2num_transform_workers(self, input_data_description):
        """
        Convert data to numerical vector.

        Parameters
        ----------
        input_data_description: dict
            Description of the input features.
 
        Returns
        -------
        model: transformation model
            Model to transform data.

        new_input_data_description: dict
            New dictionary describing the input data.

        worker_errors: dict
            Dictionary containing the errors (if any) for the different workers.
        """
        model = None
        new_input_data_description = input_data_description
        errors = None

        try:
            types = [x['type'] for x in input_data_description['input_types']]

            if 'cat' in types:  # there is something to transform
                model = data2num_model(input_data_description=input_data_description)

                # Sending the model to the workers, so they preprocess their data
                if self.pom in [1, 2, 3]:
                    worker_errors = self.MasterCommon.send_preprocessor(model)
                elif self.pom in [4, 5, 6]:
                    worker_errors = self.MasterCommon.send_preprocess(model)
                else:
                    raise ValueError('POM %s not available in MMLL' %str(self.pom))

                new_input_data_description = model.new_input_data_description
            else:
                model = None
                worker_errors = None
        except Exception as err:
            self.display('MasterNode: Error at data2num_transform_workers: ', err)
            raise

        return model, new_input_data_description, worker_errors


    def data2num_transform_workers_V(self, input_data_description):
        """
        Convert data to numerical vector in vertical partitioning.

        Parameters
        ----------
        input_data_description: dict
            Description of the input features.

        Returns
        -------
        model: transformation model
            Model to transform data.

        new_input_data_description: dict
            New dictionary describing the input data.

        worker_errors: dict
            Dictionary containing the errors (if any) for the different workers.
        """
        model = None
        errors = None
        new_input_data_description = input_data_description
        worker_errors = None

        try:
            if self.pom in [1, 2, 3, 4, 5, 6]:
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

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except:
            self.display('MasterNode: Error at data2num_transform_workers_V: ', err)
            #print('STOP HERE ')
            #import code
            #code.interact(local=locals())
            #model = None
            #pass
            raise

        return model, new_input_data_description, worker_errors


    def preprocess_data_at_workers(self, prep):
        """
        Send preprocessing object to workers.

        Parameters
        ----------
        prep: object
            Preprocessing object.

        Returns
        -------
        worker_errors: dict
            Dictionary containing the errors (if any) for the different workers.
        """
        model = None
        new_data_description = None
        worker_errors = None

        try:
            if self.pom in [1, 2, 3]:
                worker_errors = self.MasterCommon.send_preprocessor(prep)
            elif self.pom in [4, 5, 6]:
                worker_errors = self.MasterCommon.send_preprocess(prep)
            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))
            
            if len(worker_errors) == 0:
                worker_errors = None

        except Exception as err:
            self.display('MasterNode: Error at preprocess_data_at_workers: ', err)
            raise

        return worker_errors


    def preprocess_data_at_workers_V(self, prep):
        """
        Send preprocessing object to workers for vertical partitioning.

        Parameters
        ----------
        prep: object
            Preprocessing object.

        Returns
        -------
        worker_errors: dict
            Dictionary containing the errors (if any) for the different workers.
        """
        model = None
        new_data_description = None
        worker_errors = None

        try:
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

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at preprocess_data_at_workers_V: ', err)
            raise


    def rank_features_gfs(self, Xval, yval, input_data_description, method, NF=None, stop_incr=None):
        """
        Compute a greedy feature selection based on workers data.

        Parameters
        ----------
        Xval: list of lists or numpy array 
            Validation data, one pattern per row.

        yval: list of lists or numpy array 
            Validation targets, one target per row.

        input_data_description: dict
            Description of the input features.

        method: string
            Type of aggregation method to be used (direct/roundrobin).

        NF: int
            Number of features to retain.

        stop_incr: float
            Stop adding features if this tolerance value is reached.

        Returns
        -------
        ranked_inputs: list of ints
            List of ranked inputs.

        performance_evolution: list of floats
            Model performance as a function of the selected inputs.
        """
        ranked_inputs = None
        performance_evolution = None

        try:
            if self.pom in [1, 2, 3]:
                Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb()

            elif self.pom in [4, 5, 6]:
                # We get the Rxx_b and rxy_b
                if method == 'direct':
                    Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb_direct()

                if method == 'roundrobin':
                    NF = input_data_description['NI'] + 1 # account for bias
                    Rxyb_ini = np.random.uniform(-9e5, 9e5, (NF, NF))
                    rxyb_ini = np.random.uniform(-9e5, 9e5, (NF, 1))
                    Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb_roundrobin(Rxyb_ini, rxyb_ini)

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

            ranked_inputs, performance_evolution = self.compute_gfs(Rxy_b, rxy_b, Xval, yval, NF=NF, stop_incr=stop_incr)

        except Exception as err:
            self.display('MasterNode: Error at rank_features_gfs: ', err)
            raise

        return ranked_inputs, performance_evolution     


    def pca_fit_transform_workers(self, input_data_description, method, NF):
        """
        Compute a PCA transformation based on workers data.

        Parameters
        ----------
        input_data_description: dict
            Description of the input features.

        method: string
            Type of aggregation method to be used (direct/roundrobin).

        NF: int
            Number of features to retain.

        Returns
        ----------
        pca_model: model
            The trained PCA model.

        new_input_data_description: dict
            The new description of the input features.

        workers_errors: list of string
            The list of errors at every worker. Under normal operation it is an empty list.
        """
        try:
            # Compute a PCA based on workers data
            if self.pom in [1, 2, 3]:
                Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb()

            elif self.pom in [4, 5, 6]:

                # We get the Rxx_b and rxy_b
                if method == 'direct':
                    Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb_direct()

                if method == 'roundrobin':
                    NF = input_data_description['NI'] + 1 # account for bias
                    Rxyb_ini = np.random.uniform(-9e5, 9e5, (NF, NF))
                    rxyb_ini = np.random.uniform(-9e5, 9e5, (NF, 1))
                    Rxy_b, rxy_b = self.MasterCommon.get_Rxyb_rxyb_roundrobin(Rxyb_ini, rxyb_ini)

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

            Rxx = Rxy_b[1:,1:] / Rxy_b[0,0]

            from MMLL.preprocessors.pca import PCA_model
            pca_model = PCA_model()
            pca_model.fit(Rxx, NF)
    
            workers_errors = self.preprocess_data_at_workers(pca_model)
            if workers_errors is not None:
                if len(workers_errors) == 0:
                    workers_errors = None

            new_input_data_description = {'NI': NF, 'input_types': [{'type': 'num', 'name': 'pca component'}] * NF}

        except Exception as err:
            self.display('MasterNode: Error at pca_fit_transform_workers: ', err)
            raise

        return pca_model, new_input_data_description, workers_errors     


    def compute_statistics(self, X, y, stats_list):
        """
        Compute statistics of given data.

        Parameters
        ----------
        X: list of lists or numpy array 
            Input data, one pattern per row.

        y: list of lists or numpy array 
            Target data, one target per row.

        Returns
        -------
        stats_list: dict
            The list of statistics that have to be computed (rxy, meanx, medianx, npatterns, stdx, skewx, kurx, perc25, perc75, staty).
        """       
        stats_dict = None

        try:
            if self.pom in [1, 2, 3, 4, 5, 6]:
                stats_dict = self.MasterCommon.compute_stats(X, y, stats_list)
            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at compute_statistics: ', err)
            raise

        return stats_dict 


    def get_statistics_workers(self, stats_list):
        """
        Get the statistics from the workers.

        Parameters
        ----------
        stats_list: list of string
            The list of statistics that have to be computed (rxy, meanx, medianx, npatterns, stdx, skewx, kurx, perc25, perc75, staty).

        Returns
        -------
        stats_dict_workers: dict
            Statistics of every worker.
        """
        stats_dict_workers = None

        try:
            if self.pom in [1, 2, 3]:    
                stats_dict_workers = self.MasterCommon.get_stats(stats_list)
            elif self.pom in [4, 5, 6]:    
                self.MasterCommon.get_stats(stats_list)
                stats_dict_workers = self.MasterCommon.stats_dict
            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at get_statistics_workers: ', err)
            raise

        return stats_dict_workers


    def get_task_alignment(self, Xval, yval):
        """
        Compute the task alignment of the workers.

        Parameters
        ----------
        Xval: list of lists or numpy array 
            Validation data, one pattern per row.

        yval:  list of lists or numpy array 
            Validation targets, one target per row.

        Returns
        -------
        ta_dict: dict
            Task alignment estimation of every worker.

        """
        ta_dict = None

        try:
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

            elif self.pom in [4, 5, 6]:
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

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at get_task_alignment: ', err)
            raise

        return ta_dict


    def check_data_at_workers(self, input_data_description=None, target_data_description=None):
        """
        Checking data at workers. Returns None if everything is OK.

        Parameters
        ----------
        input_data_description: dict
            Description of the input features.

        target_data_description: dict
            Description of the targets.

        Returns
        ----------
        err: string
            Error message, if any.

        bad_workers: list
            List of workers with bad data.

        """
        err = 'Method not available for this POM'
        bad_workers = None

        try: 
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

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at check_data_at_workers: ', err)
            raise

        return err, bad_workers


    def deep_learning_transform_workers(self, data_description):
        """
        Convert images to numerical vector using Deep Learning.

        Parameters
        ----------
        data_description: dict
            Description of the input features.
            
        Returns
        -------
        model: DL model
            The DL model to apply to future data.
        
        new_input_data_description: dict
            Updated description of the input features.

        worker_errors: list of string
            List of errors while preprocessing data at workers.
        """
        model = None
        new_data_description = None
        errors = None

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
                if self.pom in [1, 2, 3]:
                    worker_errors = self.MasterCommon.send_preprocessor(model)
                elif self.pom in [4, 5, 6]:
                    worker_errors = self.MasterCommon.send_preprocess(model)
                else:
                    raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at deep_learning_transform_workers: ', err)
            raise

        return model, new_input_data_description, worker_errors


    def get_vocabulary_workers(self, data_description, init_vocab_dict=None):
        """
        Get vocabulary from all workers.

        Parameters
        ----------
        data_description: dict
            Description of the input features.

        init_vocab_dict: dict
            Initial vocabulary.

        Returns
        -------
        vocab: dict
            Dictionary containing the vocabulary for every worker.


        global_df_dict_filtered: dict
            Dictionary containing the vocabulary for every worker with every word appearing at least 10 times.
        """
        try:
            if self.pom in [1, 2, 3]:
                vocab = self.MasterCommon.get_vocabulary()
                vocab, global_df_dict_filtered = self.MasterCommon.get_df(vocab)
 
                return vocab, global_df_dict_filtered
    
            elif self.pom in [4, 5, 6]:
    
                if self.aggregation_type == 'roundrobin':
                    self.init_vocab_dict = init_vocab_dict

                self.MasterCommon.get_vocabulary()   # for both direct and roundrobin


                if self.aggregation_type == 'roundrobin':
                    print('MN get_vocabulary_workers, roundrobin pending...')
                    #import code
                    #code.interact(local=locals())

                # We obtain the tf and df
                self.MasterCommon.get_df(self.MasterCommon.vocab)   # for both direct and roundrobin

                return self.MasterCommon.vocab, self.MasterCommon.global_df_dict_filtered 

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at get_vocabulary_workers: ', err)
            raise


    def get_feat_freq_transformer(self, data_description, Max_freq, NF):
        """
        Get features freq from all workers, generate transformer and transform data at workers.

        Parameters
        ----------
        data_description: dict
            Description of the input features.

        Max_freq: float
            Maximal allowed frequency to select a word.

        NF: int
            Number of features to retain.

        Returns
        -------
        feature_extractor: object
            Feature extractor model.

        new_input_data_description: dict
            Updated description of the input features.
        """
        try: 
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

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at get_feat_freq_transformer: ', err)
            raise


    def record_linkage_transform_workers(self, linkage_type='full'):
        """
        Transform data at workers such that features are aligned.

        Parameters
        ----------
        linkage_type: string
            Choose the type of linkage: full/join.

        Returns
        -------
        input_data_description_dict: dict
            New dictionary describing the input data.

        target_data_description_dict: dict
            New dictionary describing the output data.
        """
        try:
            if self.pom in [1, 2, 3, 4, 5, 6]:
                hashids = self.MasterCommon.get_hashids(linkage_type)   # for both direct and roundrobin

                # record-linkage transform data at workers
                input_data_description_dict, target_data_description_dict = self.MasterCommon.linkage_hashids_transform_workers(hashids, linkage_type)
                return input_data_description_dict, target_data_description_dict 

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at record_linkage_transform_workers: ', err)
            raise


    def mn_ask_encrypter(self):
        """
        Obtain encrypter from cryptonode, under POM 4.

        Parameters
        ----------
        None
        """
        try:
            self.MasterCommon.ask_encrypter()
            # We pass the encrypter to the MasterML model
            self.MasterMLmodel.encrypter = self.MasterCommon.encrypter
            #print('Masternode WARNING: remove this')
            #self.MasterMLmodel.decrypter = self.MasterCommon.decrypter
            return

        except Exception as err:
            self.display('MasterNode: Error at mn_ask_encrypter: ', err)
            raise


    def mn_get_encrypted_data(self, use_bias=False, classes=None):
        """
        Obtain Encrypted data from workers, under POM 4.

        Parameters
        ----------
        use_bias: boolean
            Indicates if a bias must be added.

        classes: list of string
            List of possible classes.        
        """
        try:
            self.MasterCommon.get_cryptdata(use_bias, classes)
        
            # We pass the encrypted data  to the MasterML model
            self.MasterMLmodel.X_encr_dict = self.MasterCommon.X_encr_dict
            self.MasterMLmodel.y_encr_dict = self.MasterCommon.y_encr_dict
    
            self.MasterMLmodel.workers_addresses = self.MasterCommon.workers_addresses
            self.MasterMLmodel.broadcast_addresses = self.MasterCommon.broadcast_addresses
            self.MasterMLmodel.cryptonode_address = self.MasterCommon.cryptonode_address
    
            # Masternode should not have the decrypter, just for debugging... 
            #self.MasterMLmodel.decrypter = self.MasterCommon.decrypter
            #self.MasterMLmodel.encrypter = self.MasterCommon.encrypter

            # We pass the blinding data  to the MasterML model
            self.MasterMLmodel.BX_dict = self.MasterCommon.BX_dict
            self.MasterMLmodel.By_dict = self.MasterCommon.By_dict

            # We pass the blinded data to the MasterML model
            self.MasterMLmodel.X_bl_dict = self.MasterCommon.X_bl_dict
            self.MasterMLmodel.y_bl_dict = self.MasterCommon.y_bl_dict
    
            return

        except Exception as err:
            self.display('MasterNode: Error at mn_get_encrypted_data: ', err)
            raise


    def stop_workers(self):
        """
        Stop workers and start a new training.

        Parameters
        ----------
        None
        """
        try:
            if self.pom in [4, 5, 6]:
                self.MasterCommon.stop_workers_()

        except Exception as err:
            self.display('MasterNode: Error at stop_workers: ', err)
            raise


    def terminate_workers(self, workers_addresses_terminate=None):
        """
        Terminate selected workers.

        Parameters
        ----------
        workers_addresses_terminate: list of strings
            List of addresses of workers that must be terminated. If the list is empty, all the workers will stop.
        """
        try:
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

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

            # We update the list of workers
            #self.MasterCommon.workers_addresses = self.workers_addresses
            #self.MasterMLmodel.workers_addresses = self.workers_addresses
            self.MasterMLmodel.workers_addresses = self.MasterCommon.workers_addresses

        except Exception as err:
            self.display('MasterNode: Error at terminate_workers: ', err)
            raise


    def get_data_value_apriori(self, Xval, yval, stats_list):
        """
        Obtain "A priori" Data Value estimation.

        Parameters
        ----------
        Xval: list of lists or numpy array 
            Validation data, one pattern per row.

        yval:  list of lists or numpy array 
            Validation targets, one target per row.

        stats_list: list of string
            The list of statistics that have to be computed (rxy, meanx, medianx, npatterns, stdx, skewx, kurx, perc25, perc75, staty).
        """
        try:
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

            if self.pom in [1, 2, 3]:
                data_value_dict = {}
                for index, worker in enumerate(self.MasterCommon.workers_addresses):
                    data_value_dict[worker] = dv[index]
                return data_value_dict

            elif self.pom in [4, 5, 6]:
                return dv.ravel()

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at get_data_value_apriori: ', err)
            raise


    def get_data_value_aposteriori(self, Xval, yval, baseline_auc=0):
        """
        Obtain "A posterior" Data Value estimation.

        Parameters
        ----------
        Xval: list of lists or numpy array 
            Validation data, one pattern per row.

        yval:  list of lists or numpy array 
            Validation targets, one target per row.

        baseline_auc: float
            Minimum value of AUC.

        Returns
        -------
        dv: list
            List of data value estimation values for each worker.

        best_workers: list
            List of strings with the worker addresses.
        """
        if self.pom in [2, 3]:
            self.display('ERROR: In POM %s the model is not available at MasterNode.' %str(self.pom))
            return None, None

        try:
            if self.pom in [1, 4, 5, 6]:
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

                        if self.pom in [4, 5, 6]:
                            Xval_b = self.add_bias(Xval)

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
 
                dv = dv.ravel()    
                return dv, best_workers

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('MasterNode: Error at get_data_value_aposteriori: ', err)
            raise

