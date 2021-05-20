# -*- coding: utf-8 -*-
'''
WorkerNode class.
'''

__author__ = "Angel Navia-Vázquez, Marcos Fernández"
__date__ = "May 2021"


import numpy as np
import time

from MMLL.Common_to_all_objects import Common_to_all_objects



class WorkerNode(Common_to_all_objects):
    """
    This class represents the main process associated to every Worker Node, and it responds to the commands sent by the master to carry out the training procedure under all POMs.
    """

    def __init__(self, pom, comms, logger, verbose=False, **kwargs):

        """
        Create a :class:`WorkerNode` instance.

        Parameters
        ----------
        pom: integer
            The selected Privacy Operation Mode.

        comms: comms object instance
            Object providing communications.

        logger: class:`logging.Logger`
            Logging object instance.

        verbose: boolean
            Indicates if messages are print or not on screen.

        **kwargs: Arbitrary keyword arguments.
        """
        try:
            self.pom = pom                  # Selected POM
            self.comms = comms              # The comms library for this worker
            self.logger = logger            # logger
            self.verbose = verbose          # print on screen when true
            self.master_address = 'ma'       
            self.process_kwargs(kwargs)
            self.worker_address = self.comms.id
            self.cryptonode_address = 'ca'          # The id of the cryptonode (string), default value
            self.terminate = False          # used to terminate the process
            self.model_type = None          # ML model to be used, to be defined later
            self.display('WorkerNode %s: Loading comms' % str(self.worker_address))
            self.display('WorkerNode %s: Loading Data Connector' % str(self.worker_address))
            self.display('WorkerNode %s: Initiated' % str(self.worker_address))
            self.data_is_ready = False

        except Exception as err:
            self.display('WorkerNode: Error at __init__: ', err)
            raise


    def set_training_data(self, dataset_name, Xtr=None, ytr=None, input_data_description=None, target_data_description=None):
        """
        Set data to be used for training.

        Parameters
        ----------
        dataset_name: string
            The name of the dataset.

        Xtr: list of lists or numpy array 
            Training input data, one pattern per row.

        ytr: list of lists or numpy array 
            Training targets, one target per row.

        input_data_description: dict
            Description of the input features.

        target_data_description: dict
            Description of the targets.
        """
        self.dataset_name = dataset_name
        self.input_data_description = input_data_description
        self.target_data_description = target_data_description

        try:
            # We store the original data as it is received...
            self.Xtr_orig = Xtr
            self.ytr_orig = ytr
            
            sparse = False
            try:
                Xtype = Xtr.getformat()
                if Xtype == 'csr' or Xtype == 'csc':
                    sparse = True 
            except:
                pass

            # If the matrix is sparse, we skip this processing
            if not sparse:
                # We store here the data ready to be used for training
                self.Xtr_b = np.array(Xtr)            
                self.ytr = np.array(ytr)
            else:
                self.Xtr_b = Xtr            
                self.ytr = ytr

            self.NPtr  = self.Xtr_b.shape[0]
            self.NI = self.Xtr_b.shape[1]

            if self.Xtr_b.shape[0] != self.ytr.shape[0] and ytr is not None:
                self.display('ERROR: different number of patterns in Xtr and ytr (%s vs %s)' % (str(self.Xval_b.shape[0]), str(self.yval.shape[0])))
                self.Xtr_b = None
                self.ytr = None
                self.NPtr = 0
                self.display('WorkerNode: ***** Train data NOT VALID. *****')
                return
            else:
                self.display('WorkerNode got train data: %d patterns, %d features' % (self.NPtr, self.NI))

        except:
            self.display('WorkerNode: ***** Training data NOT available. *****')

            pass


    def set_validation_data(self, dataset_name, Xval=None, yval=None):
        """
        Set data to be used for validation.

        Parameters
        ----------
        dataset_name: string
            The name of the dataset.

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
                self.display('WorkerNode: ***** Validation data NOT VALID. *****')
                return
            else:
                self.display('WorkerNode got validation data: %d patterns, %d features' % (self.NPval, self.NI))
        except:
            self.display('WorkerNode: ***** Validation data NOT available. *****')
            pass


    def set_test_data(self, dataset_name, Xtst=None, ytst=None):
        """
        Set data to be used for testing.

        Parameters
        ----------
        dataset_name: string
            The name of the dataset.

        Xtst: list of lists or numpy array 
            Test data, one pattern per row.

        ytst:  list of lists or numpy array 
            Test targets, one target per row.
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
                self.display('WorkerNode: ***** Test data NOT VALID. *****')
                return
            else:
                self.display('WorkerNode got test data: %d patterns, %d features' % (self.NPtst, self.NI))
        except:
            self.display('WorkerNode: ***** Test data NOT available. *****')
            pass


    def create_model_worker(self, model_type):
        """
        Create the model object to be used for training at the Worker side.

        Parameters
        ----------
        model_type: string
            Type of model to be used.
        """
        try:
            self.model_type = model_type

            if self.pom == 1:
                if model_type == 'Kmeans':
                    from MMLL.models.POM1.Kmeans.Kmeans import Kmeans_Worker
                    self.workerMLmodel = Kmeans_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b)

                elif model_type == 'NN':
                    from MMLL.models.POM1.NeuralNetworks.neural_network import NN_Worker
                    self.workerMLmodel = NN_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                elif model_type == 'SVM':
                    from MMLL.models.POM1.SVM.SVM import SVM_Worker
                    self.workerMLmodel = SVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                elif model_type == 'FBSVM':
                    from MMLL.models.POM1.FBSVM.FBSVM import FBSVM_Worker
                    self.workerMLmodel = FBSVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                elif model_type == 'DSVM':
                    from MMLL.models.POM1.DSVM.DSVM import DSVM_Worker
                    self.workerMLmodel = DSVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

            elif self.pom == 2:
                if model_type == 'Kmeans':
                    from MMLL.models.POM2.Kmeans.Kmeans import Kmeans_Worker
                    self.workerMLmodel = Kmeans_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b)

                elif model_type == 'NN':
                    from MMLL.models.POM2.NeuralNetworks.neural_network import NN_Worker
                    self.workerMLmodel = NN_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                elif model_type == 'SVM':
                    from MMLL.models.POM2.SVM.SVM import SVM_Worker
                    self.workerMLmodel = SVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                elif model_type == 'FBSVM':
                    from MMLL.models.POM2.FBSVM.FBSVM import FBSVM_Worker
                    self.workerMLmodel = FBSVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

            elif self.pom == 3:
                if model_type == 'Kmeans':
                    from MMLL.models.POM3.Kmeans.Kmeans import Kmeans_Worker
                    self.workerMLmodel = Kmeans_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b)

                elif model_type == 'NN':
                    from MMLL.models.POM3.NeuralNetworks.neural_network import NN_Worker
                    self.workerMLmodel = NN_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                elif model_type == 'SVM':
                    from MMLL.models.POM3.SVM.SVM import SVM_Worker
                    self.workerMLmodel = SVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                elif model_type == 'FBSVM':
                    from MMLL.models.POM3.FBSVM.FBSVM import FBSVM_Worker
                    self.workerMLmodel = FBSVM_Worker(self.master_address, self.comms, self.logger,  self.verbose, self.Xtr_b, self.ytr)

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))
    
            elif self.pom == 4:
                # This object includes general ML tasks, common to all algorithms in POM4
                from MMLL.models.POM4.CommonML.POM4_CommonML import POM4_CommonML_Worker
                self.workerCommonML = POM4_CommonML_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
                self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))
                
                try:
                    self.workerCommonML.input_data_description = self.input_data_description
                    self.workerCommonML.target_data_description = self.target_data_description
                except:
                    pass

                if model_type == 'LR':
                    from MMLL.models.POM4.LR.LR import LR_Worker
                    self.workerMLmodel = LR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'LC':
                    from MMLL.models.POM4.LC.LC import LC_Worker
                    self.workerMLmodel = LC_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'MLC':
                    from MMLL.models.POM4.MLC.MLC import MLC_Worker
                    self.workerMLmodel = MLC_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'Kmeans':
                    from MMLL.models.POM4.Kmeans.Kmeans import Kmeans_Worker
                    self.workerMLmodel = Kmeans_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, cryptonode_address=self.cryptonode_address)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'KR':
                    from MMLL.models.POM4.KR.KR import KR_Worker
                    self.workerMLmodel = KR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
                    self.model_type = model_type
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'BSVM':
                    from MMLL.models.POM4.BSVM.BSVM import BSVM_Worker
                    self.workerMLmodel = BSVM_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr, cryptonode_address=self.cryptonode_address)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

            elif self.pom == 5:
                from MMLL.models.POM5.CommonML.POM5_CommonML import POM5_CommonML_Worker
                self.workerCommonML = POM5_CommonML_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))
                
                try:
                    self.workerCommonML.input_data_description = self.input_data_description
                    self.workerCommonML.target_data_description = self.target_data_description
                except:
                    pass

                if model_type == 'LR':
                    from MMLL.models.POM5.LR.LR import LR_Worker
                    self.workerMLmodel = LR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'KR':
                    from MMLL.models.POM5.KR.KR import KR_Worker
                    self.workerMLmodel = KR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'LC':
                    from MMLL.models.POM5.LC.LC import LC_Worker
                    self.workerMLmodel = LC_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))
    
                elif model_type == 'MLC':
                    from MMLL.models.POM5.MLC.MLC import MLC_Worker
                    self.workerMLmodel = MLC_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'Kmeans':
                    from MMLL.models.POM5.Kmeans.Kmeans import Kmeans_Worker
                    self.workerMLmodel = Kmeans_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'BSVM':
                    from MMLL.models.POM5.BSVM.BSVM import BSVM_Worker
                    self.workerMLmodel = BSVM_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'MBSVM':
                    from MMLL.models.POM5.MBSVM.MBSVM import MBSVM_Worker
                    self.workerMLmodel = MBSVM_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

            elif self.pom == 6:
                # This object includes general ML tasks, common to all algorithms in POM6
                from MMLL.models.POM6.CommonML.POM6_CommonML import POM6_CommonML_Worker
                self.workerCommonML = POM6_CommonML_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))
                
                try:
                    self.workerCommonML.input_data_description = self.input_data_description
                    self.workerCommonML.target_data_description = self.target_data_description
                except:
                    pass

                if model_type == 'RR':
                    from MMLL.models.POM6.RR.RR import RR_Worker
                    self.workerMLmodel = RR_Worker(self.master_address, self.worker_address,  model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'KR':
                    from MMLL.models.POM6.KR.KR import KR_Worker
                    self.workerMLmodel = KR_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'LC':
                    from MMLL.models.POM6.LC.LC import LC_Worker
                    self.workerMLmodel = LC_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'MLC':
                    from MMLL.models.POM6.MLC.MLC import MLC_Worker
                    self.workerMLmodel = MLC_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'Kmeans':
                    from MMLL.models.POM6.Kmeans.Kmeans import Kmeans_Worker
                    self.workerMLmodel = Kmeans_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'BSVM':
                    from MMLL.models.POM6.BSVM.BSVM import BSVM_Worker
                    self.workerMLmodel = BSVM_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                elif model_type == 'MBSVM':
                    from MMLL.models.POM6.MBSVM.MBSVM import MBSVM_Worker
                    self.workerMLmodel = MBSVM_Worker(self.master_address, self.worker_address, model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                    self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('WorkerNode %s: Error at create_model_worker: %s' %(str(self.worker_address), str(err)))
            raise


    def run(self):
        """
        Run the main execution loop at the worker.

        Parameters
        ----------
        None
        """
        self.display('WorkerNode_' + self.model_type + ' %s: running %s ...' % (str(self.worker_address), self.model_type))

        try:
            if self.pom in [1, 2, 3]:
                # Only for vertical partition of data 
                try:
                    self.workerMLmodel.input_data_description = self.input_data_description
                    self.workerMLmodel.target_data_description = self.target_data_description
                    self.display('WorkerNode: Received input/target_data_description')
                except:
                    pass

                self.workerMLmodel.run_worker()

            elif self.pom in [4, 5, 6]:
                # the worker checks both workerMLmodel and workerCommonML
                self.workerMLmodel.terminate = False
                self.workerCommonML.terminate = False
                self.workerMLmodel.Xtr_b = self.workerCommonML.Xtr_b
                self.workerMLmodel.ytr = self.workerCommonML.ytr

                # Only for vertical partition of data 
                try:
                    self.workerMLmodel.input_data_description = self.workerCommonML.input_data_description
                    self.workerMLmodel.target_data_description = self.workerCommonML.target_data_description
                    #self.display('WorkerNode: received input/target_data_description')
                except:
                    pass

                while not (self.workerMLmodel.terminate or self.workerCommonML.terminate):  # The worker can be terminated from the MLmodel or CommonML
                    # We receive one packet, it could be for Common or MLmodel
                    #print("I'm alive!")
                    packet, sender = self.workerCommonML.CheckNewPacket_worker()
                    if packet is not None:
                        try:
                            if packet['to'] == 'CommonML':
                                #print('worker run CommonML', packet['action'])
                                self.workerCommonML.ProcessReceivedPacket_Worker(packet, sender)
                                if packet['action'] == 'send_encrypter':
                                    # Passing the encrypter to workerMLmodel
                                    self.workerMLmodel.encrypter = self.workerCommonML.encrypter
                                if packet['action'] == 'do_local_prep':
                                    # Passing the transformed training data to workerMLmodel
                                    self.workerMLmodel.Xtr_b = self.workerCommonML.Xtr_b
                                    self.workerMLmodel.ytr = self.workerCommonML.ytr
                                    time.sleep(0.1)                               
                            if packet['to'] == 'MLmodel':
                                #print('worker run MLmodel', packet['action'])
                                self.workerMLmodel.ProcessReceivedPacket_Worker(packet, sender)
                        except Exception as err:
                            print('ERROR at Workernode', err, str(err), str(type(err)))
                            raise
                            pass

            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

        except Exception as err:
            self.display('WorkerNode %s: Error at run: %s' %(str(self.worker_address), str(err)))
            raise


    def get_preprocessors(self):
        """
        Returns the normalizer parameters and transform the training data in the workers.

        Parameters
        ----------
        None

        Returns
        -------
        preprocessors: object
            Normalizer object.
        """
        try:
            preprocessors = None
            if self.pom in [1, 2, 3]:
                if len(self.workerMLmodel.preprocessors) > 0:
                    preprocessors = self.workerMLmodel.preprocessors
                else:
                    self.display('WorkerNode: No preprocessors available at the worker')
            elif self.pom in [4, 5, 6]:
                self.display('WorkerNode: No preprocessors available at the worker for POM %s' %str(self.pom))
            else:
                raise ValueError('POM %s not available in MMLL' %str(self.pom))

            return preprocessors

        except Exception as err:
            self.display('WorkerNode %s: Error at run: %s' %(str(self.worker_address), str(err)))
            raise


    def get_model(self):
        """
        Returns the ML model as an object, if it is trained, it returns None otherwise.

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
            model_is_trained = self.workerMLmodel.is_trained
            if not model_is_trained:
                self.display('WorkerNode: Error - Model not trained yet')
                return model
            else:
                return self.workerMLmodel.model
        except:
            self.display('In this POM, the model is not available at WorkerNode.')
            return model


