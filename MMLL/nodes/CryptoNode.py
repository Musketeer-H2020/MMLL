# -*- coding: utf-8 -*-
'''
CryptoNode Class
'''

__author__ = "Angel Navia-Vázquez & Francisco González-Serrano"
__date__ = "May 2020"

import numpy as np
import sys
import pickle
sys.path.append("..")
sys.path.append("../..")
from MMLL.Common_to_all_objects import Common_to_all_objects


class CryptoNode(Common_to_all_objects):
    """
    This class represents the main process associated to the CryptoNode, and it responds to the commands sent by the master to carry out the training procedure under POMs 4
    """

    def __init__(self, pom, comms, logger, verbose=False, **kwargs):

        """
        Create a :class:`CryptoNode` instance.

        Parameters
        ----------
        pom: integer
            the selected POM

        cryptonode_address: string
            address of the crypto node

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen

        **kwargs: Arbitrary keyword arguments.

       """

        self.pom = pom                  # Selected POM
        self.comms = comms              # The comms library for this worker
        self.master_address = 'ma' 
        self.cryptonode_address = 'ca'          # The id of the cryptonode (string), default value
        
        self.logger = logger            # logger
        self.verbose = verbose          # print on screen when true

        self.process_kwargs(kwargs)

        self.terminate = False          # used to terminate the process
        self.display('CryptoNode: Loading comms')
        self.display('CryptoNode: Innitiated')
        print(self.cryptonode_address)


    def create_model_crypto(self, model_type):
        """
        Create the model object to be used for training at the worker side.
        By now only one model is available, but the list will grow and we will be able
        to choose here among a wide variety of options.

        Parameters
        ----------
        model_type: str
            Type of model to be used

        """
        self.model_type = model_type

        if self.pom == 4:
            from MMLL.models.POM4.CommonML.POM4_CommonML import POM4_CommonML_Crypto
            self.cryptoCommonML = POM4_CommonML_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose, cr=self.cr)
            self.display('CryptoNode: Created CommonML model, POM = %d' % self.pom)

            if model_type == 'LR':
                from MMLL.models.POM4.LR.LR import LR_Crypto
                self.cryptoMLmodel = LR_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose)
                self.model_type = model_type
                self.display('CryptoNode: Created %s model, POM = %d' % (model_type, self.pom))

            if model_type == 'LC':
                from MMLL.models.POM4.LC.LC import LC_Crypto
                self.cryptoMLmodel = LC_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose)
                self.model_type = model_type
                self.display('CryptoNode: Created %s model, POM = %d' % (model_type, self.pom))

            if model_type == 'Kmeans':
                from MMLL.models.POM4.Kmeans.Kmeans import Kmeans_Crypto
                self.cryptoMLmodel = Kmeans_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose)
                self.model_type = model_type
                self.display('CryptoNode: Created %s model, POM = %d' % (model_type, self.pom))

        '''        
            if model_type == 'LR':
                from MMLL.models.POM4.XC.XC import XC_Crypto
                self.cryptoMLmodel = XC_Crypto(self.master_address, self.cryptonode_address, self.model_type, self.comms, self.cr, self.logger, verbose=self.verbose)
                self.display('CryptoNode: Created %s model, POM = %d' % (self.model_type, self.pom))

        if self.pom == 4:
            # This object includes general ML tasks, common to all algorithms in POM4
            from MMLL.models.POM4.CommonML.POM4_CommonML import POM4_CommonML_Worker
            self.workerCommonML = POM4_CommonML_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
            self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))

            if model_type == 'LR':
                from MMLL.models.POM5.LR.LR import LR_Worker
                self.workerMLmodel = LR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))


            # This object includes general ML tasks, common to all algorithms in POM4
            from MMLL.models.POM4.CommonML.POM4_CommonML import POM4_CommonML_Worker
            self.workerCommonML = POM4_CommonML_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.cr, self.logger, self.verbose, self.Xtr_b, self.ytr)
            self.display('WorkerNode %s: Created CommonML model' % str(self.worker_address))
            if model_type == 'XC':
                from MMLL.models.POM4.XC.XC import XC_Worker
                self.workerMLmodel = XC_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.cr, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

            if model_type == 'LR':
                from MMLL.models.POM4.LR.LR import LR_Worker
                self.workerMLmodel = LR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.cr, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

            if model_type == 'LC':
                from MMLL.models.POM4.LC.LC import LC_Worker
                self.workerMLmodel = LC_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.cr, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

            if model_type == 'Kmeans':
                from MMLL.models.POM4.Kmeans.Kmeans import Kmeans_Worker
                self.workerMLmodel = Kmeans_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.cr, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))

            if model_type == 'KR':
                from MMLL.models.POM4.KR.KR import KR_Worker
                self.workerMLmodel = KR_Worker(self.master_address, self.worker_address, self.model_type, self.comms, self.cr, self.logger, self.verbose, self.Xtr_b, self.ytr)
                self.model_type = model_type
                self.display('WorkerNode %s: Created %s model' % (str(self.worker_address), model_type))
        '''
        
    def run(self):
        """
        Run the main execution loop at the worker
        """
        self.display('Crypto: running %s ...' % self.model_type)
        self.cryptoCommonML.run_crypto()

        '''
        if self.pom == 6 or self.pom == 1 or self.pom == 4 or self.pom == 5:
            # the worker checks both workerMLmodel and workerCommonML
            self.workerMLmodel.terminate = False
            self.workerCommonML.terminate = False
            while not (self.workerMLmodel.terminate or self.workerCommonML.terminate):  # The worker can be terminated from the MLmodel or CommonML
                # We receive one packet, it could be for Common or MLmodel
                packet, sender = self.workerCommonML.CheckNewPacket_worker()
                if packet is not None:
                    try:
                        if packet['to'] == 'CommonML':
                            self.workerCommonML.ProcessReceivedPacket_Worker(packet, sender)
                            if packet['action'] == 'send_encrypter':
                                # Passing the encrypter to workerMLmodel
                                self.workerMLmodel.encrypter = self.workerCommonML.encrypter
                                self.workerMLmodel.decrypter = self.workerCommonML.decrypter
                        if packet['to'] == 'MLmodel':
                            self.workerMLmodel.ProcessReceivedPacket_Worker(packet, sender)
                    except:
                        pass
        '''

