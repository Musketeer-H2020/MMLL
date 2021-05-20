# -*- coding: utf-8 -*-
'''
CryptoNode Class
'''

__author__ = "Angel Navia-Vázquez & Francisco González-Serrano"
__date__ = "May 2021"

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
        try:
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

                elif model_type == 'LC':
                    from MMLL.models.POM4.LC.LC import LC_Crypto
                    self.cryptoMLmodel = LC_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose)
                    self.model_type = model_type
                    self.display('CryptoNode: Created %s model, POM = %d' % (model_type, self.pom))

                elif model_type == 'Kmeans':
                    from MMLL.models.POM4.Kmeans.Kmeans import Kmeans_Crypto
                    self.cryptoMLmodel = Kmeans_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose)
                    self.model_type = model_type
                    self.display('CryptoNode: Created %s model, POM = %d' % (model_type, self.pom))

                elif model_type == 'BSVM':
                    from MMLL.models.POM4.BSVM.BSVM import BSVM_Crypto
                    self.cryptoMLmodel = BSVM_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose)
                    self.model_type = model_type
                    self.display('CryptoNode: Created %s model, POM = %d' % (model_type, self.pom))
                
                elif model_type == 'KR':
                    from MMLL.models.POM4.KR.KR import KR_Crypto
                    self.cryptoMLmodel = KR_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose)
                    self.model_type = model_type
                    self.display('CryptoNode: Created %s model, POM = %d' % (model_type, self.pom))

                elif model_type == 'MLC':
                    from MMLL.models.POM4.MLC.MLC import MLC_Crypto
                    self.cryptoMLmodel = MLC_Crypto(self.cryptonode_address, self.master_address, self.model_type, self.comms, self.logger, self.verbose)
                    self.model_type = model_type
                    self.display('CryptoNode: Created %s model, POM = %d' % (model_type, self.pom))

                else:
                    raise ValueError('Model %s not available in POM %s' % (str(model_type), str(self.pom)))
        
        except Exception as err:
            self.display('CryptoNode: Error at create_model_crypto: %s' %(str(err)))
            raise


    def run(self):
        """
        Run the main execution loop at the worker
        """
        self.display('Crypto: running %s ...' % self.model_type)
        self.cryptoCommonML.run_crypto()

