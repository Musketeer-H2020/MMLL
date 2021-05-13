# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM1

'''

__author__ = "Marcos Fernández Díaz"
__date__ = "December 2020"

import sys
import numpy as np

from MMLL.models.Common_to_POMs_123 import Common_to_POMs_123_Master, Common_to_POMs_123_Worker



class POM1_CommonML_Master(Common_to_POMs_123_Master):
    """
    This class implements the Common ML operations, run at Master node. It inherits from Common_to_POMs_123_Master.
    """

    def __init__(self, comms, logger, verbose=False):
        """
        Create a :class:`POM1_CommonML_Master` instance.

        Parameters
        ----------
        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen

        """
        self.comms = comms
        self.logger = logger
        self.verbose = verbose

        self.name = 'POM1_CommonML_Master'              # Name
        self.platform = comms.name                      # String with the platform to use (either 'pycloudmessenger' or 'local_flask')
        self.all_workers_addresses = comms.workers_ids  # All addresses of the workers
        self.workers_addresses = comms.workers_ids      # Addresses of the workers used to send messages to (can be adapted dynamically during the execution)
        self.Nworkers = len(self.workers_addresses)     # Nworkers
        self.reset()                                    # Reset variables
        self.state_dict = {}                            # Dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})



    def reset(self):
        """
        Create/reset some empty variables needed by the Master Node
        """
        self.display(self.name + ': Resetting local data')
        self.list_centroids = []
        self.list_counts = []
        self.list_dists = []
        self.list_public_keys = []
        self.list_gradients = []
        self.list_weights = []
        self.list_costs = []

       
        

#===============================================================
#                 Worker   
#===============================================================

class POM1_CommonML_Worker(Common_to_POMs_123_Worker):
    '''
    Class implementing the POM1 Common operations, run at Worker

    '''

    def __init__(self, master_address, comms, logger, verbose=False):
        """
        Create a :class:`POM1_CommonML_Worker` instance.

        Parameters
        ----------
        master_address: string
            Identifier of the master instance

        comms: comms object instance
            Object providing communication functionalities

        logger: class:`mylogging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen
        """
        self.master_address = master_address
        self.comms = comms
        self.logger = logger
        self.verbose = verbose 

        self.name = 'POM1_CommonML_Worker'            # Name
        self.worker_address = comms.id                # Id identifying the current worker
        self.platform = comms.name                    # String with the platform to use (either 'pycloudmessenger' or 'local_flask')
        self.preprocessors = []                       # List to store all the preprocessors to be applied in sequential order to new data



    def ProcessPreprocessingPacket(self, packet):
        """
        Take an action after receiving a packet for the preprocessing

        Parameters
        ----------
        packet: Dictionary
            Packet received
        """        
        if packet['action'] == 'SEND_MEANS':
            self.display(self.name + ' %s: Obtaining means' %self.worker_address)
            self.data_description = np.array(packet['data']['data_description'])
            means = np.mean(self.Xtr_b, axis=0)
            counts = self.Xtr_b.shape[0]
            action = 'COMPUTE_MEANS'
            data = {'means': means, 'counts':counts}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))            

        if packet['action'] == 'SEND_STDS':
            self.display(self.name + ' %s: Obtaining stds' %self.worker_address)
            self.global_means = np.array(packet['data']['global_means'])
            X_without_mean = self.Xtr_b-self.global_means                              
            var = np.mean(X_without_mean*X_without_mean, axis=0)
            counts = self.Xtr_b.shape[0]

            action = 'COMPUTE_STDS'
            data = {'var': var, 'counts':counts}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
    
        if packet['action'] == 'SEND_MIN_MAX':
            self.display(self.name + ' %s: Obtaining means' %self.worker_address)
            self.data_description = np.array(packet['data']['data_description'])
            mins = np.min(self.Xtr_b, axis=0)
            maxs = np.max(self.Xtr_b, axis=0)

            action = 'COMPUTE_MIN_MAX'
            data = {'mins': mins, 'maxs':maxs}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'SEND_PREPROCESSOR':
            self.display(self.name + ' %s: Receiving preprocessor' %self.worker_address)
            # Retrieve the preprocessing object
            prep_model = packet['data']['prep_model']

            # Apply the received object to Xtr_b and store back the result
            Xtr = np.copy(self.Xtr_b)
            X_prep = prep_model.transform(Xtr)
            self.Xtr_b = np.copy(X_prep)
            self.display(self.name + ' %s: Training set transformed using preprocessor' %self.worker_address)

            clip_max = np.copy(self.pgd_params['clip_max'])
            clip_max = np.expand_dims(clip_max, axis=0)
            clip_max_prep = prep_model.transform(clip_max)
            clip_max_prep = np.squeeze(clip_max_prep)
            self.pgd_params['clip_max'] = np.copy(clip_max_prep)

            clip_min = np.copy(self.pgd_params['clip_min'])
            clip_min = np.expand_dims(clip_min, axis=0)
            clip_min_prep = prep_model.transform(clip_min)
            clip_min_prep = np.squeeze(clip_min_prep)
            self.pgd_params['clip_min'] = np.copy(clip_min_prep)

            # Store the preprocessing object
            self.preprocessors.append(prep_model)
            self.display(self.name + ' %s: Final preprocessor stored' %self.worker_address)

            action = 'ACK_SEND_PREPROCESSOR'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

