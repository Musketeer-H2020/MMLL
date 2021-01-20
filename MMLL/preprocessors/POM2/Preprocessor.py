# -*- coding: utf-8 -*-
'''
Preprocessor 

'''

__author__ = "Roberto Diaz"
__date__ = "April 2020"

# Code to ensure reproducibility in the results
from numpy.random import seed
seed(1)

import numpy as np
import sys

from MMLL.models.POM2.CommonML.POM2_CommonML import POM2_CommonML_Master
import pycloudmessenger.ffl.abstractions as fflapi



class Preprocessor_Master(POM2_CommonML_Master):
    """
    This class implements Kmeans, run at Master node. It inherits from POM2_CommonML_Master.
    """

    def __init__(self, comms, logger, data_description, verbose=False, normalization_type=None, model=None):
        """
        Create a :class:`Kmeans_Master` instance.

        Parameters
        ----------
        Nworkers: integer
            number of workers innitially associated to the task

        end_users_addresses: list of strings
            list of the addresses of the workers

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen
        """
        #super().__init__(comms, logger, verbose)
        self.name = 'Preprocessor_Master'           # Name
        #self.master_address = master_address 
        #self.workers_addresses = workers_addresses
        self.comms = comms                          # comms lib
        self.logger = logger                        # logger
        self.verbose = verbose                      # print on screen when true
        self.normalization_type = normalization_type     
        self.data_description = data_description
        self.model = model
        self.means = None      
        self.stds = None      
        self.mins = None      
        self.mmaxs = None      

        self.platform = comms.name                  # Type of comms to use: either 'pycloudmessenger' or 'localflask'
        self.workers_addresses = comms.workers_ids  # Addresses of the workers
        self.Nworkers = len(self.workers_addresses) # Nworkers
        self.mean_dist = np.inf        
        self.num_features = None
        self.iter = 0
        self.counter = 0
        self.is_trained = False
        self.reset()
        self.state_dict = {}                        # dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})
        self.list_public_keys = []
           
 

    def reset(self):
        """
        Create some empty variables needed by the Master Node

        Parameters
        ----------
        NI: integer
            Number of input features
        """
        self.display(self.name + ': Resetting local data')
        self.normalized_means = []
        self.normalized_stds = []        
        
        
    def normalize_Master(self):
        """
        This is the main training loop, it runs the following actions until 
        the stop condition is met:
            - Update the execution state
            - Process the received packets
            - Perform actions according to the state

        Parameters
        ----------
        None
        """       
        self.state_dict.update({'CN': 'START_NORMALIZATION'})
        self.display(self.name + ': Starting normalization')
        self.display(self.name + ': Normalization type %s' %self.normalization_type)

        while self.state_dict['CN'] != 'END':
            self.Update_State_Master()
            self.TakeAction_Master()
            self.CheckNewPacket_Master()
            
        self.display(self.name + ': Normalization is done')
    
    
    
    def Update_State_Master(self):
        '''
        We update the state of execution.
        We control from here the data flow of the training process
        ** By now there is only one implemented option: direct transmission **

        This code needs some improvement...
        '''

        if self.normalization_type=='global_mean_std':

            if self.state_dict['CN'] == 'START_NORMALIZATION':
                self.state_dict['CN'] = 'SEND_PUBLIC_KEY'
            
            if self.checkAllStates('SEND_PUBLIC_KEY', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'CHECK_PUBLIC_KEYS'            

            if self.checkAllStates('COMPUTE_MEANS', self.state_dict):
                self.counter = 0
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'AVERAGE_GLOBAL_MEANS'

            if self.checkAllStates('COMPUTE_STDS', self.state_dict):
                self.counter = 0
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'AVERAGE_GLOBAL_STDS'
    
    
    
    def TakeAction_Master(self):
        """
        Takes actions according to the state
        """
        to = 'Preprocessing'

        # Ask workers to send public key
        if self.state_dict['CN'] == 'SEND_PUBLIC_KEY':
            action = 'SEND_PUBLIC_KEY'
            packet = {'to': to, 'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_PUBLIC_KEYS'
        
        # Checking public keys received from workers
        if self.state_dict['CN'] == 'CHECK_PUBLIC_KEYS':
            if not all(x==self.list_public_keys[0] for x in self.list_public_keys):
                self.display(self.name + ': Workers have different keys, terminating POM2 execution')
                self.state_dict['CN'] = 'END'
                return
            self.public_key = self.list_public_keys[0]
            self.display(self.name + ': Storing public key from workers')
            self.state_dict['CN'] = 'SEND_MEANS'

        # Asking the workers to send initialize means
        if self.state_dict['CN'] == 'SEND_MEANS':
            self.list_means=list()
            self.list_counts=list()
            action = 'SEND_MEANS'
            data = {'data_description': self.data_description}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_MEANS'

        # Asking the workers to send initialize stds
        if self.state_dict['CN'] == 'SEND_STDS':
            self.list_stds=list()
            self.list_counts=list()
            action = 'SEND_STDS'
            data = {'global_means': self.global_means}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_STDS'


        # Compute average of centroids and mean distance
        if self.state_dict['CN'] == 'AVERAGE_GLOBAL_MEANS':
            total_data = np.sum(self.list_counts)
            for i in range(len(self.list_counts)):
                if i == 0:
                    self.global_means = np.float(self.list_counts[i])*self.list_means[i]/total_data
                else:
                    self.global_means += np.float(self.list_counts[i])*self.list_means[i]/total_data
            self.state_dict['CN'] = 'SEND_STDS'


        # Compute average of centroids and mean distance
        if self.state_dict['CN'] == 'AVERAGE_GLOBAL_STDS':
            total_data = np.sum(self.list_counts)
            for i in range(len(self.list_counts)):
                if i == 0:
                    self.global_stds = np.float(self.list_counts[i])*self.list_stds[i]/total_data
                else:
                    self.global_stds += np.float(self.list_counts[i])*self.list_stds[i]/total_data
            #self.global_stds = np.sqrt(self.global_stds)
            self.state_dict['CN'] = 'SEND_PREPROCESSOR'


        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_PREPROCESSOR':
            action = 'SEND_PREPROCESSOR'

            if self.normalization_type == 'global_mean_std':
                self.model.mean = np.reshape(self.global_means,(-1,len(self.global_means)))
                self.model.std = np.reshape(self.global_stds,(-1,len(self.global_stds)))

                data = {'prep_model': self.model}
                packet = {'to': to, 'action': action, 'data': data}
                self.comms.broadcast(packet, self.workers_addresses)
                self.display(self.name + ': Sent %s to all workers' %action)
                self.is_trained = True
                self.state_dict['CN'] = 'wait'

            

    def ProcessReceivedPacket_Master(self, packet, sender):
        """
        Process the received packet at Master and take some actions, possibly changing the state

        Parameters
        ----------
            packet: packet object
                packet received (usually a dict with various content)

            sender: string<s
                id of the sender
        """
        if packet['action'][0:3] == 'ACK':
            self.state_dict[sender] = packet['action']
            if self.checkAllStates('ACK_SEND_PREPROCESSOR', self.state_dict): # Included here to avoid calling CheckNewPacket_Master after sending the final model (this call could imply significant delay if timeout is set to a high value)
                self.state_dict['CN'] = 'END'

        if self.state_dict['CN'] == 'WAIT_PUBLIC_KEYS':
            if packet['action'] == 'SEND_PUBLIC_KEY':
                self.list_public_keys.append(packet['data']['public_key'])
                self.state_dict[sender] = packet['action']


        if self.state_dict['CN'] == 'WAIT_MEANS':
            if packet['action'] == 'COMPUTE_MEANS':
                self.list_means.append(np.array(packet['data']['means']))
                self.list_counts.append(np.array(packet['data']['counts']))
                self.state_dict[sender] = packet['action']

        if self.state_dict['CN'] == 'WAIT_STDS':
            if packet['action'] == 'COMPUTE_STDS':
                self.list_stds.append(np.array(packet['data']['var']))
                self.list_counts.append(np.array(packet['data']['counts']))
                self.state_dict[sender] = packet['action']

        return
    

    
    

