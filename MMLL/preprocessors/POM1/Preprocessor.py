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

from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master, POM1_CommonML_Worker
import pycloudmessenger.ffl.abstractions as fflapi



class Preprocessor_Master(POM1_CommonML_Master):
    """
    This class implements Kmeans, run at Master node. It inherits from POM1_CommonML_Master.
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
        self.normalized_mins = []
        self.normalized_maxs = []
        
        
        
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
                self.state_dict['CN'] = 'SEND_MEANS'

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
    
        elif self.normalization_type=='global_min_max':

            if self.state_dict['CN'] == 'START_NORMALIZATION':
                self.state_dict['CN'] = 'SEND_MIN_MAX'

            if self.checkAllStates('COMPUTE_MIN_MAX', self.state_dict):
                self.counter = 0
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'OBTAIN_GLOBAL_MIN_MAX'

    
    
    def TakeAction_Master(self):
        """
        Takes actions according to the state
        """
        # Asking the workers to send initialize means
        if self.state_dict['CN'] == 'SEND_MEANS':
            self.list_means=list()
            self.list_counts=list()
            action = 'SEND_MEANS'
            to = 'Preprocessing'
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
            to = 'Preprocessing'
            data = {'global_means': self.global_means}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_STDS'


        # Asking the workers to send initialize min and max values
        if self.state_dict['CN'] == 'SEND_MIN_MAX':
            self.list_mins=list()
            self.list_maxs=list()
            action = 'SEND_MIN_MAX'
            to = 'Preprocessing'
            data = {'data_description': self.data_description}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_MIN_MAX'


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
            self.global_stds=np.sqrt(self.global_stds)
            self.state_dict['CN'] = 'SEND_PREPROCESSOR'

        if self.state_dict['CN'] == 'OBTAIN_GLOBAL_MIN_MAX':
            for i in range(len(self.list_mins)):
                if i == 0:
                    self.global_mins = np.array(self.list_mins[i])
                    self.global_maxs = np.array(self.list_maxs[i])
                else:
                    self.global_mins = np.minimum(self.global_mins, np.array(self.list_mins[i]))
                    self.global_maxs = np.maximum(self.global_maxs, np.array(self.list_maxs[i]))
            self.state_dict['CN'] = 'SEND_PREPROCESSOR'

        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_PREPROCESSOR':
            action = 'SEND_PREPROCESSOR'
            to = 'Preprocessing'

            if self.normalization_type == 'global_mean_std':
                self.model.mean = np.reshape(self.global_means,(-1,len(self.global_means)))
                self.model.std = np.reshape(self.global_stds,(-1,len(self.global_stds)))

                """
                self.means = self.global_means
                self.stds = self.global_stds
                data = {'normalization_type':self.normalization_type,'global_means': self.global_means,'global_stds': self.global_stds}"""

            elif self.normalization_type == 'global_min_max':
                self.model.min = np.reshape(self.global_mins,(-1,len(self.global_mins)))
                self.model.max = np.reshape(self.global_maxs,(-1,len(self.global_maxs)))

                data = {'normalization_type':self.normalization_type, 'global_mins': self.global_mins,'global_maxs': self.global_maxs}

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


        if self.state_dict['CN'] == 'WAIT_MIN_MAX':
            if packet['action'] == 'COMPUTE_MIN_MAX':
                self.list_mins.append(np.array(packet['data']['mins']))
                self.list_maxs.append(np.array(packet['data']['maxs']))
                self.state_dict[sender] = packet['action']


        return
    

    
    

#===============================================================
#                 Worker   
#===============================================================

class Preprocessor_Worker(POM1_CommonML_Worker):
    '''
    Class implementing Kmeans, run at Worker

    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None):
        """
        Create a :class:`Kmeans_Worker` instance.

        Parameters
        ----------
        master_address: string
            Identifier of the master instance

        worker_address: string
            Identifier for the worker

        platform: string
            Identifier of the comms platform to use:
                * 'pycloudmessenger': for communication through pycloudmessenger (local and cloud versions)
                * 'local_flask': for communication through local flask

        comms: comms object instance
            Object providing communication functionalities

        logger: class:`mylogging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen

        Xtr_b: ndarray
            2-D numpy array containing the input training patterns
        """
        #super().__init__(comms, logger, verbose)
        self.name = 'Preprocessor_Worker'             # Name
        #self.master_address = master_address
        #self.worker_address = worker_address    # The id of this Worker
        self.master_address = master_address
        self.comms = comms                      # The comms library
        self.logger = logger                    # logger
        self.verbose = verbose                  # print on screen when true
        self.Xtr_b = Xtr_b

        self.worker_address = comms.id
        self.platform = comms.name
        self.num_features = Xtr_b.shape[1]
        self.is_trained = False

        self.normalization_type = None      
        self.means = None      
        self.stds = None      
        self.mins = None      
        self.mmaxs = None 
        
        

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Take an action after receiving a packet

        Parameters
        ----------
            packet: packet object 
                packet received (usually a dict with various content)

        """
        self.terminate = False

        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' %self.worker_address)
            self.terminate = True
            
        
        if packet['action'] == 'SEND_MEANS':
            self.display(self.name + ' %s: Obtaining means' %self.worker_address)
            self.data_description = np.array(packet['data']['data_description'])

            means = np.mean(self.Xtr_b,axis=0)

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

                              
            var = np.mean(X_without_mean*X_without_mean,axis=0)
            counts = self.Xtr_b.shape[0]

            action = 'COMPUTE_STDS'
            data = {'var': var, 'counts':counts}
            packet = {'action': action, 'data': data}
            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

    
        if packet['action'] == 'SEND_MIN_MAX':
            self.display(self.name + ' %s: Obtaining means' %self.worker_address)
            self.data_description = np.array(packet['data']['data_description'])
            mins = np.min(self.Xtr_b,axis=0)
            maxs = np.max(self.Xtr_b,axis=0)

            action = 'COMPUTE_MIN_MAX'
            data = {'mins': mins, 'maxs':maxs}
            packet = {'action': action, 'data': data}

            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


            
        if packet['action'] == 'SEND_PREPROCESSOR':
            self.display(self.name + ' %s: Receiving preprocessor' %self.worker_address)
            self.normalization_type = packet['data']['normalization_type']            

            if self.normalization_type=='global_mean_std':
                self.means = packet['data']['global_means']
                self.stds = packet['data']['global_stds']
            else:
                self.mins = packet['data']['global_mins']
                self.maxs = packet['data']['global_maxs']


            self.is_trained = True
            self.terminate = True
            self.display(self.name + ' %s: Final preprocessor stored' %self.worker_address)

        return
