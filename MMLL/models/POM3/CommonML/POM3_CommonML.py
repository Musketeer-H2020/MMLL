# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM3.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "January 2021"


import multiprocessing
import numpy as np
import sys
from phe import paillier
from phe.util import getprimeover, powmod, invert

from MMLL.models.Common_to_POMs_123 import Common_to_POMs_123_Master, Common_to_POMs_123_Worker
from MMLL.models.POM3.CommonML.parallelization_deep import (parallelization_encryption, parallelization_encryption_rvalues, 
                                                               parallelization_encryption_int, parallelization_decryption, 
                                                               transform_encrypted_domain)



class POM3_CommonML_Master(Common_to_POMs_123_Master):
    """
    This class implements the Common ML operations, run at Master node. It inherits from Common_to_POMs_123_Master.
    """

    def __init__(self, comms, logger, verbose=False):
        """
        Create a :class:`POM3_CommonML_Master` instance.

        Parameters
        ----------
        comms: :class:`Comms_master`
            Object providing communications functionalities.

        logger: :class:`mylogging.Logger`
            Logging object instance.

        verbose: boolean
            Indicates whether to print messages on screen nor not.
        """
        self.comms = comms
        self.logger = logger
        self.verbose = verbose

        self.name = 'POM3_CommonML_Master'               # Name
        self.all_workers_addresses = comms.workers_ids   # All addresses of the workers
        self.workers_addresses = comms.workers_ids       # Addresses of the workers
        self.Nworkers = len(self.workers_addresses)      # Number of workers
        self.platform = comms.name                       # Type of comms to use (either 'pycloudmessenger' or 'localflask')
        self.reset()                                     # Reset variables
        self.precision = 1e-10                           # Precision for encrypting values
        self.num_cores = multiprocessing.cpu_count()     # For parallel processing using all the cores of the machine
        self.public_keys = {}                            # Dictionary to store public keys from all workers
        self.encrypted_Xi = {}                           # Dictionary to store encrypted Xi from all workers
        self.state_dict = {}                             # Dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})

        #self.Init_Environment()                          # Send initialization messages common to all algorithms



    def reset(self):
        """
        Create/reset some variables needed by the Master Node.

        Parameters
        ----------
        None
        """
        self.display(self.name + ': Resetting initialization data')
        self.list_num_features = []
        
        
        
    def encrypt_list(self, unencrypted_list, public_key):
        """
        Function to encrypt a list of arrays.

        Parameters
        ----------
        unencrypted_list: list of arrays
            List to encrypt.

        Returns
        ----------
        encrypted_list: list of arrays 
            Encrypted list.
        """     
        self.display(self.name + ': Encrypting data...')
        encrypted_list = list()
        pool = multiprocessing.Pool(processes=self.num_cores)
        for layer in unencrypted_list:
            layer_list = layer.ravel().tolist() # Convert array to flattened list
            iteration = [(public_key, layer_list, self.precision, x) for x in range(len(layer_list))]

            encrypted_layer_list = pool.starmap(parallelization_encryption, iteration)
            encrypted_layer_array = np.asarray(encrypted_layer_list).reshape(layer.shape)
            encrypted_list.append(encrypted_layer_array)
        pool.close() 
        return encrypted_list
    
    
    
    def transform_encrypted_domain_workers(self, encrypted_data, worker_origin, worker_destination, verbose=True):
        """
        Function to transform encrypted data from domain of worker_origin to worker_destination.

        Parameters
        ----------
        encrypted_data: list of arrays 
            Encrypted data using public key of worker_origin.
        worker_origin: int 
            Index of the worker who encrypted the data.
        worker_destination: int 
            Index of the worker to transform the encrypted data to.

        Returns
        ----------
        transformed_data: list of arrays 
            List of encrypted data in the domain of worker_destination.
        """
        if verbose:
            self.display(self.name + ': Transforming encrypted data from worker %s to worker %s...' %(worker_origin, worker_destination))
        start_index = 0
        transformed_data = []        
        pool = multiprocessing.Pool(processes=self.num_cores)
        for encrypted_layer in encrypted_data:
            transformed_layer = []
            list_layer = encrypted_layer.ravel().tolist() # Convert to flattened 
            slice_encrypted_Xi = self.encrypted_Xi[worker_origin][start_index:start_index+len(list_layer)] # Encrypted Xis for the current layer
            slice_encrypted_Xi_dest = self.encrypted_Xi[worker_destination][start_index:start_index+len(list_layer)] # E[Xis] for destination
            start_index = start_index + len(list_layer)
            iteration = [(self.public_keys[worker_origin], self.public_keys[worker_destination], list_layer, self.precision, 
                          slice_encrypted_Xi, slice_encrypted_Xi_dest, x) for x in range(len(list_layer))]
            
            transformed_layer_list = pool.starmap(transform_encrypted_domain, iteration)
            array_transformed_layer = np.asarray(transformed_layer_list).reshape(encrypted_layer.shape)
            transformed_data.append(array_transformed_layer)            
        pool.close()
        return transformed_data



    def send_worker_and_wait_receive(self, packet, worker):
        """
        Send a message to an specific worker and wait for the reply.

        Parameters
        ----------
        packet: dictionary
            Message to be sent to the worker.
        worker: string
            Address of the worker to communicate with.

        Returns
        ----------
        packet: dictionary
            Received packet from worker.
        """
        self.comms.send(packet, worker)
        self.display(self.name + ': Sent %s to worker %s' %(packet['action'], worker))

        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            while packet == None:
                try:
                    packet = self.comms.receive_poms_123(timeout=0.1) # Set a high value for timeout for ensuring that the reply is received within that time
                    try:  # For the pycloudmessenger cloud
                        sender = packet.notification['participant']
                    except Exception: # For the pycloudmessenger local
                        self.counter = (self.counter + 1) % self.Nworkers
                        sender = self.workers_addresses[self.counter]

                    if worker != sender:
                        raise Exception('%s: Expecting receive packet from worker %s but received it from worker %s' %(self.name, worker, sender))
               
                    packet = packet.content

                except KeyboardInterrupt:
                    self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                    sys.exit()
                except Exception as err:
                    if 'pycloudmessenger.ffl.fflapi.TimedOutException' in str(type(err)):
                        pass
                    else:
                        self.display(self.name + ': Error %s' %err)
                        raise

        else: # Local flask
            packet = None
            sender = None
            while packet == None:
                try:
                    packet = self.comms.receive(worker, timeout=0.1) # Indicate the worker address
                except KeyboardInterrupt:
                    self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                    sys.exit()
                except Exception as err:
                    if str(err).startswith('Timeout when receiving data'): # TimedOutException
                        pass
                    else:
                        self.display(self.name + ': Error %s' %err)
                        raise

        self.display(self.name + ': Received %s from worker %s' %(packet['action'], worker))
        return packet


    """
    def normalize_Master(self, preprocessor):
        ""
        This is the main training loop, it runs the following actions until 
        the stop condition is met:
            - Update the execution state
            - Process the received packets
            - Perform actions according to the state

        Parameters
        ----------
        None
        ""   
        self.state_dict.update({'CN': 'START_NORMALIZATION'})

        self.display(self.name + ': Initialization ready, starting sequential communications')
        global_means = np.zeros(self.num_features)
        encrypted_means = np.asarray(self.encrypt_list([global_means], self.public_keys[self.workers_addresses[0]]))
        total_counts = 0
        for index_worker, worker in enumerate(self.workers_addresses): 

            # Get updated centroids from each worker
            action = 'SEND_MEANS'
            data = {'means': encrypted_means, 'counts': total_counts}
            packet = {'to':'Preprocessing', 'action': action, 'data': data}
                                
            # Send message to specific worker and wait until receiving reply
            packet = self.send_worker_and_wait_receive(packet, worker)                    
            encrypted_means = packet['data']['means']
            total_counts = packet['data']['counts']
            # Transform encrypted centroids to the encrypted domain of the next worker
            encrypted_means = self.transform_encrypted_domain_workers(encrypted_means, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])

        global_variances = np.zeros(self.num_features)
        encrypted_variances = np.asarray(self.encrypt_list([global_variances], self.public_keys[self.workers_addresses[0]]))
        total_counts = 0
        for index_worker, worker in enumerate(self.workers_addresses): 

            # Get updated centroids from each worker
            action = 'SEND_VARIANCES'
            data = {'variances': encrypted_variances, 'means': encrypted_means, 'counts': total_counts}
            packet = {'to': 'Preprocessing', 'action': action, 'data': data}
                                
            # Send message to specific worker and wait until receiving reply
            packet = self.send_worker_and_wait_receive(packet, worker)                    
            encrypted_variances = packet['data']['variances']
            encrypted_means = packet['data']['means']
            total_counts = packet['data']['counts']
            # Transform encrypted centroids to the encrypted domain of the next worker
            encrypted_variances = self.transform_encrypted_domain_workers(encrypted_variances, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])
            encrypted_means = self.transform_encrypted_domain_workers(encrypted_means, worker, self.workers_addresses[(index_worker+1)%self.Nworkers], verbose=False)

        
        # Send final model to workers
        action = 'SEND_PREPROCESSOR'

        if preprocessor.method == 'global_mean_std':

            action = 'SEND_PREPROCESSOR'
            for index_worker, worker in enumerate(self.workers_addresses):
                data = {'model': preprocessor, 'normalization_type': preprocessor.method, 'global_means': encrypted_means, 'global_variances': encrypted_variances}
                packet = {'to' : 'Preprocessing', 'action': action, 'data': data}            
            
                # Send message to specific worker and wait until receiving reply
                packet = self.send_worker_and_wait_receive(packet, worker)
                encrypted_variances = packet['data']['global_variances']
                encrypted_means = packet['data']['global_means']
                encrypted_means = self.transform_encrypted_domain_workers(encrypted_means, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])
                encrypted_variances = self.transform_encrypted_domain_workers(encrypted_variances, worker, self.workers_addresses[(index_worker+1)%self.Nworkers], verbose=False)
         
            self.display(self.name + ': Normalization is done')
    """



    def Init_Environment(self):
        """
        This is the main initialization loop, it runs the following actions until 
        the stop condition is met:
            - Update the execution state.
            - Perform actions according to the state.
            - Process the received packets.

        Parameters
        ----------
        None
        """   
        self.state_dict.update({'CN': 'START_INITIALIZATION'})
        self.display(self.name + ': Starting initialization')

        while self.state_dict['CN'] != 'INITIALIZATION_READY':

            self.Update_State_Common_Master()
            self.TakeAction_Common_Master()
            self.CheckNewPacket_Master()
            
        # Now communications should work sequentially (not sending a message to next worker until the actual one replied)
        self.display(self.name + ': Initialization ready')



    """
    def Preprocessing(self, model):
        ""
        This is the main preprocessing loop, it runs the following actions until 
        the stop condition is met:
            - Update the execution state
            - Perform actions according to the state
            - Process the received packets

        Parameters
        ----------
        None
        ""   
        self.state_dict.update({'CN': 'START_PREPROCESSING'})
        self.display(self.name + ': Starting preprocessing')

        while self.state_dict['CN'] != 'PREPROCESSING_READY':
            self.Update_State_Common_Master()
            self.TakeAction_Common_Master()
            self.CheckNewPacket_Master()
          
        self.normalize_Master(model)  

        # Now communications should work sequentially (not sending a message to next worker until the actual one replied)
        self.display(self.name + ': Preprocessing ready')
    """



    def Update_State_Common_Master(self):
        '''
        Function to control the state of the execution.

        Parameters
        ----------
        None
        '''
        if self.state_dict['CN'] == 'START_INITIALIZATION':
            self.state_dict['CN'] = 'SET_PRECISION'

        if self.checkAllStates('ACK_SET_PRECISION', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SEND_PUBLIC_KEY'
            
        if self.checkAllStates('SEND_PUBLIC_KEY', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SET_NUM_WORKERS'

        #if self.checkAllStates('SET_NUM_FEATURES', self.state_dict):
        #    for worker in self.workers_addresses:
        #        self.state_dict[worker] = ''
        #    self.state_dict['CN'] = 'CHECK_NUM_FEATURES'
                    
        if self.checkAllStates('ACK_SET_NUM_WORKERS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'INITIALIZATION_READY'    

        #if self.state_dict['CN'] == 'START_PREPROCESSING':
        #    self.state_dict['CN'] = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
           
        #if self.checkAllStates('SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE', self.state_dict):
        #    for worker in self.workers_addresses:
        #        self.state_dict[worker] = ''
        #    self.state_dict['CN'] = 'PREPROCESSING_READY'


    
    def TakeAction_Common_Master(self):
        """
        Function to take actions according to the state.

        Parameters
        ----------
        None
        """
        to = 'Preprocessing'

        # Send the precision to encrypt numbers to all workers
        if self.state_dict['CN'] == 'SET_PRECISION':
            action = 'SET_PRECISION'
            data = {'precision': self.precision}
            packet = {'to': to,'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'

        # Ask workers to send public key
        if self.state_dict['CN'] == 'SEND_PUBLIC_KEY':
            action = 'SEND_PUBLIC_KEY'
            packet = {'to': to,'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_PUBLIC_KEYS'

        # Ask the number of features to all workers
        if self.state_dict['CN'] == 'SEND_NUM_FEATURES':
            action = 'SEND_NUM_FEATURES'
            packet = {'to': to,'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_NUM_FEATURES'

        # Check that all workers have the same number of features
        if self.state_dict['CN'] == 'CHECK_NUM_FEATURES':
            if not all(x==self.list_num_features[0] for x in self.list_num_features):
                self.display(self.name + ': Workers have different number of features, terminating POM3 execution')
                self.state_dict['CN'] = 'END'
                return
            self.num_features = self.list_num_features[0]
            self.display(self.name + ': Storing number of features')
            self.state_dict['CN'] = 'SET_NUM_WORKERS'        
        
        # Send the number of centroids to all workers
        if self.state_dict['CN'] == 'SET_NUM_WORKERS':
            action = 'SET_NUM_WORKERS'
            data = {'num_workers': self.Nworkers}
            packet = {'to': to,'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'
            
        # Ask encrypted pseudo random sequence to all workers
        if self.state_dict['CN'] == 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
            action = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            packet = {'to': to,'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
           


    def ProcessReceivedPacket_Master_(self, packet, sender):
        """
        Process the received packet at master.

        Parameters
        ----------
        packet: dictionary
            Packet received from a worker.

        sender: string
            Identification of the sender.
        """
        if self.state_dict['CN'] == 'WAIT_NUM_FEATURES':
            if packet['action'] == 'SET_NUM_FEATURES':
                self.list_num_features.append(packet['data']['num_features']) # Store all number of features
                self.state_dict[sender] = packet['action']
        
        if self.state_dict['CN'] == 'WAIT_PUBLIC_KEYS':
            if packet['action'] == 'SEND_PUBLIC_KEY':
                self.public_keys[sender] = packet['data']['public_key'] # Store all public keys
                self.state_dict[sender] = packet['action']        

        if self.state_dict['CN'] == 'WAIT_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
            if packet['action'] == 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
                self.encrypted_Xi[sender] = packet['data']['encrypted_Xi'] # Store all encrypted Xi
                self.state_dict[sender] = packet['action']

        if packet['action'] == 'EXCEEDED_NUM_CENTROIDS':
            self.display(self.name + ': Number of centroids exceeding training data size worker %s. Terminating training' %str(sender))
            self.state_dict['CN'] = 'END'
        
        


#===============================================================
#                 Worker   
#===============================================================

class POM3_CommonML_Worker(Common_to_POMs_123_Worker):
    '''
    Class implementing the POM3 Common operations, run at Worker node. It inherits from :class:`Common_to_POMs_123_Worker`.
    '''

    def __init__(self, master_address, comms, logger, verbose=False):
        """
        Create a :class:`POM3_CommonML_Worker` instance.

        Parameters
        ----------
        master_address: string
            Identifier of the master instance.

        comms: :class:`Comms_worker`
            Object providing communication functionalities.

        logger: :class:`mylogging.Logger`
            Logging object instance.

        verbose: boolean
            Indicates whether to print messages on screen nor not.
        """
        self.master_address = master_address
        self.comms = comms
        self.logger = logger
        self.verbose = verbose

        self.name = 'POM3_CommonML_Worker'            # Name of the class
        self.worker_address = comms.id                # Id identifying the current worker
        self.platform = comms.name                    # Type of comms to use: either 'pycloudmessenger' or 'localflask'
        self.preprocessors = []                       # List to store all the preprocessors to be applied in sequential order to new data 
        self.num_cores = multiprocessing.cpu_count()  # For parallel processing using all the cores of the machine
        self.modulo = 100                             # Modulo of the sequence Xi and ri to generate by each worker
        self.num_bits = 10                            # Number of bits for seed for generating sequence of Rvalues
        self.c = 829                                  # Value to generate a pseudorandom sequence common to all workers
        self.a = 839                                  # Value to generate a pseudorandom sequence common to all workers
        self.X0 = 809                                 # Value to generate a pseudorandom sequence common to all workers
    
    
    
    def generate_keypair(self, key_size=2048):
        '''
        Function to generate a pair of public/private keys.

        Parameters
        ----------
        None

        Returns
        ----------
        public_key: paillier.PaillierPublicKey
            Public key of the worker.
        private_key: phe.paillier.PaillierPrivateKey
            Private key of the worker.
        '''
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_size)    
        return public_key, private_key
    
    
    
    def encrypt_list(self, unencrypted_list):
        """
        Function to encrypt a list.

        Parameters
        ----------
        unencrypted_list: list of arrays 
            List with values to encrypt.

        Returns
        ----------
        encrypted_list: list of arrays 
            Encrypted list.
        """     
        self.display(self.name + ' %s: Encrypting data...' %self.worker_address)
        encrypted_list = list()
        pool = multiprocessing.Pool(processes=self.num_cores)
        for layer in unencrypted_list:
            layer_list = layer.ravel().tolist() # Convert array to flattened list
            iteration = [(self.public_key, layer_list, self.precision, x) for x in range(len(layer_list))]

            encrypted_layer_list = pool.starmap(parallelization_encryption, iteration)
            encrypted_layer_array = np.asarray(encrypted_layer_list).reshape(layer.shape)
            encrypted_list.append(encrypted_layer_array)
        pool.close() 
        return encrypted_list
    
    
    
    def decrypt_list(self, encrypted_list):
        """
        Function to decrypt a list.

        Parameters
        ----------
        encrypted_list: list of arrays 
            List to decrypt.

        Returns
        ----------
        unencrypted_list: list of arrays 
            Decrypted list.
        """
        self.display(self.name + ' %s: Unencrypting data...' %self.worker_address)
        unencrypted_list = list()
        pool = multiprocessing.Pool(processes=self.num_cores)
        for layer in encrypted_list:
            layer_list = layer.ravel().tolist() 
            iteration = [(self.private_key, layer_list, x) for x in range(len(layer_list))]

            unencrypted_layer_list = pool.starmap(parallelization_decryption, iteration)
            unencrypted_layer_array = np.asarray(unencrypted_layer_list).reshape(layer.shape)
            unencrypted_list.append(unencrypted_layer_array)
        pool.close()
        return unencrypted_list
    

    
    def encrypt_list_rvalues(self, unencrypted_list):
        """
        Function to encrypt a list with obfuscation.

        Parameters
        ----------
        unencrypted_list: list of arrays 
            List with values to encrypt.

        Returns
        ----------
        encrypted_list: list of arrays 
            Encrypted list.
        """   
        self.display(self.name + ' %s: Encrypting data...' %self.worker_address)
        encrypted_list = list()
        start_index = 0
        pool = multiprocessing.Pool(processes=self.num_cores)
        for layer in unencrypted_list:
            layer_list = layer.ravel().tolist()
            slice_r_values = self.r_values[start_index:start_index+len(layer_list)]
            start_index = start_index + len(layer_list)
            iteration = [(self.public_key, layer_list, self.precision, slice_r_values, x) for x in range(len(layer_list))]

            encrypted_layer_list = pool.starmap(parallelization_encryption_rvalues, iteration)
            encrypted_layer_array = np.asarray(encrypted_layer_list).reshape(layer.shape)
            encrypted_list.append(encrypted_layer_array)
        pool.close()
        return encrypted_list
    
    
    
    def encrypt_flattened_list(self, unencrypted_list):
        """
        Function to encrypt a flattened list.

        Parameters
        ----------
        unencrypted_list: list of arrays 
            List with values to encrypt.

        Returns
        ----------
        encrypted_list: list of arrays 
            Encrypted list.
        """   
        self.display(self.name + ' %s: Encrypting data...' %self.worker_address)
        pool = multiprocessing.Pool(processes=self.num_cores)
        iteration = [(self.public_key, unencrypted_list, self.precision, self.r_values, x) for x in range(len(unencrypted_list))]
        encrypted_list = pool.starmap(parallelization_encryption_int, iteration)
        pool.close()
        return encrypted_list
    
    
    
    def generate_sequence_Rvalues(self, seq_length):
        """
        Function to generate a list of obfuscation values.

        Parameters
        ----------
        seq_length: int
            Lenght of the list of obfuscation values.

        Returns
        ----------
        r_values: list
            List with obfuscation values.
        """   
        self.display(self.name + ' %s: Generating random sequence ri for encryption...' %self.worker_address)
        # R values for encryption
        r0 = getprimeover(self.num_bits) # Random number less than modulo of the public key of worker 0
        rc = 1 # For fast computation        
        r_values = [] # Use a list to store sequence in order to avoid overflow of integers in array (in pure Python there is no limit for the largest integer but in np arrays the max integer is 18446744073709551615 for type unsigned int np.uint64)
        r_values.append(r0)         
        for index in range(1, seq_length):
            r_values.append(powmod(r_values[index-1], self.a, self.modulo)) # Fast computation        
        return r_values
        
        
        
    def generate_sequence_Xi(self, seq_length):
        """
        Function to generate a list with a pseudo random sequence that should be common to all workers.

        Parameters
        ----------
        seq_length: int
            Lenght of the list of obfuscation values.

        Returns
        ----------
        Xi: list
            List with pseudo random sequence.
        """  
        self.display(self.name + ' %s: Generating pseudo random sequence Xi...' %self.worker_address)
        Xi = [] # Use a list to store sequence in order to avoid overflow of integers in array (in pure Python there is no limit for the largest integer but in np arrays the max integer is 18446744073709551615 for type unsigned int np.uint64)
        Xi.append(self.X0)
        for index in range(1, seq_length):
            Xi.append((self.a*Xi[index-1] + self.c) %self.modulo)       
        return Xi



    def ProcessPreprocessingPacket(self, packet):
        """
        Take an action after receiving a packet for the preprocessing.

        Parameters
        ----------
        packet: dictionary
            Packet received from master.
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

            # Store the preprocessing object
            self.preprocessors.append(prep_model)
            self.display(self.name + ' %s: Final preprocessor stored' %self.worker_address)

            action = 'ACK_SEND_PREPROCESSOR'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        if packet['action'] == 'SET_PRECISION':
            self.display(self.name + ' %s: Storing precision' %self.worker_address)
            self.precision = packet['data']['precision'] # Store the precision for encryption
            action = 'ACK_SET_PRECISION'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        

        if packet['action'] == 'SEND_PUBLIC_KEY':
            self.display(self.name + ' %s: Generating public/private keys' %self.worker_address)
            self.public_key, self.private_key = self.generate_keypair()
            action = 'SEND_PUBLIC_KEY'
            data = {'public_key': self.public_key}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
            
        if packet['action'] == 'SET_NUM_WORKERS':
            self.display(self.name + ' %s: Storing number of workers' %self.worker_address)
            self.num_workers = packet['data']['num_workers'] # Store the number of centroids
            action = 'ACK_SET_NUM_WORKERS'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
                  
        
        if packet['action'] == 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
            # Review: include here the code to calculate the length of the sequence to generate (we need to know number of centroids in advance)
            # Generate random sequence for encrypting
            self.r_values = self.generate_sequence_Rvalues(self.num_features)
            # Generate pseudo random sequence (the same for all workers)
            Xi = self.generate_sequence_Xi(self.num_features)
            # Encrypt pseudo random sequence using sequence r_values
            encrypted_Xi = self.encrypt_flattened_list(Xi)
            action = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            data = {'encrypted_Xi': encrypted_Xi}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        """
        if packet['action'] == 'SEND_MEANS':
            self.display(self.name + ' %s: Obtaining means' %self.worker_address)
            encrypted_old_means = np.array(packet['data']['means'])
            old_means = np.asarray(self.decrypt_list(encrypted_old_means))
            old_counts = np.array(packet['data']['counts'])

            # Calcutate local worker means
            means = np.mean(self.Xtr_b, axis=0)
            counts = self.Xtr_b.shape[0]

            # Update global means
            new_counts = counts+old_counts
            new_means = (counts/new_counts)*means + (old_counts/new_counts)*old_means

            action = 'COMPUTE_MEANS'
            encrypted_means = np.asarray(self.encrypt_list_rvalues(new_means))
            data = {'means': encrypted_means, 'counts':new_counts}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))            

        if packet['action'] == 'SEND_VARIANCES':
            self.display(self.name + ' %s: Obtaining stds' %self.worker_address)

            encrypted_old_variances = np.array(packet['data']['variances'])
            old_variances = np.asarray(self.decrypt_list(encrypted_old_variances))
            old_counts = np.array(packet['data']['counts'])

            encrypted_global_means = np.array(packet['data']['means'])
            self.global_means = np.asarray(self.decrypt_list(encrypted_global_means))
          
            X_without_mean = self.Xtr_b-self.global_means                              
            variances = np.mean(X_without_mean*X_without_mean, axis=0)
            counts = self.Xtr_b.shape[0]
            
            new_counts = counts+old_counts
            new_variances = (counts/new_counts)*variances + (old_counts/new_counts)*old_variances

            action = 'COMPUTE_VARIANCES'
            encrypted_variances = np.asarray(self.encrypt_list_rvalues(new_variances))
            encrypted_means = np.asarray(self.encrypt_list_rvalues(self.global_means))
            data = {'variances': encrypted_variances, 'counts': counts, 'means': encrypted_means}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
    
            
        if packet['action'] == 'SEND_PREPROCESSOR':
            self.display(self.name + ' %s: Receiving preprocessor' %self.worker_address)

            # Store the preprocessing object
            self.prep_model = packet['data']['model']
            self.prep_model.method = packet['data']['normalization_type']

            # Retrieve global means and variances
            encrypted_global_means = np.array(packet['data']['global_means'])
            global_means = np.asarray(self.decrypt_list(encrypted_global_means))
            encrypted_global_variances = np.array(packet['data']['global_variances'])
            global_variances = np.asarray(self.decrypt_list(encrypted_global_variances))

            # Store the global mean and variance
            self.prep_model.mean = global_means
            self.prep_model.std = np.sqrt(global_variances)
            self.display(self.name + ' %s: Final preprocessor stored' %self.worker_address)

            # Apply the received object to Xtr_b and store back the result
            Xtr = np.copy(self.Xtr_b)
            X_prep = self.prep_model.transform(Xtr)
            self.Xtr_b = np.copy(X_prep)
            self.display(self.name + ' %s: Training set transformed using preprocessor' %self.worker_address)

            action = 'ACK_SEND_PREPROCESSOR'
            encrypted_global_variances = np.asarray(self.encrypt_list_rvalues(global_variances))
            encrypted_global_means = np.asarray(self.encrypt_list_rvalues(global_means))
            data = {'global_variances': encrypted_global_variances, 'global_means': encrypted_global_means}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            self.preprocessor_ready = True
        """

