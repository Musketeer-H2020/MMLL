# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM2.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "December 2020"


import multiprocessing
import numpy as np
import sys
from phe import paillier

from MMLL.models.Common_to_POMs_123 import Common_to_POMs_123_Master, Common_to_POMs_123_Worker
from MMLL.models.POM2.CommonML.parallelization_deep import parallelization_encryption, parallelization_decryption



class POM2_CommonML_Master(Common_to_POMs_123_Master):
    """
    This class implements the Common ML operations, run at Master node. It inherits from :class:`Common_to_POMs_123_Master`.
    """

    def __init__(self, comms, logger, verbose=False):
        """
        Create a :class:`POM2_CommonML_Master` instance.

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

        self.name = 'POM2_CommonML_Master'              # Name
        self.platform = comms.name                      # String with the platform to use (either 'pycloudmessenger' or 'local_flask')
        self.all_workers_addresses = comms.workers_ids  # All addresses of the workers
        self.workers_addresses = comms.workers_ids      # Addresses of the workers
        self.Nworkers = len(self.workers_addresses)     # Nworkers
        self.reset()                                    # Reset variables
        self.public_key = None                          # Initialize public key attribute
        self.state_dict = {}                            # Dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})
        self.num_cores = multiprocessing.cpu_count()    # For parallel processing using all the cores of the machine



    def reset(self):
        """
        Create/reset some variables needed by the Master Node.

        Parameters
        ----------
        None
        """
        self.display(self.name + ': Resetting local data')
        self.list_centroids = []
        self.list_counts = []
        self.list_dists = []
        self.list_public_keys = []
        self.list_gradients = []
        self.list_weights = []
        self.list_costs = []
    
    
    
    def encrypt_list(self, unencrypted_list):
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
            iteration = [(self.public_key, layer_list, x) for x in range(len(layer_list))]

            encrypted_layer_list = pool.starmap(parallelization_encryption, iteration)
            encrypted_layer_array = np.asarray(encrypted_layer_list).reshape(layer.shape)
            encrypted_list.append(encrypted_layer_array)
        pool.close() 
        return encrypted_list

        
        

#===============================================================
#                 Worker   
#===============================================================

class POM2_CommonML_Worker(Common_to_POMs_123_Worker):
    '''
    Class implementing the POM2 Common operations, run at Worker node. It inherits from :class:`Common_to_POMs_123_Worker`.
    '''

    def __init__(self, master_address, comms, logger, verbose=False):
        """
        Create a :class:`POM2_CommonML_Worker` instance.

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

        self.name = 'POM2_CommonML_Worker'                               # Name
        self.worker_address = comms.id                                   # Id identifying the current worker
        self.platform = comms.name                                       # String with the platform to use (either 'pycloudmessenger' or 'local_flask')  
        self.preprocessors = []                                          # List to store all the preprocessors to be applied in sequential order to new data
        self.num_cores = multiprocessing.cpu_count()                     # For parallel processing using all the cores of the machine
        self.public_key, self.private_key = self.generate_keypair()      # Generate encryption keys

    
    
    def generate_keypair(self):
        '''
        Function to generate fixed public/private keys. All workers must have the same keys in this POM.

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
        
        # Code for ensuring that all workers have the same public/private key
        # Define the parameters to have a deterministic private key for every DON
        p = 171559256702780004991283749110701227884652471193128007613924635320100687401096739476028482538656183558591135418057938515173918904905439507695619310083808945858491196504031100176746684862039936705606866775850254074894787149927149836971403702097092044054383995762728123302061602124098217597433472905725418199677
        
        q = 176013204223192844337359459652600248262333480591521011804247258496703137390359333466882732515849382851768958909663925018314107850205286845178145157453667741217449594875069013868481597215510979099591285593412610104574706409528574802816708181995974424279634444995034588162208028893315654240608688923399874009483
        
        # Define parameters to generate deterministic public keys for every worker (the same for all)
        n = p*q        
        
        # Deterministic key pair generation (the same for all DONs) 
        public_key = paillier.PaillierPublicKey(n)
        private_key = paillier.PaillierPrivateKey(public_key, p, q)
    
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
            iteration = [(self.public_key, layer_list, x) for x in range(len(layer_list))]

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



    def decrypt_array(self, encrypted_list):
        """
        Function to decrypt an array.

        Parameters
        ----------
        encrypted_array: array
            Encrypted array.

        Returns
        ----------
        unencrypted_array: array 
            Decrypted array.
        """
        self.display(self.name + ' %s: Unencrypting data...' %self.worker_address)

        pool = multiprocessing.Pool(processes=self.num_cores)
        array_list = encrypted_list.ravel().tolist()
        iteration = [(self.private_key, array_list, x) for x in range(len(array_list))]

        unencrypted_array_list = pool.starmap(parallelization_decryption, iteration)
        unencrypted_array = np.asarray(unencrypted_array_list).reshape(encrypted_list.shape)

        pool.close()
        return unencrypted_array



    def ProcessPreprocessingPacket(self, packet):
        """
        Process the received packet at worker.

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

        """
        if packet['action'] == 'SEND_MEANS':
            self.display(self.name + ' %s: Obtaining means' %self.worker_address)
            self.data_description = np.array(packet['data']['data_description'])
            means = np.mean(self.Xtr_b, axis=0)
            counts = self.Xtr_b.shape[0]
            action = 'COMPUTE_MEANS'
            encrypted_means = np.asarray(self.encrypt_list(means))
            data = {'means': encrypted_means, 'counts': counts}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))            

        if packet['action'] == 'SEND_STDS':
            self.display(self.name + ' %s: Obtaining stds' %self.worker_address)
            encrypted_global_means = np.array(packet['data']['global_means'])
            self.global_means = np.asarray(self.decrypt_list([encrypted_global_means]))          
            X_without_mean = self.Xtr_b-self.global_means                              
            var = np.mean(X_without_mean*X_without_mean, axis=0)
            counts = self.Xtr_b.shape[0]

            action = 'COMPUTE_STDS'
            encrypted_var = np.asarray(self.encrypt_list(var))
            data = {'var': encrypted_var, 'counts':counts}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))    
            
        if packet['action'] == 'SEND_PREPROCESSOR':
            self.display(self.name + ' %s: Receiving preprocessor' %self.worker_address)

            # Store the preprocessing object
            prep_model = packet['data']['prep_model']
            # Decrypt means and stds
            prep_model.mean = np.asarray(self.decrypt_list(prep_model.mean))
            prep_model.std = np.sqrt(np.asarray(self.decrypt_list(prep_model.std)))

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

        if packet['action'] == 'SEND_PUBLIC_KEY':
            action = 'SEND_PUBLIC_KEY'
            data = {'public_key': self.public_key}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        """    


