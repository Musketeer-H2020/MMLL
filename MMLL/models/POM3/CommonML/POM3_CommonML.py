# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM3

'''

__author__ = "Marcos Fernández Díaz"
__date__ = "June 2020"


import multiprocessing
import numpy as np
import sys
from phe import paillier
from phe.util import getprimeover, powmod, invert

from MMLL.models.POM3.CommonML.POM3_ML import POM3ML
from MMLL.models.POM3.CommonML.parallelization_deep import (parallelization_encryption, parallelization_encryption_rvalues, 
                                                               parallelization_encryption_int, parallelization_decryption, 
                                                               transform_encrypted_domain)



class POM3_CommonML_Master(POM3ML):
    """
    This class implements the Common ML operations, run at Master node. It inherits from POM3ML.
    """

    def __init__(self, workers_addresses, comms, logger, verbose=False):
        """
        Create a :class:`POM3_CommonML_Master` instance.

        Parameters
        ----------
        workers_addresses: list of strings
            list of the addresses of the workers

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen

        """
        self.workers_addresses = workers_addresses
        self.comms = comms
        self.logger = logger
        self.verbose = verbose

        self.name = 'POM3_CommonML_Master'           # Name
        self.precision = 1e-10                       # Precision for encrypting values
        self.num_cores = multiprocessing.cpu_count() # For parallel processing using all the cores of the machine
        self.platform = comms.name                   # Type of comms to use (either 'pycloudmessenger' or 'localflask')



    def checkAllStates(self, condition, state_dict):
        """
        Checks if all worker states satisfy a given condition

        Parameters
        ----------
        condition: String
            Condition to check
        state_dict: Dictionary
            Dictionary whose values need to be compared against condition

        Returns
        ----------
        all_active: Boolean
            Flag indicating if all values inside dictionary are equal to condition
        """
        all_active = True
        for worker in self.workers_addresses:
            if state_dict[worker] != condition:
                all_active = False
                break
        return all_active



    def terminate_Workers(self, workers_addresses_terminate=None):
        """
        Send order to terminate Workers

        Parameters
        ----------
        users_addresses_terminate: List of strings
            Addresses of the workers to be terminated

        """
        packet = {'action': 'STOP'}
        # Broadcast packet to all workers
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ' sent STOP to all Workers')



    def reset(self):
        """
        Create some empty variables needed by the Master Node

        Parameters
        ----------
        None
        """
        self.display(self.name + ': Resetting local data')
        self.list_centroids = []
        self.list_counts = []
        self.list_dists = []
        self.list_num_features = []
        
        
        
    def encrypt_list(self, unencrypted_list, public_key):
        """
        Function to encrypt a list of arrays.

        Parameters
        ----------
        unencrypted_list: List of arrays
            List to encrypt
        public_key: phe.paillier.PaillierPublicKey
            Public key to use for encryption

        Returns
        ----------
        encrypted_list: List of arrays
            Encrypted list
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
    
    
    
    def transform_encrypted_domain_workers(self, encrypted_data, worker_origin, worker_destination):
        """
        Transforms encrypted data from domain of worker_origin to worker_destination

        Parameters
        ----------
        encrypted_data: List of arrays 
            Encrypted data using public key of worker_origin
        worker_origin: int 
            Index of the worker who encrypted the data
        worker_destination: int 
            Index of the worker to transform the encrypted data to

        Returns
        ----------
        transformed_data: List of arrays 
            List of encrypted data in the domain of worker_destination
        """
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
        Sends a message to an specific worker and waits for the reply

        Parameters
        ----------
        packet: dictionary
            Message to be sent to the worker
        worker: string
            Address of the worker to communicate with

        Returns
        ----------
        packet: Dictionary
            Received packet from worker
        """
        self.comms.send(packet, worker)
        self.display(self.name + ': Sent %s to worker %s' %(packet['action'], worker))

        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            while packet == None:
                try:
                    packet = self.comms.receive_poms_123(timeout=10.) # Set a high value for timeout for ensuring that the reply is received within that time
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
                    packet = self.comms.receive(worker, timeout=10.) # Indicate the worker address
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



    def CheckNewPacket_Master(self):
        """
        Checks if there is a new message in the Master queue

        Parameters
        ----------
        None
        """
        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            try:
                packet = self.comms.receive_poms_123(10) # We only receive a dictionary at a time even if there are more than 1 workers

                try:  # For the pycloudmessenger cloud
                    sender = packet.notification['participant']
                except Exception: # For the pycloudmessenger local
                    self.counter = (self.counter + 1) % self.Nworkers
                    sender = self.workers_addresses[self.counter]
                  
                packet = packet.content
                self.display(self.name + ': Received %s from worker %s' %(packet['action'], sender))
                self.ProcessReceivedPacket_Master(packet, sender)
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
            for sender in self.workers_addresses:
                try:
                    packet = self.comms.receive(sender, timeout=10)
                    self.display(self.name + ': Received %s from worker %s' %(packet['action'], sender))
                    self.ProcessReceivedPacket_Master(packet, sender)
                except KeyboardInterrupt:
                    self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                    sys.exit()
                except Exception as err:
                    if str(err).startswith('Timeout when receiving data'): # TimedOutException
                        pass
                    else:
                        self.display(self.name + ': Error %s' %err)
                        raise

        
        

#===============================================================
#                 Worker   
#===============================================================

class POM3_CommonML_Worker(POM3ML):
    '''
    Class implementing the POM3 Common operations, run at Worker

    '''

    def __init__(self, logger, verbose=False):
        """
        Create a :class:`POM3_CommonML_Worker` instance.

        Parameters
        ----------
        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen
        """
        self.logger = logger
        self.verbose = verbose

        self.name = 'POM3_CommonML_Worker'           # Name
        self.num_cores = multiprocessing.cpu_count() # For parallel processing using all the cores of the machine
        self.modulo = 100                            # Modulo of the sequence Xi and ri to generate by each worker
        self.num_bits = 10                           # Number of bits for seed for generating sequence of Rvalues
        self.c = 829                                 # Value to generate a pseudorandom sequence common to all workers
        self.a = 839                                 # Value to generate a pseudorandom sequence common to all workers
        self.X0 = 809                                # Value to generate a pseudorandom sequence common to all workers
    
    
    
    def generate_keypair(self, key_size=2048):
        '''
        Function to generate a pair of public/private key

        Parameters
        ----------
        key_size: Int
            Size of the key in bits

        Returns
        ----------
        public_key: phe.paillier.PaillierPublicKey
            Public key for the worker
        private_key: phe.paillier.PaillierPrivateKey
            Private key for the worker
        '''
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_size)    
        return public_key, private_key
    
    
    
    def encrypt_list(self, unencrypted_list):
        """
        Function to encrypt a list.

        Parameters
        ----------
        unencrypted_list: List of arrays 
            List with values to encrypt

        Returns
        ----------
        encrypted_list: List of arrrays
            Encrypted list
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
        encrypted_list: List of arrays 
            List to decrypt

        Returns
        ----------
        unencrypted_list: List of arrrays
            Decrypted list
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
            List to encrypt

        Returns
        ----------
        encrypted_list: List of arrrays
            Encrypted list
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
        unencrypted_list: List of arrays 
            List to encrypt

        Returns
        ----------
        encrypted_list: List
            Encrypted list
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
        seq_length: Int
            Lenght of the list of obfuscation values

        Returns
        ----------
        r_values: List
            List with obfuscation values
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
        seq_length: Int
            Lenght of the list of obfuscation values

        Returns
        ----------
        Xi: List
            List with pseudo random sequence.
        """  
        self.display(self.name + ' %s: Generating pseudo random sequence Xi...' %self.worker_address)
        Xi = [] # Use a list to store sequence in order to avoid overflow of integers in array (in pure Python there is no limit for the largest integer but in np arrays the max integer is 18446744073709551615 for type unsigned int np.uint64)
        Xi.append(self.X0)
        for index in range(1, seq_length):
            Xi.append((self.a*Xi[index-1] + self.c) %self.modulo)       
        return Xi



    def run_worker(self):
        """
        This is the training executed at every Worker

        Parameters
        ----------
        None
        """
        self.display(self.name + ' %s: READY and waiting instructions' %(self.worker_address))
        self.terminate = False

        while not self.terminate:
            self.CheckNewPacket_worker()
    
    
    
    def CheckNewPacket_worker(self):
        """
        Checks if there is a new message in the Worker queue

        Parameters
        ----------
        None
        """
        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            try:
                packet = self.comms.receive_poms_123(timeout=10)
                packet = packet.content
                sender = 'Master'
                self.display(self.name + ' %s: Received %s from %s' % (self.worker_address, packet['action'], sender))
                self.ProcessReceivedPacket_Worker(packet)
            except KeyboardInterrupt:
                self.display(self.name + '%s: Shutdown requested by Keyboard...exiting' %self.worker_address)
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
            try:
                packet = self.comms.receive(self.master_address, timeout=10)
                sender = 'Master'
                self.display(self.name + ' %s: Received %s from %s' % (self.worker_address, packet['action'], sender))
                self.ProcessReceivedPacket_Worker(packet)
            except KeyboardInterrupt:
                self.display(self.name + '%s: Shutdown requested by Keyboard...exiting' %self.worker_address)
                sys.exit()
            except Exception as err:
                if str(err).startswith('Timeout when receiving data'): # TimedOutException
                    pass
                else:
                    self.display(self.name + ': Error %s' %err)
                    raise

