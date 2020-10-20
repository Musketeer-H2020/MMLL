# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM2

'''

__author__ = "Marcos Fernández Díaz"
__date__ = "June 2020"


import multiprocessing
import numpy as np
import sys
from phe import paillier

from MMLL.models.POM2.CommonML.POM2_ML import POM2ML
from MMLL.models.POM2.CommonML.parallelization_deep import parallelization_encryption, parallelization_decryption



class POM2_CommonML_Master(POM2ML):
    """
    This class implements the Common ML operations, run at Master node. It inherits from POM2ML.
    """

    def __init__(self, workers_addresses, comms, logger, verbose=False):
        """
        Create a :class:`POM2_CommonML_Master` instance.

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

        self.name = 'POM2_CommonML_Master'           # Name
        self.num_cores = multiprocessing.cpu_count() # For parallel processing using all the cores of the machine
        self.platform = comms.name                   # Type of comms to use (either 'pycloudmessenger' or 'local_flask')



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
        self.list_public_keys = []
        self.list_gradients = []
    
    
    
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
    
    
    
    def encrypt_list(self, unencrypted_list):
        """
        Function to encrypt a list of arrays.

        Parameters
        ----------
        unencrypted_list: List of arrays
            List to encrypt

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
            iteration = [(self.public_key, layer_list, x) for x in range(len(layer_list))]

            encrypted_layer_list = pool.starmap(parallelization_encryption, iteration)
            encrypted_layer_array = np.asarray(encrypted_layer_list).reshape(layer.shape)
            encrypted_list.append(encrypted_layer_array)
        pool.close() 
        return encrypted_list



    def train_Master(self):
        """
        This is the main training loop, it runs the following actions until the stop condition is met:
            - Update the execution state
            - Perform actions according to the state
            - Process the received packets

        Parameters
        ----------
        None
        """        
        self.state_dict.update({'CN': 'START_TRAIN'})
        self.display(self.name + ': Starting training')

        while self.state_dict['CN'] != 'END':
            self.Update_State_Master()
            self.TakeAction_Master()
            self.CheckNewPacket_Master()
            
        self.display(self.name + ': Training is done')



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

class POM2_CommonML_Worker(POM2ML):
    '''
    Class implementing the POM2 Common operations, run at Worker

    '''

    def __init__(self, logger, verbose=False):
        """
        Create a :class:`POM2_CommonML_Worker` instance.

        Parameters
        ----------
        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen
        """
        self.logger = logger
        self.verbose = verbose

        self.name = 'POM2_CommonML_Worker'           # Name
        self.num_cores = multiprocessing.cpu_count() # For parallel processing using all the cores of the machine
    
    
    
    def generate_keypair(self):
        '''
        Function to generate fixed public/private keys. All workers must have the same keys.

        Parameters
        ----------
        None

        Returns
        ----------
        public_key: paillier.PaillierPublicKey
            Public key
        private_key: phe.paillier.PaillierPrivateKey
            Private key
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
            List with values to encrypt

        Returns
        ----------
        encrypted_list: List of arrays 
            Encrypted list
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
        encrypted_list: List of arrays 
            List to decrypt

        Returns
        ----------
        unencrypted_list: List of arrays 
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
                    self.display(self.nam<e + ': Error %s' %err)
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

