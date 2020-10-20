# -*- coding: utf-8 -*-
'''
Kmeans model 

'''

__author__ = "Marcos Fernández Díaz"
__date__ = "June 2020"


# Code to ensure reproducibility in the results
from numpy.random import seed
seed(1)

import numpy as np

from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master, POM3_CommonML_Worker



class model():
    """
    This class contains the Kmeans model
    """

    def __init__(self):
        """
        Initializes Kmeans model centroids
        """
        self.centroids = None


    def predict(self, X_b):
        """
        Predicts outputs given the model and inputs

        Parameters
        ----------
        X_b: ndarray
            2-D numpy array containing the input patterns

        Returns
        -------
        preds: ndarray
            1-D array containing the predictions
        """
        # Calculate the vector with euclidean distances between all observations and the defined centroids        
        dists = dists = np.sqrt(np.abs(-2 * np.dot(self.centroids, X_b.T) + np.sum(X_b**2, axis=1) + np.sum(self.centroids**2, axis=1)[:, np.newaxis])) # Shape of vector (num_centroids, num_observations_X_b)
        min_dists = np.min(dists, axis=0) # Array of distances of every observation to the closest centroid
        mean_dists = np.mean(min_dists) # Average distance of all observations to all centroids (scalar)
        preds = np.argmin(dists, axis=0) # Identification of closest centroid for every observation. Shape (num_observations_X_b,)

        return preds



class Kmeans_Master(POM3_CommonML_Master):
    """
    This class implements Kmeans, run at Master node. It inherits from POM3_CommonML_Master.
    """

    def __init__(self, comms, logger, verbose=False, NC=None, Nmaxiter=None, tolerance=None):
        """
        Create a :class:`Kmeans_Master` instance.

        Parameters
        ----------
        comms: comms object instance
            Object providing communication functionalities

        logger: class:`mylogging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen

        NC: int
            Number of clusters

        Nmaxiter: int
            Maximum number of iterations

        tolerance: float
            Minimum tolerance for continuing training
        """
        self.comms = comms   
        self.logger = logger
        self.verbose = verbose   
        self.num_centroids = int(NC)
        self.Nmaxiter = int(Nmaxiter)
        self.tolerance = tolerance

        self.name = 'POM3_Kmeans_Master'              # Name
        self.platform = comms.name                    # String with the platform to use (either 'pycloudmessenger' or 'local_flask')
        self.workers_addresses = comms.workers_ids    # Addresses of the workers
        self.Nworkers = len(self.workers_addresses)   # Nworkers
        super().__init__(self.workers_addresses, comms, logger, verbose)
        self.num_features = None                      # Number of features
        self.iter = -1                                # Number of iterations
        self.mean_dist = np.inf                       # Mean distance 
        self.is_trained = False
        self.reset()
        self.public_keys = {}                         # Dictionary to store public keys from all workers
        self.encrypted_Xi = {}                        # Dictionary to store encrypted Xi from all workers
        self.state_dict = {}                          # Dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})
            
            

    def train_Master(self):
        """
        This is the main training loop, it runs the following actions until 
        the stop condition is met:
            - Update the execution state
            - Perform actions according to the state
            - Process the received packets

        Parameters
        ----------
        None
        """   
        self.state_dict.update({'CN': 'START_TRAIN'})
        self.display(self.name + ': Starting training')

        while self.state_dict['CN'] != 'INITIALIZATION_READY':
            self.Update_State_Master()
            self.TakeAction_Master()
            self.CheckNewPacket_Master()
            
        # Now communications should work sequentially (not sending a message to next worker until the actual one replied)
        self.display(self.name + ': Initialization ready, starting sequential communications')
        centroids = np.zeros((self.num_centroids, self.num_features))
        encrypted_centroids = np.asarray(self.encrypt_list(centroids, self.public_keys[self.workers_addresses[0]]))
        while self.iter != self.Nmaxiter:
            new_mean_dist = 0
            total_counts = 0
            for index_worker, worker in enumerate(self.workers_addresses): 
                if self.iter==-1:
                    action = 'SEND_CENTROIDS' # Initialize centroids
                    data = {'centroids': encrypted_centroids}
                    packet = {'action': action, 'data': data}
                    
                else:
                    # Get updated centroids from each worker
                    action = 'COMPUTE_LOCAL_CENTROIDS'
                    data = {'centroids': encrypted_centroids}
                    packet = {'action': action, 'data': data}
                                
                # Send message to specific worker and wait until receiving reply
                packet = self.send_worker_and_wait_receive(packet, worker)                    
                encrypted_centroids = packet['data']['centroids']
                # Transform encrypted centroids to the encrypted domain of the next worker
                encrypted_centroids = self.transform_encrypted_domain_workers(encrypted_centroids, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])

                if packet['action'] == 'UPDATE_CENTROIDS':
                    counts = packet['data']['counts']
                    mean_dist = packet['data']['mean_dist']
                    new_mean_dist += mean_dist*np.sum(counts, axis=0)
                    total_counts += np.sum(counts, axis=0)

            self.iter += 1

            # Check for termination at the end of each iteration according to the tolerance
            if packet['action']=='UPDATE_CENTROIDS':
                new_mean_dist = new_mean_dist / total_counts
                self.display(self.name + ': Average distance to closest centroid: %0.4f, iteration %d' %(new_mean_dist, self.iter))

                # Check for termination at the end of each iteration according to the tolerance
                if self.iter == self.Nmaxiter:
                    self.display(self.name + ': Stopping training, maximum number of iterations reached!')
                    break
                elif self.tolerance >= 0:
                    if np.abs(self.mean_dist-new_mean_dist) < self.tolerance:
                        self.display(self.name + ': Stopping training, minimum tolerance reached!')
                        break
                    else:
                        self.mean_dist = new_mean_dist
        
        # Send final model to workers
        action = 'SEND_FINAL_MODEL'
        for index_worker, worker in enumerate(self.workers_addresses):
            data = {'centroids': encrypted_centroids}
            packet = {'action': action, 'data': data}            
            
            # Send message to specific worker and wait until receiving reply
            packet = self.send_worker_and_wait_receive(packet, worker)
            encrypted_centroids = packet['data']['centroids']
            encrypted_centroids = self.transform_encrypted_domain_workers(encrypted_centroids, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])
         
        self.is_trained = True   
        self.display(self.name + ': Training is done')
    
    
    
    def Update_State_Master(self):
        '''
        Function to control the state of the execution
        '''
        if self.state_dict['CN'] == 'START_TRAIN':
            self.state_dict['CN'] = 'SET_PRECISION'

        if self.checkAllStates('ACK_SET_PRECISION', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SEND_PUBLIC_KEY'
            
        if self.checkAllStates('SEND_PUBLIC_KEY', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SEND_NUM_FEATURES'

        if self.checkAllStates('SET_NUM_FEATURES', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'CHECK_NUM_FEATURES'
        
        if self.checkAllStates('ACK_SET_NUM_CENTROIDS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SET_NUM_WORKERS'
            
        if self.checkAllStates('ACK_SET_NUM_WORKERS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            
        if self.checkAllStates('SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'INITIALIZATION_READY'
    
    
    
    def TakeAction_Master(self):
        """
        Takes actions according to the state
        """
        # Send the precision to encrypt numbers to all workers
        if self.state_dict['CN'] == 'SET_PRECISION':
            action = 'SET_PRECISION'
            data = {'precision': self.precision}
            packet = {'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'

        # Ask workers to send public key
        if self.state_dict['CN'] == 'SEND_PUBLIC_KEY':
            action = 'SEND_PUBLIC_KEY'
            packet = {'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_PUBLIC_KEYS'

        # Ask the number of features to all workers
        if self.state_dict['CN'] == 'SEND_NUM_FEATURES':
            action = 'SEND_NUM_FEATURES'
            packet = {'action': action}
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
            self.state_dict['CN'] = 'SET_NUM_CENTROIDS'
        
        # Send the number of centroids to all workers
        if self.state_dict['CN'] == 'SET_NUM_CENTROIDS':
            action = 'SET_NUM_CENTROIDS'
            data = {'num_centroids': self.num_centroids}
            packet = {'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'
        
        # Send the number of centroids to all workers
        if self.state_dict['CN'] == 'SET_NUM_WORKERS':
            action = 'SET_NUM_WORKERS'
            data = {'num_workers': self.Nworkers}
            packet = {'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'
            
        # Ask encrypted pseudo random sequence to all workers
        if self.state_dict['CN'] == 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
            action = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            packet = {'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            


    def ProcessReceivedPacket_Master(self, packet, sender):
        """
        Process the received packet at Master and take some actions, possibly changing the state

        Parameters
        ----------
        packet: Dictionary
            Packet received

        sender: String
            Id of the sender
        """
        if packet['action'][0:3] == 'ACK':
            self.state_dict[sender] = packet['action']

        if packet['action'] == 'EXCEEDED_NUM_CENTROIDS':
            self.display(self.name + ': Number of centroids exceeding training data size worker %s. Terminating training' %str(sender))
            self.state_dict['CN'] = 'END'

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


    
    

#===============================================================
#                 Worker   
#===============================================================

class Kmeans_Worker(POM3_CommonML_Worker):
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

        comms: comms object instance
            Object providing communication functionalities

        logger: class:`mylogging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen

        Xtr_b: ndarray
            2-D numpy array containing the input training patterns
        """
        self.master_address = master_address
        self.comms = comms
        self.logger = logger
        self.verbose = verbose
        self.Xtr_b = Xtr_b

        super().__init__(logger, verbose)
        self.name = 'POM3_Kmeans_Worker'             # Name
        self.worker_address = comms.id               # Id identifying the current worker
        self.platform = comms.name                   # Type of comms to use: either 'pycloudmessenger' or 'localflask'
        self.num_features = Xtr_b.shape[1]           # Number of features
        self.model = model()                         # Model    
        self.is_trained = False                      # Flag to know if the model has been trained
        
        

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Take an action after receiving a packet

        Parameters
        ----------
        packet: Dictionary
            Packet received
        """        
        self.terminate = False

        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' %self.worker_address)
            self.terminate = True

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

        if packet['action'] == 'SEND_NUM_FEATURES':
            self.display(self.name + ' %s: Sending number of features' %self.worker_address)
            action = 'SET_NUM_FEATURES'
            data = {'num_features': self.num_features}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        if packet['action'] == 'SET_NUM_CENTROIDS':
            self.display(self.name + ' %s: Storing number of centroids' %self.worker_address)
            self.num_centroids = packet['data']['num_centroids'] # Store the number of centroids
            
            # Check maximum number of possible centroids
            if self.num_centroids > self.Xtr_b.shape[0]:
                self.display(self.name + ' %s: Number of clusters exceeds number of training samples. Terminating training' %self.worker_address)
                action = 'EXCEEDED_NUM_CENTROIDS'
                packet = {'action': action}
            else:
                action = 'ACK_SET_NUM_CENTROIDS'
                packet = {'action': action}
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
            self.r_values = self.generate_sequence_Rvalues(self.num_centroids*self.num_features)
            # Generate pseudo random sequence (the same for all workers)
            Xi = self.generate_sequence_Xi(self.num_centroids*self.num_features)
            # Encrypt pseudo random sequence using sequence r_values
            encrypted_Xi = self.encrypt_flattened_list(Xi)
            action = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            data = {'encrypted_Xi': encrypted_Xi}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        if packet['action'] == 'SEND_CENTROIDS':
            self.display(self.name + ' %s: Initializing centroids' %self.worker_address)
            encrypted_centroids = packet['data']['centroids']
            centroids = np.asarray(self.decrypt_list(encrypted_centroids))
            
            # Suffle randomly the observations in the training set
            np.random.shuffle(self.Xtr_b)
            centroids_local = self.Xtr_b[:self.num_centroids, :] # Take the first K observations, this avoids selecting the same point twice
            centroids += centroids_local

            # Encrypt centroids before sending them to the master
            encrypted_centroids = np.asarray(self.encrypt_list_rvalues(list(centroids)))
            action = 'INIT_CENTROIDS'
            data = {'centroids': encrypted_centroids}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'COMPUTE_LOCAL_CENTROIDS':
            self.display(self.name + ' %s: Updating centroids' %self.worker_address)
            encrypted_centroids = packet['data']['centroids']
            # Unencrypt received centroids
            centroids = np.asarray(self.decrypt_list(encrypted_centroids))
            # Calculate the vector with euclidean distances between all observations and the defined centroids        
            dists = np.sqrt(np.abs(-2 * np.dot(centroids, self.Xtr_b.T) + np.sum(self.Xtr_b**2, axis=1) + np.sum(centroids**2, axis=1)[:, np.newaxis])) # Matrix of euclidean distances between all observations in training set and centroids. Shape of vector (num_centroids x num_observations_X)
            min_dists = np.min(dists, axis=0) # Array of distances of every observation to the closest centroid
            mean_dists = np.mean(min_dists) # Average distance of all observations to all centroids (scalar)
            cluster_allocs = np.argmin(dists, axis=0) # Identification of closest centroid for every observation
            counts = np.bincount(cluster_allocs, minlength=self.num_centroids) # Count the number of observations in each cluster (shape (num_centroids, )
            
            clusters = []
            copy_centroids = centroids.copy()
            for i in range(self.num_centroids):
                clusters.append(self.Xtr_b[cluster_allocs==i])
                if counts[i]>0:
                    copy_centroids[i,:] = (1/len(clusters[i]))*np.sum(clusters[i], axis=0)

            encrypted_centroids = np.asarray(self.encrypt_list_rvalues(copy_centroids))
            action = 'UPDATE_CENTROIDS'
            data = {'centroids': encrypted_centroids, 'counts': counts, 'mean_dist':mean_dists}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'SEND_FINAL_MODEL':
            self.display(self.name + ' %s: Receiving final model' %self.worker_address)
            encrypted_centroids = packet['data']['centroids']
            self.model.centroids = np.asarray(self.decrypt_list(encrypted_centroids))
            self.is_trained = True
            self.display(self.name + ' %s: Final model stored' %self.worker_address)

            encrypted_centroids = np.asarray(self.encrypt_list_rvalues(self.model.centroids))
            action = 'UPDATE_CENTROIDS_FINAL_MODEL'
            data = {'centroids': encrypted_centroids}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
