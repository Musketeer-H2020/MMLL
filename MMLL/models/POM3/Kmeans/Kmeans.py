# -*- coding: utf-8 -*-
'''
Kmeans model under POM3.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "January 2021"


# Code to ensure reproducibility in the results
#from numpy.random import seed
#seed(1)

import numpy as np
from math import floor

from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master, POM3_CommonML_Worker
from MMLL.models.Common_to_models import Common_to_models



class Kmeans_model(Common_to_models):
    """
    This class contains the Kmeans model.
    """

    def __init__(self, logger):
        """
        Create a :class:`Kmeans_model` instance.

        Parameters
        ----------
        logger: :class:`mylogging.Logger`
            Logging object instance.
        """
        self.logger = logger
        self.is_trained = False
        self.supported_formats = ['pkl', 'onnx', 'pmml']
        self.name = 'Kmeans'
        self.centroids = None


    def predict(self, X_b):
        """
        Uses the Kmeans model to predict new outputs given the inputs.

        Parameters
        ----------
        X_b: ndarray
            Array containing the input patterns.

        Returns
        -------
        preds: ndarray
            Array containing the predictions.
        """
        # Calculate the vector with euclidean distances between all observations and the defined centroids        
        dists = dists = np.sqrt(np.abs(-2 * np.dot(self.centroids, X_b.T) + np.sum(X_b**2, axis=1) + np.sum(self.centroids**2, axis=1)[:, np.newaxis])) # Shape of vector (num_centroids, num_observations_X_b)
        min_dists = np.min(dists, axis=0) # Array of distances of every observation to the closest centroid
        mean_dists = np.mean(min_dists) # Average distance of all observations to all centroids (scalar)
        preds = np.argmin(dists, axis=0) # Identification of closest centroid for every observation. Shape (num_observations_X_b,)

        return preds



class Kmeans_Master(POM3_CommonML_Master):
    """
    This class implements Kmeans, run at Master node. It inherits from :class:`POM3_CommonML_Master`.
    """

    def __init__(self, comms, logger, verbose=False, NC=None, Nmaxiter=None, tolerance=None):
        """
        Create a :class:`Kmeans_Master` instance.

        Parameters
        ----------
        comms: :class:`Comms_master`
            Object providing communication functionalities.

        logger: :class:`mylogging.Logger`
            Logging object instance.

        verbose: boolean
            Indicates whether to print messages on screen nor not.

        NC: int
            Number of clusters.

        Nmaxiter: int
            Maximum number of iterations.

        tolerance: float
            Minimum tolerance for continuing training.
        """
        self.num_centroids = int(NC)
        self.Nmaxiter = int(Nmaxiter)
        self.tolerance = tolerance

        super().__init__(comms, logger, verbose)            # Initialize common class for POM3
        self.name = 'POM3_Kmeans_Master'                    # Name of the class
        #self.Init_Environment()                             # Send initialization messages common to all algorithms
        self.iter = -1                                      # Number of iterations
        self.mean_dist = np.inf                             # Mean distance to centroids 
        self.is_trained = False                             # Flag to know if the model has been trained
            
            

    def train_Master_(self):
        """
        Main loop controlling the training of the algorithm.

        Parameters
        ----------
        None
        """   
        self.iter = -1
        self.mean_dist = np.inf
        self.is_trained = False

        self.Init_Environment() 
        self.state_dict['CN'] = 'START_TRAIN'

        while self.state_dict['CN'] != 'TRAINING_READY':
            self.Update_State_Master()
            self.TakeAction_Master()
            self.CheckNewPacket_Master()
            
        # Now communications should work sequentially (not sending a message to next worker until the actual one replied)
        self.display(self.name + ': Initialization ready, starting sequential communications')
        zero_counts = np.zeros((self.num_centroids, 1))
        zero_centroids = np.zeros((self.num_centroids, self.num_features))
        encrypted_zero_centroids = np.asarray(self.encrypt_list(zero_centroids, self.public_keys[self.workers_addresses[0]]))

        while self.iter != self.Nmaxiter:
            #added
            if self.iter!=-1:
                encrypted_iteration_centroids = encrypted_centroids
            encrypted_centroids = encrypted_zero_centroids
            new_mean_dist = 0
            counts = zero_counts
            #end added
            for index_worker, worker in enumerate(self.workers_addresses): 
                if self.iter==-1:
                    action = 'SEND_CENTROIDS' # Initialize centroids
                    data = {'accumulated_centroids': encrypted_centroids}
                    packet = {'to':'MLModel','action': action, 'data': data}
                    
                else:
                    # Get updated centroids from each worker
                    action = 'COMPUTE_LOCAL_CENTROIDS'
                    data = {'iteration_centroids': encrypted_iteration_centroids, 'accumulated_centroids': encrypted_centroids, 'counts': counts, 'mean_dist': new_mean_dist}
                    packet = {'to':'MLModel', 'action': action, 'data': data}
                                
                # Send message to specific worker and wait until receiving reply
                packet = self.send_worker_and_wait_receive(packet, worker)
                
                encrypted_centroids = packet['data']['accumulated_centroids']
                # Transform encrypted centroids to the encrypted domain of the next worker
                encrypted_centroids = self.transform_encrypted_domain_workers(encrypted_centroids, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])

                if packet['action'] == 'UPDATE_CENTROIDS':
                    counts = packet['data']['counts']
                    new_mean_dist = packet['data']['mean_dist']
                    encrypted_iteration_centroids = packet['data']['iteration_centroids']
                    encrypted_iteration_centroids = self.transform_encrypted_domain_workers(encrypted_iteration_centroids, worker, self.workers_addresses[(index_worker+1)%self.Nworkers], verbose=False)

            self.iter += 1

            # Check for termination at the end of each iteration according to the tolerance
            if packet['action']=='UPDATE_CENTROIDS':
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
            packet = {'to':'MLModel', 'action': action, 'data': data}            
            
            # Send message to specific worker and wait until receiving reply
            packet = self.send_worker_and_wait_receive(packet, worker)
            encrypted_centroids = packet['data']['centroids']
            encrypted_centroids = self.transform_encrypted_domain_workers(encrypted_centroids, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])
         
        self.is_trained = True   
        self.display(self.name + ': Training is done')
    
    
    
    def Update_State_Master(self):
        '''
        Function to control the state of the execution.

        Parameters
        ----------
        None
        '''
        if self.state_dict['CN'] == 'START_TRAIN':
            self.state_dict['CN'] = 'SET_NUM_CENTROIDS'
         
        if self.checkAllStates('ACK_SET_NUM_CENTROIDS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SEND_NUM_FEATURES'

        if self.checkAllStates('SET_NUM_FEATURES', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'CHECK_NUM_FEATURES'
   
        if self.checkAllStates('SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'TRAINING_READY'
    
    
    
    def TakeAction_Master(self):
        """
        Function to take actions according to the state.

        Parameters
        ----------
        None
        """
        to = 'MLmodel'
        
        # Send the number of centroids to all workers
        if self.state_dict['CN'] == 'SET_NUM_CENTROIDS':
            action = 'SET_NUM_CENTROIDS'
            data = {'num_centroids': self.num_centroids}
            packet = {'to': to,'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'

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
            self.state_dict['CN'] = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'         
            
        # Ask encrypted pseudo random sequence to all workers
        if self.state_dict['CN'] == 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
            action = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            packet = {'to': to,'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'


    

#===============================================================
#                 Worker   
#===============================================================

class Kmeans_Worker(POM3_CommonML_Worker):
    '''
    Class implementing Kmeans, run at Worker node. It inherits from :class:`POM3_CommonML_Worker`.
    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None):
        """
        Create a :class:`Kmeans_Worker` instance.

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

        Xtr_b: ndarray
            Array containing the inputs for training.
        """
        self.Xtr_b = Xtr_b

        super().__init__(master_address, comms, logger, verbose)    # Initialize common class for POM3
        self.name = 'POM3_Kmeans_Worker'                            # Name of the class
        self.num_features = Xtr_b.shape[1]                          # Number of features
        self.model = Kmeans_model(logger)                           # Model    
        self.is_trained = False                                     # Flag to know if the model has been trained
        
        

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Process the received packet at worker.

        Parameters
        ----------
        packet: dictionary
            Packet received from the master.
        """        
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


        if packet['action'] == 'SEND_NUM_FEATURES':
            self.display(self.name + ' %s: Sending number of features' %self.worker_address)
            action = 'SET_NUM_FEATURES'
            data = {'num_features': self.num_features}
            packet = {'action': action, 'data': data}
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
            encrypted_accumulated_centroids = packet['data']['accumulated_centroids']
            accumulated_centroids = np.asarray(self.decrypt_list(encrypted_accumulated_centroids))
            
            # Random point initialization (data leakage in POM1)  
            # Suffle randomly the observations in the training set
            # np.random.shuffle(self.Xtr_b)
            # centroids_local = self.Xtr_b[:self.num_centroids, :] # Take the first K observations, this avoids selecting the same point twice
            # accumulated_centroids += centroids_local/self.num_workers

            # Naive sharding initialization (no leakage in POM1)
            centroids = self.naive_sharding(self.Xtr_b, self.num_centroids)
            accumulated_centroids += centroids/self.num_workers

            # Encrypt centroids before sending them to the master
            encrypted_accumulated_centroids = np.asarray(self.encrypt_list_rvalues(list(accumulated_centroids)))
            action = 'INIT_CENTROIDS'
            data = {'accumulated_centroids': encrypted_accumulated_centroids}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            

        if packet['action'] == 'COMPUTE_LOCAL_CENTROIDS':
            self.display(self.name + ' %s: Updating centroids' %self.worker_address)
            old_counts = packet['data']['counts']
            old_mean_dist = packet['data']['mean_dist']
            encrypted_accumulated_centroids = packet['data']['accumulated_centroids']
            encrypted_iteration_centroids = packet['data']['iteration_centroids']

            # Unencrypt received centroids
            accumulated_centroids = np.asarray(self.decrypt_list(encrypted_accumulated_centroids))
            iteration_centroids = np.asarray(self.decrypt_list(encrypted_iteration_centroids))

            # Calculate the vector with euclidean distances between all observations and the defined centroids        
            dists = np.sqrt(np.abs(-2 * np.dot(iteration_centroids, self.Xtr_b.T) + np.sum(self.Xtr_b**2, axis=1) + np.sum(iteration_centroids**2, axis=1)[:, np.newaxis])) # Matrix of euclidean distances between all observations in training set and centroids. Shape of vector (num_centroids x num_observations_X)
            min_dists = np.min(dists, axis=0) # Array of distances of every observation to the closest centroid
            mean_dist = np.mean(min_dists) # Average distance of all observations to all centroids (scalar)
            cluster_allocs = np.argmin(dists, axis=0) # Identification of closest centroid for every observation
            counts = np.bincount(cluster_allocs, minlength=self.num_centroids).reshape(-1,1) # Count the number of observations in each cluster (shape (num_centroids, )
            
            # Compute local centroids
            clusters = []
            centroids = accumulated_centroids.copy()
            for i in range(self.num_centroids):
                clusters.append(self.Xtr_b[cluster_allocs==i])
                if counts[i]>0:
                    centroids[i,:] = (1/len(clusters[i]))*np.sum(clusters[i], axis=0)

            # Update accumulated centroids, counts and mean_dist
            new_counts = counts + old_counts
            new_mean_dist = np.sum(counts)/np.sum(new_counts)*mean_dist + np.sum(old_counts)/np.sum(new_counts)*old_mean_dist

            # Check empty clusters
            if np.any(new_counts==0):
                new_centroids = accumulated_centroids.copy()
                for i in range(self.num_centroids):
                    if new_counts[i]>0:
                        new_centroids[i,:] = counts[i]/new_counts[i]*centroids[i,:] + old_counts[i]/new_counts[i]*accumulated_centroids[i,:]
            else:
                # Average centroids taking into account the number of observations of the training set in each worker with respect to the total, including the training observations of all workers
                new_centroids = (counts/new_counts)*centroids + (old_counts/new_counts)*accumulated_centroids # Array broadcasting

            # Encrypt centroids before sending them to the master
            encrypted_accumulated_centroids = np.asarray(self.encrypt_list_rvalues(new_centroids))
            encrypted_iteration_centroids = np.asarray(self.encrypt_list_rvalues(iteration_centroids))
            action = 'UPDATE_CENTROIDS'
            data = {'iteration_centroids': encrypted_iteration_centroids, 'accumulated_centroids': encrypted_accumulated_centroids, 'counts': new_counts, 'mean_dist': new_mean_dist}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'SEND_FINAL_MODEL':
            self.display(self.name + ' %s: Receiving final model' %self.worker_address)
            encrypted_centroids = packet['data']['centroids']
            self.model.centroids = np.asarray(self.decrypt_list(encrypted_centroids))
            self.model.is_trained = True
            self.is_trained = True
            self.display(self.name + ' %s: Final model stored' %self.worker_address)

            encrypted_centroids = np.asarray(self.encrypt_list_rvalues(self.model.centroids))
            action = 'UPDATE_CENTROIDS_FINAL_MODEL'
            data = {'centroids': encrypted_centroids}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))



    def naive_sharding(self, ds, k):
        """
        Create cluster centroids using deterministic naive sharding algorithm.
    
        Parameters
        ----------
        ds : numpy array
            The dataset to be used for centroid initialization.
        k : int
            The desired number of clusters for which centroids are required.

        Returns
        -------
        centroids : numpy array
            Collection of k centroids as a numpy array.
        """        
        n = ds.shape[1]
        m = ds.shape[0]
        centroids = np.zeros((k,n))
    
        # Sum all elements of each row, add as col to original dataset, sort
        composite = np.sum(ds, axis=1)
        composite = np.expand_dims(composite, axis=1)
        ds = np.append(composite, ds, axis=1)
        ds.sort(axis=0)
    
        # Step value for dataset sharding
        step = floor(m/k)
    
        # Vectorize mean ufunc for numpy array
        vfunc = np.vectorize(self._get_mean)
    
        # Divide matrix rows equally by k-1 (so that there are k matrix shards)
        # Sum columns of shards, get means; these columnar means are centroids
        for j in range(k):
            if j == k-1:
                centroids[j:] = vfunc(np.sum(ds[j*step:,1:], axis=0), step)
            else:
                centroids[j:] = vfunc(np.sum(ds[j*step:(j+1)*step,1:], axis=0), step)
    
        return centroids



    def _get_mean(self, sums, step):
        """
        Vectorizable ufunc for getting means of summed shard columns.
        
        Parameters
        ----------
        sums : float
            The summed shard columns.
        step : int
            The number of instances per shard.
        Returns
        -------
        sums/step (means) : numpy array
            The means of the shard columns.
        """

        return sums/step

