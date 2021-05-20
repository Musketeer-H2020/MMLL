# -*- coding: utf-8 -*-
'''
Kmeans model under POM1.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "December 2020"


# Code to ensure reproducibility in the results
#from numpy.random import seed
#seed(1)

import numpy as np
from math import floor

from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master, POM1_CommonML_Worker
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
        dists = np.sqrt(np.abs(-2 * np.dot(self.centroids, X_b.T) + np.sum(X_b**2, axis=1) + np.sum(self.centroids**2, axis=1)[:, np.newaxis])) # Shape of vector (num_centroids, num_observations_X_b)
        min_dists = np.min(dists, axis=0) # Array of distances of every observation to the closest centroid
        mean_dists = np.mean(min_dists) # Average distance of all observations to all centroids (scalar)
        preds = np.argmin(dists, axis=0) # Identification of closest centroid for every observation. Shape (num_observations_X_b,)

        return preds




class Kmeans_Master(POM1_CommonML_Master):
    """
    This class implements Kmeans, run at Master node. It inherits from :class:`POM1_CommonML_Master`.
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

        super().__init__(comms, logger, verbose)    # Initialize common class for POM1
        self.name = 'POM1_CommonML_Master'          # Name
        self.mean_dist = np.inf                     # Mean distance 
        self.iter = 0                               # Number of iterations
        self.is_trained = False                     # Flag to know if the model is trained
        self.model = Kmeans_model(logger)           # Kmeans model
            
            

    def Update_State_Master(self):
        '''
        Function to control the state of the execution.

        Parameters
        ----------
        None
        '''
        if self.state_dict['CN'] == 'START_TRAIN':
            self.state_dict['CN'] = 'SEND_CENTROIDS'

        if self.checkAllStates('INIT_CENTROIDS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'AVERAGE_INIT_CENTROIDS'

        if self.checkAllStates('UPDATE_CENTROIDS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'AVERAGE_CENTROIDS'
    
    
    
    def TakeAction_Master(self):
        """
        Function to take actions according to the state.

        Parameters
        ----------
        None
        """
        to = 'MLmodel'

        # Asking the workers to send initialize centroids
        if self.state_dict['CN'] == 'SEND_CENTROIDS':
            action = 'SEND_CENTROIDS'
            data = {'num_centroids': self.num_centroids}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_init_centroids'
            
        # Average the initial centroids from every worker
        if self.state_dict['CN'] == 'AVERAGE_INIT_CENTROIDS':
            self.list_centroids = np.array(self.list_centroids)
            self.model.centroids = np.sum(self.list_centroids, axis=0) / self.Nworkers
            self.reset()
            self.state_dict['CN'] = 'COMPUTE_LOCAL_CENTROIDS'

        # Compute average of centroids and mean distance
        if self.state_dict['CN'] == 'AVERAGE_CENTROIDS':
            list_centroids = np.array(self.list_centroids) # Array of shape (num_dons x num_centroids x num_features)
            list_counts = np.array(self.list_counts) # Array of shape (num_dons x num_centroids)
            list_dists = np.array(self.list_dists) # Array of shape (num_dons x 1)
            
            # Average all mean distances received from each DON according to total number of observations per DON with respect to the total 
            # observations including all DONs
            self.new_mean_dist = np.dot(list_dists.T, np.sum(list_counts, axis=1)) / np.sum(list_counts[:,:])

            # Average centroids taking into account the number of observations of the training set in each DON with respect to the total
            # including the training observations of all DONs
            if np.all(np.sum(list_counts, axis=0)): # If all centroids have at least one observation in one of the DONs
                self.model.centroids = np.sum((list_centroids.T * (list_counts / np.sum(list_counts, axis=0)).T).T, axis=0) # Shape (num_centroids x num_features)
            else: # Modify only non-empty centroids
                for i in range(self.num_centroids):
                    if np.sum(list_counts[:,i])>0:
                        self.model.centroids[i,:] = np.zeros_like(list_centroids[0, i])
                        for kdon in range(self.Nworkers):
                            self.model.centroids[i,:] = self.model.centroids[i,:]+list_centroids[kdon,i,:]*list_counts[kdon,i]/np.sum(list_counts[:,i])
            self.reset()
            self.iter += 1
            self.state_dict['CN'] = 'CHECK_TERMINATION'

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            self.display(self.name + ': Average distance to closest centroid: %0.4f, iteration %d' %(self.new_mean_dist, self.iter))
            if self.iter == self.Nmaxiter:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')
            elif self.tolerance >= 0:
                if np.abs(self.new_mean_dist-self.mean_dist) < self.tolerance:
                    self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                    self.display(self.name + ': Stopping training, minimum tolerance reached!')
                else:
                    self.state_dict['CN'] = 'COMPUTE_LOCAL_CENTROIDS'
                    self.mean_dist = self.new_mean_dist

        # Asking the workers to compute local centroids
        if self.state_dict['CN'] == 'COMPUTE_LOCAL_CENTROIDS':
            action = 'COMPUTE_LOCAL_CENTROIDS'
            data = {'centroids': self.model.centroids}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_update_centroids'
        
        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_FINAL_MODEL':
            action = 'SEND_FINAL_MODEL'
            data = {'centroids': self.model.centroids}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent %s to all workers' %action)
            self.model.is_trained = True
            self.is_trained = True
            self.state_dict['CN'] = 'wait'
            


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
        if packet['action'] == 'EXCEEDED_NUM_CENTROIDS':
            self.display(self.name + ': Number of centroids exceeding training data size worker %s. Terminating training' %str(sender))
            self.state_dict['CN'] = 'END'
        
        if self.state_dict['CN'] == 'wait_init_centroids':
            if packet['action'] == 'INIT_CENTROIDS':
                self.list_centroids.append(packet['data']['centroids'])
                self.state_dict[sender] = packet['action']

        if self.state_dict['CN'] == 'wait_update_centroids':
            if packet['action'] == 'UPDATE_CENTROIDS':
                if self.check_empty_clusters(packet['data']['centroids']):
                    self.display(self.name + ': Received empty clusters from worker %s. Terminating training' %str(sender))
                    self.state_dict['CN'] = 'END'
                else:
                    self.list_centroids.append(packet['data']['centroids'])
                    self.list_counts.append(packet['data']['counts'])
                    self.list_dists.append(packet['data']['mean_dist'])
                    self.state_dict[sender] = packet['action']
    
    
    
    def check_empty_clusters(self, array):
        """
        Function to check if there are empty clusters in array.
        
        Parameters
        ----------
        array: numpy array
            Array with centroids.

        Returns
        -------
        flag: boolean
            Flag indicating whether there are empty clusters.
        """
        flag = False
        if array.shape[0] != self.num_centroids:
            flag = True
        return flag

    
    

#===============================================================
#                 Worker   
#===============================================================

class Kmeans_Worker(POM1_CommonML_Worker):
    '''
    Class implementing Kmeans, run at Worker node. It inherits from :class:`POM1_CommonML_Worker`.
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

        super().__init__(master_address, comms, logger, verbose)    # Initialize common class for POM1
        self.name = 'POM1_KMeans_Worker'                            # Name
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
        if packet['action'] == 'SEND_CENTROIDS':
            self.display(self.name + ' %s: Initializing centroids' %self.worker_address)
            self.num_centroids = packet['data']['num_centroids']
            
            # Check maximum number of possible centroids
            if self.num_centroids > self.Xtr_b.shape[0]:
                self.display(self.name + ' %s: Number of clusters exceeds number of training samples. Terminating training' %self.worker_address)
                action = 'EXCEEDED_NUM_CENTROIDS'
                packet = {'action': action}
                
            else:   
                # Random point initialization (data leakage in POM1)         
                # Suffle randomly the observations in the training set
                # np.random.shuffle(self.Xtr_b)
                # centroids = self.Xtr_b[:self.num_centroids, :] # Take the first K observations, this avoids selecting the same point twice

                # Naive sharding initialization (no leakage in POM1)
                centroids = self.naive_sharding(self.Xtr_b, self.num_centroids)

                action = 'INIT_CENTROIDS'
                data = {'centroids': centroids}
                packet = {'action': action, 'data': data}
                
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))            
            
        if packet['action'] == 'COMPUTE_LOCAL_CENTROIDS':
            self.display(self.name + ' %s: Updating centroids' %self.worker_address)
            centroids = packet['data']['centroids']
            
            dists = np.sqrt(np.abs(-2 * np.dot(centroids, self.Xtr_b.T) + np.sum(self.Xtr_b**2, axis=1) + np.sum(centroids**2, axis=1)[:, np.newaxis])) # Matrix of euclidean distances between all observations in training set and centroids. Shape of vector (num_centroids x num_observations_X)
            min_dists = np.min(dists, axis=0) # Array of distances of every observation to the closest centroid
            mean_dists = np.mean(min_dists) # Average distance of all observations to all centroids (scalar)
            cluster_allocs = np.argmin(dists, axis=0) # Identification of closest centroid for every observation
            counts = np.bincount(cluster_allocs, minlength=self.num_centroids) # Count the number of observations in each cluster (shape (num_centroids, )
            
            clusters = []
            self.model.centroids = centroids.copy()
            for i in range(self.num_centroids):
                clusters.append(self.Xtr_b[cluster_allocs==i])
                if counts[i]>0:
                    self.model.centroids[i,:] = (1/len(clusters[i]))*np.sum(clusters[i], axis=0)

            action = 'UPDATE_CENTROIDS'
            data = {'centroids': self.model.centroids, 'counts': counts, 'mean_dist': mean_dists}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))            
            
        if packet['action'] == 'SEND_FINAL_MODEL':
            self.display(self.name + ' %s: Receiving final model' %self.worker_address)
            self.model.centroids = packet['data']['centroids']
            self.model.is_trained = True
            self.is_trained = True
            self.display(self.name + ' %s: Final model stored' %self.worker_address)

            action = 'ACK_FINAL_MODEL'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))



    def naive_sharding(self, ds, k):
        """
        Initialize cluster centroids using deterministic naive sharding algorithm.
    
        Parameters
        ----------
        ds: numpy array
            The dataset to be used for centroid initialization.
        k: int
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
        sums: float
            The summed shard columns.
        step: int
            The number of instances per shard.

        Returns
        -------
        sums/step (means): numpy array
            The means of the shard columns.
        """

        return sums/step

