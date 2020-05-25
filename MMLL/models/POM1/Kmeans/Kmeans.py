# -*- coding: utf-8 -*-
'''
Kmeans model 

'''

__author__ = "Marcos Fernández Díaz"
__date__ = "May 2020"

# Code to ensure reproducibility in the results
from numpy.random import seed
seed(1)

import numpy as np

from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master, POM1_CommonML_Worker



class model():

    def __init__(self):
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
        dists = np.sqrt(np.abs(-2 * np.dot(self.centroids, X_b.T) + np.sum(X_b**2, axis=1) + np.sum(self.centroids**2, axis=1)[:, np.newaxis])) # Shape of vector (num_centroids, num_observations_X_b)
        min_dists = np.min(dists, axis=0) # Array of distances of every observation to the closest centroid
        mean_dists = np.mean(min_dists) # Average distance of all observations to all centroids (scalar)
        preds = np.argmin(dists, axis=0) # Identification of closest centroid for every observation. Shape (num_observations_X_b,)

        return preds




class Kmeans_Master(POM1_CommonML_Master):
    """
    This class implements Kmeans, run at Master node. It inherits from POM1_CommonML_Master.
    """

    def __init__(self, comms, logger, verbose=False, NC=None, Nmaxiter=None, tolerance=None):
        """
        Create a :class:`Kmeans_Master` instance.

        Parameters
        ----------
        master_address: string
            Identifier of the master instance

        workers_addresses: string
            Address of the workers to sent/receive message to/from

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

        NC: int
            Number of clusters

        Nmaxiter: int
            Maximum number of iterations

        tolerance: float
            Minimum tolerance for continuing training
        """
        #super().__init__(comms, logger, verbose)
        self.name = 'POM1_Kmeans_Master'            # Name
        
        self.comms = comms                          # comms lib
        self.logger = logger                        # logger
        self.verbose = verbose                      # print on screen when true       
        self.num_centroids = NC                     # No. Centroids
        self.Nmaxiter = Nmaxiter
        self.tolerance = tolerance

        self.platform = comms.name                  # Type of comms to use: either 'pycloudmessenger' or 'localflask'
        self.workers_addresses = comms.workers_ids  # Addresses of the workers
        self.Nworkers = len(self.workers_addresses) # Nworkers
        self.counter = -1
        self.mean_dist = np.inf        
        self.num_features = None
        self.iter = 0
        self.is_trained = False
        self.model = model()
        self.reset()
        self.state_dict = {}                        # dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})
            
            

    def Update_State_Master(self):
        '''
        We update the state of execution.
        We control from here the data flow of the training process
        ** By now there is only one implemented option: direct transmission **

        This code needs some improvement...
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
        Takes actions according to the state
        """
        # Asking the workers to send initialize centroids
        if self.state_dict['CN'] == 'SEND_CENTROIDS':
            action = 'SEND_CENTROIDS'
            data = {'num_centroids': self.num_centroids}
            packet = {'action': action, 'data': data}
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
                        self.model.centroids[i,:] = np.zeros(self.num_features)
                        for kdon in range(self.Nworkers):
                            self.model.centroids[i,:] = self.model.centroids[i,:]+list_centroids[kdon,i,:]*list_counts[kdon,i]/np.sum(list_counts[:,i])

            self.reset()
            self.iter += 1
            self.state_dict['CN'] = 'CHECK_TERMINATION'

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            if self.iter == self.Nmaxiter:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Average distance to closest centroid: %0.4f, iteration %d' %(self.new_mean_dist, self.iter))
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')
            elif self.tolerance >= 0:
                if np.abs(self.new_mean_dist-self.mean_dist) < self.tolerance:
                    self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                    self.display(self.name + ': Average distance to closest centroid: %0.4f, iteration %d' %(self.new_mean_dist, self.iter))
                    self.display(self.name + ': Stopping training, minimum tolerance reached!')
                else:
                    self.state_dict['CN'] = 'COMPUTE_LOCAL_CENTROIDS'
                    self.display(self.name + ': Average distance to closest centroid: %0.4f, iteration %d' %(self.new_mean_dist, self.iter))
                    self.mean_dist = self.new_mean_dist

        # Asking the workers to compute local centroids
        if self.state_dict['CN'] == 'COMPUTE_LOCAL_CENTROIDS':
            action = 'COMPUTE_LOCAL_CENTROIDS'
            data = {'centroids': self.model.centroids}
            packet = {'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_update_centroids'
        
        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_FINAL_MODEL':
            action = 'SEND_FINAL_MODEL'
            data = {'centroids': self.model.centroids}
            packet = {'action': action, 'data': data}
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
            if self.checkAllStates('ACK_FINAL_MODEL', self.state_dict): # Included here to avoid calling CheckNewPacket_Master after sending the final model (this call could imply significant delay if timeout is set to a high value)
                self.state_dict['CN'] = 'END'

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
        if array.shape[0] != self.num_centroids:
            return True
        return False

    
    

#===============================================================
#                 Worker   
#===============================================================

class Kmeans_Worker(POM1_CommonML_Worker):
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
        #super().__init__(logger, verbose)
        self.name = 'POM2_KMeans_Worker'        # Name

        self.master_address = master_address
        self.comms = comms                      # The comms library
        self.logger = logger                    # logger
        self.verbose = verbose                  # print on screen when true
        self.Xtr_b = Xtr_b

        self.worker_address = comms.id
        self.platform = comms.name
        self.num_features = Xtr_b.shape[1]
        self.model = model()
        self.is_trained = False
        
        

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
            
        
        if packet['action'] == 'SEND_CENTROIDS':
            self.display(self.name + ' %s: Initializing centroids' %self.worker_address)
            self.num_centroids = packet['data']['num_centroids']
            
            # Check maximum number of possible centroids
            if self.num_centroids > self.Xtr_b.shape[0]:
                self.display(self.name + ' %s: Number of clusters exceeds number of training samples. Terminating training' %self.worker_address)
                action = 'EXCEEDED_NUM_CENTROIDS'
                packet = {'action': action}
                
            else:            
                # Suffle randomly the observations in the training set
                np.random.shuffle(self.Xtr_b)
                centroids = self.Xtr_b[:self.num_centroids, :] # Take the first K observations, this avoids selecting the same point twice

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
            self.is_trained = True
            self.display(self.name + ' %s: Final model stored' %self.worker_address)

            action = 'ACK_FINAL_MODEL'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))