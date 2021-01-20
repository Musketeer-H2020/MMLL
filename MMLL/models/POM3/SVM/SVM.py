# -*- coding: utf-8 -*-
'''
SVM model 

'''

__author__ = "Marcos Fernández Díaz"
__date__ = "January 2021"


import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master, POM3_CommonML_Worker
from MMLL.models.POM3.Kmeans.Kmeans import Kmeans_Master, Kmeans_Worker



class model():
    """
    This class contains the Kmeans model
    """

    def __init__(self):
        """
        Initializes Kmeans model centroids
        """
        self.centroids = None
        self.sigma = None
        self.weights = None


    def kernelMatrix(self, setDim1, setDim2):
        sqrtDists = cdist(setDim1, setDim2, 'euclidean')
        gamma = 1 / (self.sigma**2)
        return np.exp(-gamma * np.power(sqrtDists, 2))


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
        Kmn = self.kernelMatrix(X_b, self.centroids)
        softoutput = Kmn.dot(self.weights)
        hardoutput = softoutput
        hardoutput[hardoutput>=0] = 1
        hardoutput[hardoutput<0] = -1
        
        return hardoutput 



class SVM_Master(POM3_CommonML_Master):
    """
    This class implements SVM, run at Master node. It inherits from POM3_CommonML_Master.
    """

    def __init__(self, comms, logger, verbose=False, NC=None, Nmaxiter=None, tolerance =None, sigma=None, C=None, NmaxiterGD=None, eta=None):
        """
        Create a :class:`SVM_Master` instance.

        Parameters
        ----------
        comms: comms object instance
            Object providing communication functionalities

        logger: class `mylogging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen

        NC: int
            Number of clusters

        Nmaxiter: int
            Maximum number of iterations

        tolerance: float
            Minimum tolerance for continuing training

        sigma: float

        C: 
            
        NmaxiterGD: int
            Maximum number of iterations for the SVM
        eta: float
            
        """
        self.num_centroids = int(NC)
        self.Nmaxiter = int(Nmaxiter)
        self.tolerance = tolerance
        self.sigma = sigma
        self.C = C
        self.NmaxiterGD = NmaxiterGD
        self.eta = eta

        super().__init__(comms, logger, verbose)                     # Initialize common class for POM3
        """
        # Initialize Kmeans first
        self.kmeans_master = Kmeans_Master(self.comms, self.logger, self.verbose, self.num_centroids, self.Nmaxiter, self.tolerance)
        self.public_keys = self.kmeans_master.public_keys            # Store the public keys of all workers
        self.num_features = self.kmeans_master.num_features          # Store encrypted sequences of all workers
        """
        self.name = 'POM3_SVM_Master'                                # Name of the class
        self.weights = np.zeros((self.num_centroids, 1))             # Weights for the SVM
        self.iter = 0                                                # Number of iterations already executed
        self.is_trained = False                                      # Flag to know if the model is trained
        self.initialization_ready = False                            # Flag to know if the initialization needed for POM3 is ready



    def train_Master_(self):
        """
        This is the main training loop, it runs the following actions until the stop condition is met:
            - Update the execution state
            - Perform actions according to the state
            - Process the received packets

        Parameters
        ----------
        None
        """    
        self.weights = np.zeros((self.num_centroids, 1))
        self.iter = 0
        self.is_trained = False 
        self.state_dict['CN'] = 'START_TRAIN'
    
        while self.state_dict['CN'] != 'INITIALIZATION_READY':
            self.Update_State_Master()
            self.TakeAction_Master()
            self.CheckNewPacket_Master()

        # Now communications should work sequentially (not sending a message to next worker until the actual one replied)
        self.display(self.name + ': Initialization ready, starting sequential communications')
        encrypted_weights = self.encrypt_list(self.weights, self.public_keys[self.workers_addresses[0]]) # Encrypt weights using worker 0 public key

        while self.iter != self.NmaxiterGD:
            for index_worker, worker in enumerate(self.workers_addresses): 
                action = 'UPDATE_MODEL'
                data = {'weights': encrypted_weights}
                packet = {'to': 'MLModel', 'action': action, 'data': data}
                
                # Send message to specific worker and wait until receiving reply
                packet = self.send_worker_and_wait_receive(packet, worker)                    
                encrypted_weights = packet['data']['weights']
                encrypted_weights = self.transform_encrypted_domain_workers(encrypted_weights, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])

            self.iter += 1
            self.display(self.name + ': Iteration %d' %self.iter)
            
        self.display(self.name + ': Stopping training, maximum number of iterations reached!')            
        action = 'SEND_FINAL_MODEL'
        for index_worker, worker in enumerate(self.workers_addresses):
            data = {'weights': encrypted_weights}
            packet = {'to': 'MLModel', 'action': action, 'data': data}

            # Send message to specific worker and wait until receiving reply
            packet = self.send_worker_and_wait_receive(packet, worker)
            encrypted_weights = packet['data']['weights']
            encrypted_weights = self.transform_encrypted_domain_workers(encrypted_weights, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])
            
        self.is_trained = True
        self.display(self.name + ': Training is done')
    
    
    
    def Update_State_Master(self):
        '''
        We update the state of execution.
        We control from here the data flow of the training process
        ** By now there is only one implemented option: direct transmission **

        This code needs some improvement...
        '''
        if self.checkAllStates('ACK_LAUNCH_KMEANS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'TRAIN_KMEANS'

        if self.checkAllStates('ACK_INITIALIZE_PARAMETERS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'INITIALIZATION_READY'
    
    
    
    def TakeAction_Master(self):
        """
        Takes actions according to the state
        """
        to = 'MLmodel'

        # Train Kmeans first
        if self.state_dict['CN'] == 'START_TRAIN':
            action = 'LAUNCH_KMEANS'
            packet = {'to': to, 'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Train Kmeans
        if self.state_dict['CN'] == 'TRAIN_KMEANS':
            kmeans_master = Kmeans_Master(self.comms, self.logger, self.verbose, self.num_centroids, self.Nmaxiter, self.tolerance)
            kmeans_master.workers_addresses = self.workers_addresses
            kmeans_master.Nworkers = self.Nworkers
            kmeans_master.train_Master()
            self.public_keys = kmeans_master.public_keys # Store the public key
            self.encrypted_Xi = kmeans_master.encrypted_Xi
        
            # Terminate Kmeans workers
            kmeans_master.terminate_workers_() 
            self.state_dict['CN'] = 'INITIALIZE_PARAMETERS'

        # Asking the workers to send initialize centroids
        if self.state_dict['CN'] == 'INITIALIZE_PARAMETERS':
            action = 'INITIALIZE_PARAMETERS'
            data = {'sigma': self.sigma, 'C': self.C, 'eta': self.eta}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'


    

#===============================================================
#                 Worker   
#===============================================================

class SVM_Worker(POM3_CommonML_Worker):
    '''
    Class implementing SVM, run at Worker

    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`SVM_Worker` instance.

        Parameters
        ----------
        master_address: string
            Identifier of the master instance

        comms: comms object instance
            Object providing communication functionalities

        logger: class:`logging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen

        Xtr_b: ndarray
            2-D numpy array containing the input training patterns

        ytr: ndarray
            1-D numpy array containing the target training patterns
        """
        self.Xtr_b = Xtr_b
        self.ytr = ytr

        super().__init__(master_address, comms, logger, verbose)      # Initialize common class for POM3
        self.name = 'POM3_SVM_Worker'                                 # Name of the class
        self.model = model()                                          # Model  
        self.is_trained = False                                       # Flag to know if the model has been trained

        """
        # Train Kmeans first
        kmeans_worker = Kmeans_Worker(master_address, comms, logger, verbose, Xtr_b.copy())
        kmeans_worker.run_worker()

        # Save results from Kmeans
        self.model.centroids = kmeans_worker.model.centroids          # Save centroids
        self.num_centroids = self.model.centroids.shape[0]            # Number of centroids 
        self.public_key = kmeans_worker.public_key                    # Public key
        self.private_key = kmeans_worker.private_key                  # Private key
        self.precision = kmeans_worker.precision                      # Store precision
        self.r_values = kmeans_worker.r_values                        # Random values for encryption
        self.preprocessor_ready = kmeans_worker.preprocessor_ready    # Store flag for preprocessing

        if self.preprocessor_ready:
            self.Xtr_b = np.copy(kmeans_worker.Xtr_b)                 # Normalize data                
            self.prep_model = kmeans_worker.prep_model                # Store preprocessing object
        """
    
    

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Take an action after receiving a packet

        Parameters
        ----------
        packet: Dictionary
            Packet received
        """
        if packet['action'] == 'LAUNCH_KMEANS':
            self.display(self.name + ' %s: Launching Kmeans worker' %self.worker_address)    
    
            # Initialize Kmeans
            kmeans_worker = Kmeans_Worker(self.master_address, self.comms, self.logger, self.verbose, self.Xtr_b.copy())
            #kmeans_worker.public_key = self.public_key
            #kmeans_worker.private_key = self.private_key
            #kmeans_worker.precision = self.precision
            #kmeans_worker.num_workers = self.num_workers
            #self.preprocessors = kmeans_worker.preprocessors

            # Reply to master
            action = 'ACK_LAUNCH_KMEANS'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

            # Run Kmeans
            kmeans_worker.run_worker()
            if kmeans_worker.model.centroids is not None:
                self.model.centroids = kmeans_worker.model.centroids       # Save centroids
                self.num_centroids = self.model.centroids.shape[0]         # Number of centroids
                self.public_key = kmeans_worker.public_key                 # Public key
                self.private_key = kmeans_worker.private_key               # Private key
                self.r_values = kmeans_worker.r_values                     # Random values for encryption
                self.precision = kmeans_worker.precision                   # Precision
                self.num_workers = kmeans_worker.num_workers               # Number of workers
                self.preprocessors = kmeans_worker.preprocessors           # Preprocessors


        if packet['action'] == 'INITIALIZE_PARAMETERS':
            self.display(self.name + ' %s: Storing C and sigma' %self.worker_address)
            self.model.sigma = packet['data']['sigma']
            self.C = packet['data']['C']
            self.eta = packet['data']['eta']

            self.Kmn = self.model.kernelMatrix(self.Xtr_b, self.model.centroids)
            self.Y_col = np.reshape(self.ytr, (len(self.ytr), 1))            
            action = 'ACK_INITIALIZE_PARAMETERS'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        
        if packet['action'] == 'UPDATE_MODEL':
            self.display(self.name + ' %s: Obtaining gradients' %self.worker_address)
            encrypted_weights = packet['data']['weights']
            # Unencrypt received centroids 
            self.model.weights = np.asarray(self.decrypt_list(encrypted_weights))
            gradients, cost_function = self.get_gradients(self.model.weights)
            self.model.weights = self.model.weights - self.eta*gradients

            encrypted_weights = np.asarray(self.encrypt_list_rvalues(self.model.weights))
            action = 'UPDATE_MODEL'
            data = {'cost_function': cost_function, 'weights': encrypted_weights}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))  
            
            
        if packet['action'] == 'SEND_FINAL_MODEL':
            self.display(self.name + ' %s: Receiving final model' %self.worker_address)
            encrypted_weights = packet['data']['weights']
            self.model.weights = np.asarray(self.decrypt_list(encrypted_weights))
            pred_tr = self.model.predict(self.Xtr_b)
            accuracy = accuracy_score(self.ytr, pred_tr)
            self.display(self.name + ' %s: Accuracy in training set: %0.4f' %(self.worker_address, accuracy))
            self.is_trained = True
            self.display(self.name + ' %s: Final model stored' %self.worker_address)
            
            # Encrypt again final model weights
            encrypted_weights = self.encrypt_list_rvalues(self.model.weights)            
            action = 'UPDATE_MODEL'
            data = {'weights': encrypted_weights}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
    
   
    
    def get_gradients(self, weights):
        cost_function = 0
        gradients = np.zeros((self.num_centroids, 1))
        hinge = self.Kmn.dot(weights) * self.Y_col
        cost_samples = 1 - hinge
            
        cost_samples[cost_samples<0] = 0
        cost_function = cost_function + self.C * cost_samples.sum()

        classification = np.zeros((len(self.ytr), 1))
        classification[hinge<1] = 1
        errors = self.Y_col * classification

        clYC = classification * self.C * (-1 * self.Y_col)
        KmnclYC = self.Kmn * clYC
        gradients = gradients + KmnclYC.sum(axis=0).reshape((self.num_centroids, 1))
        
        return gradients, cost_function

