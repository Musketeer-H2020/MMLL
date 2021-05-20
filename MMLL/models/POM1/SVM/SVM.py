# -*- coding: utf-8 -*-
'''
Semiparametric Support Vector Machine model under POM1.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "December 2020"


import sys
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master, POM1_CommonML_Worker
from MMLL.models.Common_to_models import Common_to_models
from MMLL.models.POM1.Kmeans.Kmeans import Kmeans_Master, Kmeans_Worker



class SVM_model(Common_to_models):
    """
    This class contains the Semiparametric SVM model.
    """

    def __init__(self, logger):
        """
        Create a :class:`SVM_model` instance.

        Parameters
        ----------
        logger: :class:`mylogging.Logger`
            Logging object instance.
        """
        self.logger = logger
        self.is_trained = False
        self.supported_formats = ['pkl', 'onnx', 'pmml']
        self.name = 'SVM'

        self.centroids = None
        self.sigma = None
        self.weights = None



    def kernelMatrix(self, setDim1, setDim2):
        """
        Computes a kernel matrix given two datasets.

        Parameters
        ----------
        setDim1: ndarray
            Array containing M input patterns.

        setDim2: ndarray
            Array containing N input patterns.

        Returns
        -------
        preds: ndarray
            An MxN kernel matrix, every position contains the kernel evaluation of a data from setDim1 with another from setDim2.
        """
        sqrtDists = cdist(setDim1, setDim2, 'euclidean')
        gamma = 1 / (self.sigma**2)
        return np.exp(-gamma * np.power(sqrtDists, 2))



    def predict(self, X_b):
        """
        Uses the model to predict new outputs given the inputs.

        Parameters
        ----------
        X_b: ndarray
            Array containing the input patterns.

        Returns
        -------
        preds: ndarray
            Array containing the predictions.
        """
        Kmn = self.kernelMatrix(X_b, self.centroids)
        softoutput = Kmn.dot(self.weights)
        hardoutput = softoutput
        hardoutput[hardoutput>=0] = 1
        hardoutput[hardoutput<0] = -1
        
        return hardoutput 



class SVM_Master(POM1_CommonML_Master):
    """
    This class implements SVM, run at Master node. It inherits from :class:`POM1_CommonML_Master`.
    """

    def __init__(self, comms, logger, verbose=False, NC=None, Nmaxiter=None, tolerance =None, sigma=None, C=None, NmaxiterGD=None, eta=None):
        """
        Create a :class:`SVM_Master` instance.

        Parameters
        ----------
        comms: :class:`Comms_master`
            Object providing communication functionalities.

        logger: :class:`mylogging.Logger`
            Logging object instance.

        verbose: boolean
            Indicates whether to print messages on screen nor not.

        NC: int
            Number of support vectors in the semiparametric model.

        Nmaxiter: int
            Maximum number of iterations.

        tolerance: float
            Minimum tolerance for continuing training.

        sigma: float
            The parameter of the gaussian kernel.

        C: float
            The cost parameter in the cost function.
            
        NmaxiterGD: int
            Maximum number of iterations for the SVM.

        eta: float
            The step of the gradient descent algorithm.
        """
        self.num_centroids = int(NC)
        self.Nmaxiter = int(Nmaxiter)
        self.tolerance = tolerance
        self.sigma = sigma
        self.C = C
        self.NmaxiterGD = NmaxiterGD
        self.eta = eta

        super().__init__(comms, logger, verbose)                     # Initialize common class for POM1
        self.name = 'POM1_SVM_Master'                                # Name
        self.iter = 0                                                # Number of iterations
        self.model = SVM_model(logger)                               # SVM model
        self.model.sigma = sigma                                     # Initialize sigma
        self.model.weights = np.zeros((self.num_centroids, 1))       # Initialize model weights
        self.is_trained = False                                      # Flag to know if the model is trained
    
    
    
    def Update_State_Master(self):
        '''
        Function to control the state of the execution.

        Parameters
        ----------
        None
        '''
        if self.checkAllStates('ACK_LAUNCH_KMEANS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'TRAIN_KMEANS'

        if self.checkAllStates('ACK_INITIALIZE_PARAMETERS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'COMPUTE_GRADIENTS'

        if self.checkAllStates('UPDATE_GRADIENTS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'UPDATE_MODEL'
    
    
    
    def TakeAction_Master(self):
        """
        Function to take actions according to the state.

        Parameters
        ----------
        None
        """
        to = 'MLmodel'

        # Ask workers to launch Kmeans first
        if self.state_dict['CN'] == 'START_TRAIN':
            self.model.weights = np.zeros((self.num_centroids, 1))       # Initialize model weights
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
            self.model.centroids = kmeans_master.model.centroids # Store the calculated centroids after Kmeans training
        
            # Terminate Kmeans workers
            kmeans_master.terminate_workers_() 
            self.state_dict['CN'] = 'INITIALIZE_PARAMETERS'

        # Asking the workers to send initialize centroids
        if self.state_dict['CN'] == 'INITIALIZE_PARAMETERS':
            action = 'INITIALIZE_PARAMETERS'
            data = {'sigma': self.sigma, 'C': self.C}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Compute average of centroids and mean distance
        if self.state_dict['CN'] == 'UPDATE_MODEL':
            list_gradients = np.array(self.list_gradients) # Array of shape (num_dons x num_centroids x 1)
            list_costs = np.array(self.list_costs)            
            gradients = np.sum(list_gradients, axis=0) # Add received gradients
            self.cost_function = np.sum(list_costs)            
            self.model.weights = self.model.weights - self.eta*gradients

            self.reset()
            self.iter += 1
            self.state_dict['CN'] = 'CHECK_TERMINATION'

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            self.display(self.name + ': Iteration %d, cost function: %0.4f' %(self.iter, self.cost_function))
            if self.Xval is not None:
                predictions = self.model.predict(self.Xval)
                accuracy = accuracy_score(self.yval, predictions)
                self.display(self.name + ': Accuracy on validation set: %0.4f' %accuracy)
            if self.iter == self.NmaxiterGD:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')        
            else:
                self.state_dict['CN'] = 'COMPUTE_GRADIENTS'

        # Asking the workers to compute local gradients
        if self.state_dict['CN'] == 'COMPUTE_GRADIENTS':
            action = 'COMPUTE_LOCAL_GRADIENTS'
            data = {'weights': self.model.weights}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_gradients'
        
        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_FINAL_MODEL':
            action = 'SEND_FINAL_MODEL'
            data = {'weights': self.model.weights}
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
        if self.state_dict['CN'] == 'wait_gradients':
            if packet['action'] == 'UPDATE_GRADIENTS':
                self.list_gradients.append(packet['data']['gradients'])
                self.list_costs.append(packet['data']['cost_function'])
                self.state_dict[sender] = packet['action']


    
    

#===============================================================
#                 Worker   
#===============================================================

class SVM_Worker(POM1_CommonML_Worker):
    '''
    Class implementing a semiparametric SVM, run at Worker node. It inherits from :class:`POM1_CommonML_Worker`.
    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`SVM_Worker` instance.

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

        ytr: ndarray
            Array containing the labels for training.
        """

        '''
        # Train Kmeans first
        kmeans_worker = Kmeans_Worker(master_address, comms, logger, verbose, Xtr_b.copy(), ytr.copy())
        kmeans_worker.run_worker()

        # Save results from Kmeans
        if kmeans_worker.model.centroids is not None:
            self.model.centroids = kmeans_worker.model.centroids       # Save centroids
            self.num_centroids = self.model.centroids.shape[0]         # Number of centroids
        else:
            sys.exit()                                                 # For preprocessing demos
        if len(kmeans_worker.preprocessors) > 0:
            self.Xtr_b = np.copy(kmeans_worker.Xtr_b)                  # Normalize data    
            self.ytr = np.copy(kmeans_worker.ytr)                      # Normalize data            
            self.preprocessors = kmeans_worker.preprocessors           # Store preprocessing objects
        '''   

        self.Xtr_b = Xtr_b
        self.ytr = ytr

        super().__init__(master_address, comms, logger, verbose)       # Initialize common class for POM1
        self.name = 'POM1_SVM_Worker'                                  # Name
        self.model = SVM_model(logger)                                 # Model  
        self.is_trained = False                                        # Flag to know if the model has been trained



    def ProcessReceivedPacket_Worker(self, packet):
        """
        Process the received packet at worker.

        Parameters
        ----------
        packet: dictionary
            Packet received from the master.
        """    
        if packet['action'] == 'LAUNCH_KMEANS':
            self.display(self.name + ' %s: Launching Kmeans worker' %self.worker_address)    
    
            # Initialize Kmeans
            kmeans_worker = Kmeans_Worker(self.master_address, self.comms, self.logger, self.verbose, self.Xtr_b.copy())

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

        if packet['action'] == 'INITIALIZE_PARAMETERS':
            self.display(self.name + ' %s: Storing C and sigma' %self.worker_address)
            self.model.sigma = packet['data']['sigma']
            self.C = packet['data']['C']

            self.Kmn = self.model.kernelMatrix(self.Xtr_b, self.model.centroids)
            self.Y_col = np.reshape(self.ytr, (len(self.ytr), 1))            
            action = 'ACK_INITIALIZE_PARAMETERS'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        
        if packet['action'] == 'COMPUTE_LOCAL_GRADIENTS':
            self.display(self.name + ' %s: Obtaining gradients' %self.worker_address)
            self.model.weights = packet['data']['weights']
            gradients, cost_function = self.get_gradients(self.model.weights)
            action = 'UPDATE_GRADIENTS'
            data = {'cost_function': cost_function, 'gradients': gradients}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))  
            
            
        if packet['action'] == 'SEND_FINAL_MODEL':
            self.display(self.name + ' %s: Receiving final model' %self.worker_address)
            self.model.weights = packet['data']['weights']
            pred_tr = self.model.predict(self.Xtr_b)
            accuracy = accuracy_score(self.ytr, pred_tr)
            self.display(self.name + ' %s: Accuracy in training set: %0.4f' %(self.worker_address, accuracy))
            self.model.is_trained = True
            self.is_trained = True
            self.display(self.name + ' %s: Final model stored' %self.worker_address)
            action = 'ACK_FINAL_MODEL'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
    
   
    
    def get_gradients(self, weights):
        """
        Compute the gradients to update the weights using the local data of this worker node.
        
        Parameters
        ----------
        weights: ndarray
            The weights of the SVM model.

        Returns
        -------
        gradients: ndarray
            The gradient of every weight.
        cost_function: float
            The part of the cost function associated to the local dataset.
        """
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

