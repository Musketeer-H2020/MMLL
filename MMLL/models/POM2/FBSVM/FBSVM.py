# -*- coding: utf-8 -*-
'''
Federated Budget Support Vector Machine model under POM2.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "April 2021"


import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc

from MMLL.models.POM2.CommonML.POM2_CommonML import POM2_CommonML_Master, POM2_CommonML_Worker
from MMLL.models.Common_to_models import Common_to_models



class FBSVM_model(Common_to_models):
    """
    This class contains the Federated Budgeted SVM model.
    """

    def __init__(self, logger):
        """
        Create a :class:`FBSVM_model` instance.

        Parameters
        ----------
        logger: :class:`mylogging.Logger`
            Logging object instance.
        """
        self.logger = logger
        self.is_trained = False
        self.supported_formats = ['pkl', 'onnx', 'pmml']
        self.name = 'FBSVM'

        self.centroids = None
        self.weights = None
        self.sigma = None



    def predict(self, X_b):
        """
        Use the model to predict new outputs given for an unlabeled dataset.

        Parameters
        ----------
        X_b: ndarray
            Array containing the input patterns.

        Returns
        -------
        preds: ndarray
            Array containing the predictions.
        """
        NP = X_b.shape[0]
        NC = self.centroids.shape[0]
        XC2 = -2 * np.dot(X_b, self.centroids.T)
        XC2 += np.sum(np.multiply(X_b, X_b), axis=1).reshape((NP, 1))
        XC2 += np.sum(np.multiply(self.centroids, self.centroids), axis=1).reshape((1, NC))
        
        KXC = np.exp(-XC2 / 2.0 /  (self.sigma ** 2))
        KXC = np.hstack( (np.ones((NP, 1)), KXC))
        prediction_values = np.dot(KXC, self.weights)

        return prediction_values




class FBSVM_Master(POM2_CommonML_Master):
    """
    This class implements FBSVM, run at Master node. It inherits from :class:`POM2_CommonML_Master`.
    """

    def __init__(self, comms, logger, verbose=False, NC=None, Nmaxiter=None, tolerance=None, sigma=None, C=None, num_epochs_worker=None, eps=None, mu=None, NI=None, minvalue=None, maxvalue=None):
        """
        Create a :class:`FBSVM_Master` instance.

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
            The cost parameter in the cost function .
            
        num_epochs_worker: int
            Number of epochs in every worker before sending the weights to the master node in every iteration.

        eps: float
            Threshold to update the a variables in the IRWLS algorithm.

        mu: float
            Step to update the weights in the master node after every iteration.

        NI: int
            Number of data features.

        minvalue: float
            The centroids are initialized randomly from an uniforme distribution. This is the minimum value.

        maxvalue: float
            The centroids are initialized randomly from an uniforme distribution. This is the maximum value.
        """
        self.num_centroids = int(NC)
        self.Nmaxiter = int(Nmaxiter)
        self.tolerance = tolerance
        self.sigma = sigma
        self.C = C
        self.num_epochs_worker = int(num_epochs_worker)
        self.eps = eps
        self.mu = mu
        self.num_features = NI
        self.minvalue = minvalue
        self.maxvalue = maxvalue

        super().__init__(comms, logger, verbose)             # Initialize common class for POM2
        self.name = 'POM2_FBSVM_Master'                      # Name
        self.iter = 0                                        # Number of iterations
        self.is_trained = False                              # Flag to know if the model is trained
    
    
    
    def Update_State_Master(self):
        '''
        Function to control the state of the execution.

        Parameters
        ----------
        None
        '''
        if self.state_dict['CN'] == 'START_TRAIN':
            if self.public_key is None:
                self.state_dict['CN'] = 'SEND_PUBLIC_KEY'
            else:
                self.state_dict['CN'] = 'INITIALIZE_PARAMETERS'

        if self.checkAllStates('SEND_PUBLIC_KEY', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'CHECK_PUBLIC_KEYS'  

        if self.checkAllStates('ACK_INITIALIZE_PARAMETERS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'LOCAL_TRAIN'

        if self.checkAllStates('UPDATE_MODEL', self.state_dict):
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

        # Ask workers to send public key
        if self.state_dict['CN'] == 'SEND_PUBLIC_KEY':
            action = 'SEND_PUBLIC_KEY'
            packet = {'to': to,'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_PUBLIC_KEYS'
        
        # Checking public keys received from workers
        if self.state_dict['CN'] == 'CHECK_PUBLIC_KEYS':
            if not all(x==self.list_public_keys[0] for x in self.list_public_keys):
                self.display(self.name + ': Workers have different keys, terminating POM2 execution')
                self.state_dict['CN'] = 'END'
                return
            self.public_key = self.list_public_keys[0]
            self.display(self.name + ': Storing public key from workers')
            self.state_dict['CN'] = 'START_TRAIN'

        # Compute initial centroids and weights
        if self.state_dict['CN'] == 'START_TRAIN':
            self.display(self.name + ': Initializing centroids...')

            # Random initial values       
            centroids = np.random.uniform(self.minvalue, self.maxvalue, (int(1.1*self.num_centroids), self.num_features))

            # We prune the closest ones
            C2 = np.sum(centroids ** 2, axis=1).reshape((-1, 1))
            Dcc = np.abs(C2 -2 * np.dot(centroids, centroids.T) + C2.T)
            Kcc = np.exp(-Dcc / 2 / (self.sigma**2))
            Kcc = Kcc - np.eye(Dcc.shape[0])
            stop_poda = False
            while not stop_poda:
                # Computing kcc
                maximos = np.max(Kcc, axis=0)
                max_pos = np.argmax(Kcc, axis=0)
                max_pos2 = np.argmax(maximos)
                max_pos1 = max_pos[max_pos2]
                kcc = Kcc[max_pos1, max_pos2]
                if centroids.shape[0] > self.num_centroids:
                    # Delete the first
                    centroids = np.delete(centroids, [max_pos1], 0)
                    Kcc = np.delete(Kcc, [max_pos1], 0)
                    Kcc = np.delete(Kcc, [max_pos1], 1)
                else:
                    stop_poda = True

            # Initialize parameters
            self.centroids = centroids
            self.weights = np.random.normal(0, .0001, (self.num_centroids+1, 1))
            self.weights = self.encrypt_list(list(self.weights))
            self.zero_grad = self.encrypt_list(list(np.zeros((self.num_centroids+1, 1))))
            self.state_dict['CN'] = 'INITIALIZE_PARAMETERS'

        # Asking the workers to send initialize centroids
        if self.state_dict['CN'] == 'INITIALIZE_PARAMETERS':
            action = 'INITIALIZE_PARAMETERS'
            data = {'sigma': self.sigma, 'C': self.C, 'centroids': self.centroids, 'eps': self.eps, 'num_epochs_worker': self.num_epochs_worker}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Compute average of centroids and mean distance
        if self.state_dict['CN'] == 'UPDATE_MODEL':
            old_weights = np.copy(self.weights)
            grad = self.zero_grad
            for kworker in range(len(self.list_weights)):
                grad_worker = self.list_weights[kworker] - self.weights
                grad += grad_worker

            grad = grad / len(self.list_weights)
            self.weights = self.weights + self.mu * grad

            self.reset()
            self.iter += 1
            self.state_dict['CN'] = 'CHECK_TERMINATION'

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            self.display(self.name + ': Iteration %d' %self.iter)
            if self.iter == self.Nmaxiter:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')   
            else:
                self.state_dict['CN'] = 'LOCAL_TRAIN'

        # Asking the workers to compute local gradients
        if self.state_dict['CN'] == 'LOCAL_TRAIN':
            action = 'LOCAL_TRAIN'
            data = {'weights': self.weights}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_weights'
        
        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_FINAL_MODEL':
            action = 'SEND_FINAL_MODEL'
            data = {'weights': self.weights}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent %s to all workers' %action)
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
        if self.state_dict['CN'] == 'WAIT_PUBLIC_KEYS':
            if packet['action'] == 'SEND_PUBLIC_KEY':
                self.list_public_keys.append(packet['data']['public_key'])
                self.state_dict[sender] = packet['action']

        if self.state_dict['CN'] == 'wait_weights':
            if packet['action'] == 'UPDATE_MODEL':
                self.list_weights.append(packet['data']['weights'])
                self.state_dict[sender] = packet['action']


    
    

#===============================================================
#                 Worker   
#===============================================================

class FBSVM_Worker(POM2_CommonML_Worker):
    '''
    Class implementing FBSVM, run at Worker. It inherits from :class:`POM2_CommonML_Worker`.
    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`FBSVM_Worker` instance.

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
        self.Xtr_b = Xtr_b
        self.ytr = ytr.reshape((-1, 1))

        super().__init__(master_address, comms, logger, verbose)    # Initialize common class for POM1
        self.name = 'POM2_FBSVM_Worker'                             # Name
        self.model = FBSVM_model(logger)                            # Model  
        self.tr_size = self.Xtr_b.shape[0]
        self.is_trained = False                                     # Flag to know if the model has been trained
    
    

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Process the received packet at worker.

        Parameters
        ----------
        packet: dictionary
            Packet received from the master.
        """
        if packet['action'] == 'SEND_PUBLIC_KEY':
            action = 'SEND_PUBLIC_KEY'
            data = {'public_key': self.public_key}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'INITIALIZE_PARAMETERS':
            self.display(self.name + ' %s: Initializing FBSVM parameters...' %self.worker_address)
            self.model.sigma = packet['data']['sigma']
            self.C = packet['data']['C']
            self.model.centroids = packet['data']['centroids']
            self.eps = packet['data']['eps']
            self.num_epochs_worker = packet['data']['num_epochs_worker']
            self.num_centroids = self.model.centroids.shape[0]

            # Computing KXC only once
            XC2 = -2 * np.dot(self.Xtr_b, self.model.centroids.T)
            XC2 += np.sum(np.multiply(self.Xtr_b, self.Xtr_b), axis=1).reshape((self.tr_size, 1))
            XC2 += np.sum(np.multiply(self.model.centroids, self.model.centroids), axis=1).reshape((1, self.num_centroids))
            KXC = np.exp(-XC2 / 2.0 /  (self.model.sigma ** 2)) 
            self.KXC = np.hstack((np.ones((self.tr_size, 1)), KXC))

            # Computing Kcc
            C2 = np.sum(self.model.centroids ** 2, axis=1).reshape((-1, 1))
            Dcc = np.abs(C2 -2 * np.dot(self.model.centroids, self.model.centroids.T) + C2.T)
            self.Kcc = np.exp(-Dcc / 2 / self.model.sigma**2)  
        
            action = 'ACK_INITIALIZE_PARAMETERS'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        
        if packet['action'] == 'LOCAL_TRAIN':
            self.display(self.name + ' %s: Updating local model' %self.worker_address)
            encrypted_weights = packet['data']['weights']
            # Unencrypt received centroids 
            weights = np.asarray(self.decrypt_list(encrypted_weights))

            Kcc_ = np.zeros((self.num_centroids+1, self.num_centroids+1))
            Kcc_[1:, 1:] = self.Kcc

            # We use the global model as starting point
            w_worker = np.copy(weights)
            Lp_old = 1e20 # Any very large value
            stop_training = False
            kiter = 1
            while not stop_training:
                w_worker_old = np.copy(w_worker)
                o = np.dot(self.KXC, w_worker)
                e = self.ytr - o
                ey = (e * self.ytr).ravel()
                a = 2 * self.C / self.eps * np.ones(self.tr_size)
                which = ey < 0
                a[which] = 0
                which = ey >= self.eps
                a[which] = 2 * self.C / ey[which]
                a = a.reshape((-1, 1))
                KTK = np.dot(self.KXC.T, self.KXC * a)
                KTy = np.dot(self.KXC.T, (self.ytr * a).reshape(self.tr_size, 1))
                w_worker_new = np.dot(np.linalg.inv(KTK + Kcc_), KTy)

                Lp = 0.5 * np.dot(np.dot(w_worker_new.T, Kcc_), w_worker_new)[0, 0] + 0.5 * np.sum(a * e * e)

                # We continue training until error increases or max iters are reached 
                if kiter == self.num_epochs_worker:
                    stop_training = True
                if Lp > Lp_old:
                    stop_training = True
                else:
                    w_worker = w_worker_new

                Lp_old = Lp
                self.display(self.name + ' %s: Iter=%d, Lp=%0.2f' %(self.worker_address, kiter, Lp))
                kiter += 1 

            encrypted_weights = np.asarray(self.encrypt_list(w_worker))
            action = 'UPDATE_MODEL'
            data = {'weights': encrypted_weights}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))  
            
            
        if packet['action'] == 'SEND_FINAL_MODEL':
            self.display(self.name + ' %s: Receiving final model' %self.worker_address)
            encrypted_weights = packet['data']['weights']
            self.model.weights = np.asarray(self.decrypt_list(encrypted_weights))
            pred_tr = self.model.predict(self.Xtr_b)
            fpr_tr, tpr_tr, _ = roc_curve(self.ytr, pred_tr)
            roc_auc_tr = auc(fpr_tr, tpr_tr)
            filter_neg = pred_tr < 0
            filter_pos = pred_tr >= 0
            pred_tr[filter_pos] = 1
            pred_tr[filter_neg] = -1
            accuracy = accuracy_score(self.ytr, pred_tr)
            self.display(self.name + ' %s: Accuracy in training set: %0.4f' %(self.worker_address, accuracy))
            self.display(self.name + ' %s: AUC in training set: %0.4f' %(self.worker_address, roc_auc_tr))
            self.model.is_trained = True
            self.is_trained = True
            self.display(self.name + ' %s: Final model stored' %self.worker_address)
            action = 'ACK_FINAL_MODEL'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

