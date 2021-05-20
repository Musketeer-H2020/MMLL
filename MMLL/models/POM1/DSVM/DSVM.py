# -*- coding: utf-8 -*-
'''
Distributed Support Vector Machine model under POM1.
'''

__author__ = "Marcos Fernández Díaz and Angel Navia Vázquez"
__date__ = "March 2021"


import numpy as np
import time
from sklearn.metrics import accuracy_score, roc_curve, auc

from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master, POM1_CommonML_Worker
from MMLL.models.Common_to_models import Common_to_models



class DSVM_model(Common_to_models):
    """
    This class contains the Distributed SVM model.
    """

    def __init__(self, logger):
        """
        Create a :class:`DSVM_model` instance.

        Parameters
        ----------
        logger: :class:`mylogging.Logger`
            Logging object instance.
        """
        self.logger = logger
        self.is_trained = False
        self.supported_formats = ['pkl', 'onnx', 'pmml']
        self.name = 'DSVM'

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




class DSVM_Master(POM1_CommonML_Master):
    """
    This class implements DSVM, run at Master node. It inherits from :class:`POM1_CommonML_Master`.
    """

    def __init__(self, comms, logger, verbose=False, NC=None, Nmaxiter=None, tolerance=None, sigma=None, C=None, eps=None, NI=None, minvalue=None, maxvalue=None):
        """
        Create a :class:`DSVM_Master` instance.

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
            
        eps: float
            Threshold to update the a variables in the IRWLS algorithm.

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
        self.eps = eps
        self.NI = NI
        self.minvalue = minvalue
        self.maxvalue = maxvalue

        super().__init__(comms, logger, verbose)                # Initialize common class for POM1
        self.name = 'POM1_DSVM_Master'                          # Name
        self.iter = 0                                           # Number of iterations
        self.model = DSVM_model(logger)                         # DSVM model
        self.model.sigma = sigma                                # Initialize sigma
        self.num_features = self.NI                             # Initialize number of features
        self.is_trained = False                                 # Flag to know if the model is trained
   
   
 
    def Update_State_Master(self):
        '''
        Function to control the state of the execution.

        Parameters
        ----------
        None
        '''
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

        # Ask workers to launch Kmeans first
        if self.state_dict['CN'] == 'START_TRAIN':
            self.display(self.name + ': Initializing centroids ...')

            # Random initial values. Workers only need the same seed as the master to generate the centroids
            seed = time.time()
            self.seed = int((seed - int(seed)) * 10000)
            np.random.seed(seed=self.seed)
            self.model.centroids = np.random.uniform(self.minvalue, self.maxvalue, (self.num_centroids, self.num_features))

            # Computing Kcc_
            start = time.time()
            self.display(self.name + ': Computing Kcc ...')
            C2 = np.sum(self.model.centroids ** 2, axis=1).reshape((-1, 1))
            Dcc = np.abs(C2 -2 * np.dot(self.model.centroids, self.model.centroids.T) + C2.T)
            Kcc_orig = np.exp(-Dcc / 2 / (self.model.sigma**2))
            # Extended with bias
            self.Kcc_ = np.zeros((self.num_centroids + 1, self.num_centroids + 1))
            self.Kcc_[1:, 1:] = Kcc_orig
            end = time.time()
            del C2, Dcc, Kcc_orig

            if self.Xval is not None:
                # Computing KXC_val at Master
                start = time.time()
                NPval = self.Xval.shape[0]
                XC2 = -2 * np.dot(self.Xval, self.model.centroids.T)
                XC2 += np.sum(np.multiply(self.Xval, self.Xval), axis=1).reshape((-1, 1))
                XC2 += np.sum(np.multiply(self.model.centroids, self.model.centroids), axis=1).reshape((1, self.num_centroids))
                KXC_val = np.exp(-XC2 / 2.0 /  (self.model.sigma ** 2)) 
                self.KXC_val = np.hstack((np.ones((NPval, 1)), KXC_val))
                end = time.time()
                del XC2, KXC_val

            # Initialize weights
            self.model.weights = np.random.normal(0, .0001, (self.num_centroids+1, 1))
            self.state_dict['CN'] = 'INITIALIZE_PARAMETERS'

        # Asking the workers to send initialize centroids
        if self.state_dict['CN'] == 'INITIALIZE_PARAMETERS':
            action = 'INITIALIZE_PARAMETERS'
            data = {'sigma': self.model.sigma, 'C': self.C, 'seed': self.seed, 'minvalue': self.minvalue, 'maxvalue': self.maxvalue, 'num_centroids': self.num_centroids, 'eps': self.eps}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Compute average of centroids and mean distance
        if self.state_dict['CN'] == 'UPDATE_MODEL':
            old_weights = np.copy(self.model.weights)
            KTK = sum(self.C_reconstruct)
            KTy = sum(self.rx)
            new_weights = np.dot(np.linalg.inv(KTK + self.Kcc_ + 0.00001 * np.eye(self.num_centroids + 1)), KTy)

            # Linesearch for the best landa
            landas = np.arange(-1.0, 2, 0.01)
            auc_val = landas * 0

            if self.Xval is not None:
                preds_old = np.dot(self.KXC_val, old_weights)
                preds_new = np.dot(self.KXC_val, new_weights)

                for k in range(len(landas)):
                    landa = landas[k]
                    preds_val = (1 - landa) * preds_old + landa * preds_new
                    fpr_val, tpr_val, _ = roc_curve(list(self.yval), preds_val)
                    auc_val[k] = auc(fpr_val, tpr_val)

                which = np.argmax(auc_val)
                landa = landas[which]

                self.model.weights = (1 - landa) * old_weights + landa * new_weights
                self.inc_w = np.linalg.norm(self.model.weights - old_weights) / np.linalg.norm(old_weights)

            del old_weights, new_weights, landas, KTK, KTy
            self.reset()
            self.iter += 1
            self.state_dict['CN'] = 'CHECK_TERMINATION'

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            self.display(self.name + ': Iteration %d, inc_w: %0.4f' %(self.iter, self.inc_w))
            if self.Xval is not None:
                preds_val = np.dot(self.KXC_val, self.model.weights)
                fpr_val, tpr_val, _ = roc_curve(list(self.yval), preds_val)
                roc_auc_val = auc(fpr_val, tpr_val)
                filter_neg = preds_val < 0
                filter_pos = preds_val >= 0
                preds_val[filter_pos] = 1
                preds_val[filter_neg] = -1
                accuracy = accuracy_score(self.yval, preds_val)
                self.display(self.name + ': AUC on validation set: %0.4f' %roc_auc_val)
                self.display(self.name + ': Accuracy on validation set: %0.4f' %accuracy)
            if self.iter == self.Nmaxiter:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')
            elif self.inc_w < self.tolerance:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Stopping training, minimum tolerance reached!')     
            else:
                self.state_dict['CN'] = 'LOCAL_TRAIN'

        # Asking the workers to compute local gradients
        if self.state_dict['CN'] == 'LOCAL_TRAIN':
            action = 'LOCAL_TRAIN'
            data = {'weights': self.model.weights}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_weights'
        
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
        if self.state_dict['CN'] == 'wait_weights':
            if packet['action'] == 'UPDATE_MODEL':
                self.rx.append(packet['data']['rx'])
                self.state_dict[sender] = packet['action']
                
                # Distinguish the type of reconstruction for the matrices
                if 'v_triu' in packet['data']:
                    v_triu = packet['data']['v_triu']
                    C_reconstruct = np.zeros((self.num_centroids+1, self.num_centroids+1))
                    C_reconstruct[np.triu_indices(C_reconstruct.shape[0], k=0)] = v_triu
                    self.C_reconstruct.append(C_reconstruct + C_reconstruct.T)
                    del v_triu, C_reconstruct
                else:
                    Q = packet['data']['Q']
                    R = packet['data']['R']
                    self.C_reconstruct.append(np.dot(Q, R))
                    del Q, R

    
    

#===============================================================
#                 Worker   
#===============================================================

class DSVM_Worker(POM1_CommonML_Worker):
    '''
    Class implementing DSVM, run at Worker. It inherits from :class:`POM1_CommonML_Worker`.
    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`DSVM_Worker` instance.

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
        self.name = 'POM1_DSVM_Worker'                              # Name
        self.model = DSVM_model(logger)                             # Model  
        self.tr_size = self.Xtr_b.shape[0]                          # Store the number of observations
        self.num_features = self.Xtr_b.shape[1]                     # Store the number of features
        self.is_trained = False                                     # Flag to know if the model has been trained
    
    

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Process the received packet at worker.

        Parameters
        ----------
        packet: dictionary
            Packet received from the master.
        """
        if packet['action'] == 'INITIALIZE_PARAMETERS':
            self.display(self.name + ' %s: Initializing DSVM parameters...' %self.worker_address)
            self.model.sigma = packet['data']['sigma']
            self.C = packet['data']['C']
            self.eps = packet['data']['eps']
            self.num_centroids = packet['data']['num_centroids']
            seed = packet['data']['seed']
            minvalue = packet['data']['minvalue']
            maxvalue = packet['data']['maxvalue']

            # Generate same centroids as master
            np.random.seed(seed=seed)
            self.model.centroids = np.random.uniform(minvalue, maxvalue, (self.num_centroids, self.num_features))
            self.num_centroids = self.model.centroids.shape[0]

            # Computing KXC only once
            XC2 = -2 * np.dot(self.Xtr_b, self.model.centroids.T)
            XC2 += np.sum(np.multiply(self.Xtr_b, self.Xtr_b), axis=1).reshape((self.tr_size, 1))
            XC2 += np.sum(np.multiply(self.model.centroids, self.model.centroids), axis=1).reshape((1, self.num_centroids))
            KXC = np.exp(-XC2 / 2.0 /  (self.model.sigma ** 2))
            self.KXC = np.hstack((np.ones((self.tr_size, 1)), KXC))
            del XC2, KXC 
        
            action = 'ACK_INITIALIZE_PARAMETERS'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        
        if packet['action'] == 'LOCAL_TRAIN':
            self.display(self.name + ' %s: Updating local model' %self.worker_address)
            weights = packet['data']['weights']

            # We use the global model as starting point
            o = np.dot(self.KXC, weights)
            e = (self.ytr - o)
            ey = (e * self.ytr).ravel()
            a = 2 * self.C / self.eps * np.ones(self.tr_size)
            which = ey < 0
            a[which] = 0
            which = ey >= self.eps
            a[which] = 2 * self.C / ey[which]
            a = a.reshape((-1, 1))
            Cx = np.dot(self.KXC.T, self.KXC * a)
            rx = np.dot(self.KXC.T, (self.ytr * a).reshape(self.tr_size, 1))

            # Deciding what to transmit
            NSV = np.sum(a>0)
        
            if 2*self.num_centroids*NSV < (self.num_centroids+1)*(self.num_centroids+2)/2: #Then it is better to transmit the QR decomposition
                Q, R = np.linalg.qr(Cx)
                Q = Q[:, 0:NSV]
                R = R[0:NSV, :]
                # The worker sends Q, R to the Master
                data = {'rx': rx, 'Q': Q, 'R': R}
                del Q, R
            else:
                # We improve symmetry
                Cx = (Cx + Cx.T)/2
                C_triu = np.triu(Cx)
                Cdiag_div2 = np.diag(np.diag(Cx) / 2)
                X = C_triu - Cdiag_div2  # Cx = X + X.T
                # Get the upper triangular part of this matrix
                v_triu = X[np.triu_indices(X.shape[0], k=0)]          
                # The worker transmits the upper diagonals of Cxx to the Master
                data = {'rx': rx, 'v_triu': v_triu}
                del C_triu, Cdiag_div2, X, v_triu

            del weights, o, e, ey, a, which, Cx, rx
            action = 'UPDATE_MODEL'
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))  
            
            
        if packet['action'] == 'SEND_FINAL_MODEL':
            self.display(self.name + ' %s: Receiving final model' %self.worker_address)
            self.model.weights = packet['data']['weights']
            #pred_tr = self.model.predict(self.Xtr_b)
            pred_tr = np.dot(self.KXC, self.model.weights)
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

