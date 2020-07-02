# -*- coding: utf-8 -*-
'''
Neural Network model
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "June 2020"


# Code to ensure reproducibility in the results
from numpy.random import seed
seed(1)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(2)

import os
import numpy as np
from keras import backend as K
from keras import losses
from keras.models import model_from_json
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Disables the warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA", doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from MMLL.models.POM2.CommonML.POM2_CommonML import POM2_CommonML_Master, POM2_CommonML_Worker



class model():

    def __init__(self, model_architecture, loss='categorical_crossentropy', metric='accuracy'):
        """
        Initializes keras model

        Parameters
        ----------
        model_architecture: JSON
            JSON containing the neural network architecture as defined by Keras (in model.to_json())
        loss: String
            Type of loss to use (should be one from https://keras.io/api/losses/)
        metric: String
            Type of metric to use (should be one from https://keras.io/api/metrics/)
        """
        self.keras_model = model_from_json(model_architecture) # Store the model architecture
        self.keras_model.compile(loss=loss, optimizer='Adam', metrics=[metric])  # Optimizer is not used (but must be one from https://keras.io/api/optimizers/)



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
        preds = self.keras_model.predict(X_b) # One-hot encoding
        preds = np.argmax(preds, axis=-1)     # Labels

        return preds




class NN_Master(POM2_CommonML_Master):
    """
    This class implements Neural nets, run at Master node. It inherits from POM2_CommonML_Master.
    """
    def __init__(self, comms, logger, verbose=False, model_architecture=None, Nmaxiter=None, learning_rate=None):
        """
        Create a :class:`NN_Master` instance.

        Parameters
        ----------
        comms: comms object instance
            Object providing communication functionalities

        logger: class:`mylogging.Logger`
            Logging object instance

        verbose: boolean
            Indicates if messages are print or not on screen

        model_architecture: JSON
            JSON containing the neural network architecture as defined by Keras (in model.to_json())

        Nmaxiter: int
            Maximum number of iterations

        learning_rate: float
            Learning rate for training
        """
        self.comms = comms    
        self.logger = logger
        self.verbose = verbose
        self.model_architecture = model_architecture
        self.Nmaxiter = Nmaxiter
        self.learning_rate = learning_rate

        self.name = 'POM2_NN_Master'                # Name
        self.platform = comms.name                  # Type of comms to use: either 'pycloudmessenger' or 'localflask'
        self.workers_addresses = comms.workers_ids  # Addresses of the workers
        super().__init__(self.workers_addresses, comms, logger, verbose)
        self.Nworkers = len(self.workers_addresses) # Nworkers
        self.reset()                                # Reset local data
        self.model = model_from_json(model_architecture)      # Keras model initialization
        self.model_weights = self.model.get_weights()
        self.display(self.name + ': Model architecture:')
        self.model.summary(print_fn=self.display)
        self.iter = 0                               # Number of iterations
        self.is_trained = False                     # Flag to know if the model has been trained
        self.state_dict = {}                        # Dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})



    def Update_State_Master(self):
        '''
        Function to control the state of the execution
        '''
        if self.state_dict['CN'] == 'START_TRAIN':
            self.state_dict['CN'] = 'SEND_PUBLIC_KEY'
            
        if self.checkAllStates('SEND_PUBLIC_KEY', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'CHECK_PUBLIC_KEYS'
            
        if self.checkAllStates('ACK_INIT_MODEL', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'COMPUTE_GRADIENTS'

        if self.checkAllStates('UPDATE_GRADIENTS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'UPDATE_MODEL'
    
    
    
    def TakeAction_Master(self):
        """
        Takes actions according to the state
        """
        # Ask workers to send public key
        if self.state_dict['CN'] == 'SEND_PUBLIC_KEY':
            action = 'SEND_PUBLIC_KEY'
            packet = {'action': action}
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
            self.state_dict['CN'] = 'INIT_MODEL'

        # Send model to all workers
        if self.state_dict['CN'] == 'INIT_MODEL':
            action = 'INIT_MODEL'
            data = {'model_json': self.model_architecture}
            packet = {'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Compute average of gradients and update model
        if self.state_dict['CN'] == 'UPDATE_MODEL':
            for index_layer in range(len(self.model_weights)):
                layer_gradients = []
                for worker in range(self.Nworkers):
                    layer_gradients.append(self.list_gradients[worker][index_layer])                 
                mean_gradients = np.mean(layer_gradients, axis=0) # Average layer gradients for all workers
                self.model_weights[index_layer] = self.model_weights[index_layer] - self.learning_rate*mean_gradients
                
            self.reset()
            self.state_dict['CN'] = 'CHECK_TERMINATION'
            self.iter += 1

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            if self.iter == self.Nmaxiter:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Iteration %d' %self.iter)
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')
            else:
                self.state_dict['CN'] = 'COMPUTE_GRADIENTS'               
                self.display(self.name + ': Iteration %d' %self.iter)

        # Asking the workers to compute local gradients
        if self.state_dict['CN'] == 'COMPUTE_GRADIENTS':
            if self.iter == 0:
                # In the first iteration encrypt the weights
                self.model_weights = self.encrypt_list(self.model_weights)
            action = 'COMPUTE_LOCAL_GRADIENTS'
            data = {'model_weights': self.model_weights}
            packet = {'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_gradients'
        
        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_FINAL_MODEL':
            action = 'SEND_FINAL_MODEL'
            data = {'model_weights': self.model_weights}
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

        if self.state_dict['CN'] == 'WAIT_PUBLIC_KEYS':
            if packet['action'] == 'SEND_PUBLIC_KEY':
                self.list_public_keys.append(packet['data']['public_key'])
                self.state_dict[sender] = packet['action']

        if self.state_dict['CN'] == 'wait_gradients':
            if packet['action'] == 'UPDATE_GRADIENTS':
                self.list_gradients.append(packet['data']['gradients'])
                self.state_dict[sender] = packet['action']
  
    
    

#===============================================================
#                 Worker   
#===============================================================

class NN_Worker(POM2_CommonML_Worker):
    '''
    Class implementing Neural nets, run at Worker

    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`NN_Worker` instance.

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

        Xtr_b: np.ndarray
            2-D numpy array containing the inputs for training

        ytr: np.ndarray
            2-D numpy array containing the labels for training
        """
        self.master_address = master_address
        self.comms = comms
        self.logger = logger
        self.verbose = verbose
        self.Xtr_b = Xtr_b
        self.ytr = ytr

        super().__init__(logger, verbose)
        self.name = 'POM2_NN_Worker'                           # Name
        self.worker_address = comms.id                         # Addresses of the workers
        self.platform = comms.name                             # Type of comms to use: either 'pycloudmessenger' or 'localflask'
        self.num_classes = ytr.shape[1]                        # Number of classes
        self.sess = tf.compat.v1.InteractiveSession()          # Create TF session
        init = tf.compat.v1.global_variables_initializer()     # Initialize variables
        self.sess.run(init)                                    # Start TF session
        self.is_trained = False                                # Flag to know if the model has been trained
        
        

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

        if packet['action'] == 'SEND_PUBLIC_KEY':
            self.public_key, self.private_key = self.generate_keypair()
            action = 'SEND_PUBLIC_KEY'
            data = {'public_key': self.public_key}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        if packet['action'] == 'INIT_MODEL':
            self.display(self.name + ' %s: Initializing local model' %self.worker_address)
            model_json = packet['data']['model_json']
            # Initialize local model
            self.current_index = 0
            self.model = model(model_json)
            self.display(self.name + ': Model architecture:')
            self.model.keras_model.summary(print_fn=self.display)
            self.label_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, self.num_classes])
            self.loss = losses.categorical_crossentropy(self.label_placeholder, self.model.keras_model.output)
            self.gradients = K.gradients(self.loss, self.model.keras_model.trainable_weights)
            action = 'ACK_INIT_MODEL'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'COMPUTE_LOCAL_GRADIENTS':
            # Unencrypt received weights
            model_weights = self.decrypt_list(packet['data']['model_weights'])
            self.display(self.name + ' %s: Computing local gradients' %self.worker_address)
            gradients = self.get_weight_grad(model_weights, num_data=500)
            # Encrypt gradients
            encrypted_gradients = self.encrypt_list(gradients)
            action = 'UPDATE_GRADIENTS'
            data = {'gradients': encrypted_gradients}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'SEND_FINAL_MODEL':            
            model_weights = self.decrypt_list(packet['data']['model_weights'])
            self.model.keras_model.set_weights(model_weights)
            self.display(self.name + ' %s: Final model stored' %self.worker_address)
            [_, accuracy] = self.model.keras_model.evaluate(self.Xtr_b, self.ytr, verbose=self.verbose)
            self.display(self.name + ' %s: Accuracy in training set: %0.4f' %(self.worker_address, accuracy))
            self.is_trained = True

            action = 'ACK_FINAL_MODEL'
            packet = {'action': action}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))



    def get_weight_grad(self, model_weights, num_data=None):
        """ Gets gradient of model for given inputs and outputs for all weights"""
        if num_data==None or num_data>self.Xtr_b.shape[0]:
            x_batch = self.Xtr_b
            y_batch = self.ytr
        else:
            data_indexes = (np.array(range(num_data)) + self.current_index) %self.Xtr_b.shape[0]
            self.current_index = (self.current_index + num_data) %self.Xtr_b.shape[0]
            x_batch = np.take(self.Xtr_b, data_indexes, axis=0) # Slice along first axis
            y_batch = np.take(self.ytr, data_indexes, axis=0)

        self.model.keras_model.set_weights(model_weights)        
        output_grad = self.sess.run(self.gradients, feed_dict={self.label_placeholder: y_batch, self.model.keras_model.input: x_batch})

        return output_grad
