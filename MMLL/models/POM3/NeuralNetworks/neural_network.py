# -*- coding: utf-8 -*-
'''
Neural Network model
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "November 2020"


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

from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master, POM3_CommonML_Worker



class model():
    """
    This class contains the neural network model
    """

    def __init__(self, model_architecture, optimizer='Adam', loss='categorical_crossentropy', metric='accuracy'):
        """
        Initializes keras model

        Parameters
        ----------
        model_architecture: JSON
            JSON containing the neural network architecture as defined by Keras (in model.to_json())
        optimizer: String
            Type of optimizer to use (must be one from https://keras.io/api/optimizers/)
        loss: String
            Type of loss to use (must be one from https://keras.io/api/losses/)
        metric: String
            Type of metric to use (must be one from https://keras.io/api/metrics/)
        """
        self.keras_model = model_from_json(model_architecture)                        # Store the model architecture
        self.keras_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])    # Compile the model



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
        return preds




class NN_Master(POM3_CommonML_Master):
    """
    This class implements Neural nets, run at Master node. It inherits from POM3_CommonML_Master.
    """
    def __init__(self, comms, logger, verbose=False, model_architecture=None, Nmaxiter=10, learning_rate=0.0001, model_averaging='True', optimizer='adam', loss='categorical_crossentropy', metric='accuracy', batch_size=32, num_epochs=1):
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

        model_averaging: Boolean
            Wether to use model averaging (True) or gradient averaging (False)

        optimizer: String
            Type of optimizer to use (should be one from https://keras.io/api/optimizers/)

        loss: String
            Type of loss to use (should be one from https://keras.io/api/losses/)

        metric: String
            Type of metric to use (should be one from https://keras.io/api/metrics/)

        batch_size: Int
            Size of the batch to use for training in each worker locally

        num_epochs: Int
            Number of epochs to train in each worker locally before sending the result to the master
        """
        self.model_architecture = model_architecture
        self.Nmaxiter = Nmaxiter
        self.learning_rate = learning_rate
        self.model_averaging = model_averaging.lower()               # Convert string to lowercase
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.batch_size = batch_size
        self.num_epochs = num_epochs           

        super().__init__(comms, logger, verbose)                     # Initialize common class for POM3
        self.name = 'POM3_NN_Master'                                 # Name of the class
        #self.Init_Environment()                                      # Send initialization messages common to all algorithms
        model = model_from_json(model_architecture)                  # Keras model initialization
        self.display(self.name + ': Model architecture:')
        model.summary(print_fn=self.display)                         # Print the model architecture
        self.model_weights = model.get_weights()                     # Weights of the initial model
        self.iter = 0                                                # Number of iterations already executed
        self.is_trained = False                                      # Flag to know if the model has been trained



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
        model = model_from_json(self.model_architecture)
        self.model_weights = model.get_weights()
        self.iter = 0
        self.is_trained = False

        self.Init_Environment() 
        self.state_dict['CN'] = 'START_TRAIN'

        while self.state_dict['CN'] != 'TRAINING_READY':
            self.Update_State_Master()
            self.TakeAction_Master()
            self.CheckNewPacket_Master()

        # Now communications should work sequentially (not sending a message to next worker until the actual one replied)
        self.display(self.name + ': Initialization ready, starting sequential communications')
        encrypted_weights = self.encrypt_list(self.model_weights, self.public_keys[self.workers_addresses[0]]) # Encrypt weights using worker 0 public key

        while self.iter != self.Nmaxiter:
            for index_worker, worker in enumerate(self.workers_addresses):
                if self.model_averaging == 'true':
                    action = 'LOCAL_TRAIN'
                else:
                    action = 'UPDATE_MODEL'
                data = {'model_weights': encrypted_weights}
                packet = {'to':'MLModel', 'action': action, 'data': data}
                
                # Send message to specific worker and wait until receiving reply
                packet = self.send_worker_and_wait_receive(packet, worker)                    
                encrypted_weights = packet['data']['model_weights']                
                encrypted_weights = self.transform_encrypted_domain_workers(encrypted_weights, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])

            self.iter += 1
            self.display(self.name + ': Iteration %d' %self.iter)
            
            if self.iter == self.Nmaxiter:
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')
                break
            
        action = 'SEND_FINAL_MODEL'
        for index_worker, worker in enumerate(self.workers_addresses):
            data = {'model_weights': encrypted_weights}
            packet = {'to':'MLModel', 'action': action, 'data': data}

            # Send message to specific worker and wait until receiving reply
            packet = self.send_worker_and_wait_receive(packet, worker)
            encrypted_weights = packet['data']['model_weights']
            encrypted_weights = self.transform_encrypted_domain_workers(encrypted_weights, worker, self.workers_addresses[(index_worker+1)%self.Nworkers])
            
        self.is_trained = True
        self.display(self.name + ': Training is done')



    def Update_State_Master(self):
        '''
        Function to control the state of the execution

        Parameters
        ----------
        None
        '''
        if self.state_dict['CN'] == 'START_TRAIN':
            self.state_dict['CN'] = 'INIT_MODEL'

        if self.checkAllStates('ACK_INIT_MODEL', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            
        if self.model_averaging == 'true':
            if self.checkAllStates('SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'COMPILE_INIT'

            if self.checkAllStates('ACK_COMPILE_INIT', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'FIT_INIT'

            if self.checkAllStates('ACK_FIT_INIT', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'TRAINING_READY'

        else:
            if self.checkAllStates('SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'SET_LEARNING_RATE'
        
            if self.checkAllStates('ACK_SET_LEARNING_RATE', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'TRAINING_READY'
    
    
    
    def TakeAction_Master(self):
        """
        Takes actions according to the state

        Parameters
        ----------
        None
        """
        to = 'MLmodel'

        # Send model to all workers
        if self.state_dict['CN'] == 'INIT_MODEL':
            action = 'INIT_MODEL'
            data = {'model_json': self.model_architecture}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'
        
        # Send the number of centroids to all workers
        if self.state_dict['CN'] == 'SET_LEARNING_RATE':
            action = 'SET_LEARNING_RATE'
            data = {'learning_rate': self.learning_rate}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'
        
        # Checking public keys received from workers
        if self.state_dict['CN'] == 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
            action = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            packet = {'to': to, 'action': action}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'

        # Asking the workers to compile the model
        if self.state_dict['CN'] == 'COMPILE_INIT':
            action = 'COMPILE_INIT'
            data = {'optimizer': self.optimizer, 'loss': self.loss, 'metric': self.metric}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'

        # Asking the workers to initialize fit parameters
        if self.state_dict['CN'] == 'FIT_INIT':
            action = 'FIT_INIT'
            data = {'batch_size': self.batch_size, 'num_epochs': self.num_epochs}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'
  
    
    

#===============================================================
#                 Worker   
#===============================================================

class NN_Worker(POM3_CommonML_Worker):
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
        self.Xtr_b = Xtr_b
        self.ytr = ytr

        super().__init__(master_address, comms, logger, verbose)      # Initialize common class for POM3
        self.name = 'POM3_NN_Worker'                                  # Name of the class
        self.num_classes = ytr.shape[1]                               # Number of classes
        self.num_features = Xtr_b.shape[1]                            # Number of features
        self.sess = tf.compat.v1.InteractiveSession()                 # Create TF session
        init = tf.compat.v1.global_variables_initializer()            # Initialize variables
        self.sess.run(init)                                           # Start TF session
        self.is_trained = False                                       # Flag to know if the model has been trained
        
        

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Take an action after receiving a packet

        Parameters
        ----------
        packet: Dictionary
            Packet received
        """        
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

            # Store number of weights for sequence generation
            num_weights = 0
            for layer in self.model.keras_model.get_weights():
                layer_list = layer.ravel().tolist()
                num_weights += len(layer_list)
            self.num_weights = num_weights

            action = 'ACK_INIT_MODEL'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
            # Review: include here the code to calculate the length of the sequence to generate (we need to know number of centroids in advance)
            # Generate random sequence for encrypting
            self.r_values = self.generate_sequence_Rvalues(self.num_weights)
            # Generate pseudo random sequence (the same for all workers)
            Xi = self.generate_sequence_Xi(self.num_weights)
            # Encrypt pseudo random sequence using sequence r_values
            encrypted_Xi = self.encrypt_flattened_list(Xi)
            action = 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE'
            data = {'encrypted_Xi': encrypted_Xi}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        if packet['action'] == 'SET_LEARNING_RATE':
            self.display(self.name + ' %s: Storing learning rate' %self.worker_address)
            self.learning_rate = packet['data']['learning_rate'] # Store the learning rate
            action = 'ACK_SET_LEARNING_RATE'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'COMPILE_INIT':
            self.display(self.name + ' %s: Initializing compiler' %self.worker_address)
            optimizer = packet['data']['optimizer']
            loss = packet['data']['loss']
            metric = packet['data']['metric']
            # Compile the model
            self.model.keras_model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
            action = 'ACK_COMPILE_INIT'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'FIT_INIT':
            self.display(self.name + ' %s: Storing batch size and number of epochs' %self.worker_address)
            self.batch_size = packet['data']['batch_size']
            self.num_epochs =  packet['data']['num_epochs']
            action = 'ACK_FIT_INIT'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'LOCAL_TRAIN':
            self.display(self.name + ' %s: Updating model locally' %self.worker_address)
            # Unencrypt received weights
            model_weights = self.decrypt_list(packet['data']['model_weights'])
            self.model.keras_model.set_weights(model_weights)
            self.model.keras_model.fit(self.Xtr_b, self.ytr, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)
            # Encrypt weights
            encrypted_weights = self.encrypt_list_rvalues(self.model.keras_model.get_weights())
            action = 'LOCAL_UPDATE'
            data = {'model_weights': encrypted_weights}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'UPDATE_MODEL':
            # Unencrypt received weights
            model_weights = self.decrypt_list(packet['data']['model_weights'])
            self.display(self.name + ' %s: Computing local gradients' %self.worker_address)
            gradients = self.get_weight_grad(model_weights, num_data=500)

            # Update weights and send them back to the CN
            for index_layer in range(len(model_weights)):
                model_weights[index_layer] = model_weights[index_layer] - (self.learning_rate / self.num_workers) * gradients[index_layer]

            # Encrypt gradients
            encrypted_weights = self.encrypt_list_rvalues(model_weights)
            action = 'UPDATE_MODEL'
            data = {'model_weights': encrypted_weights}
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

            # Encrypt again final model weights
            encrypted_weights = self.encrypt_list_rvalues(model_weights)            
            action = 'UPDATE_MODEL'
            data = {'model_weights': encrypted_weights}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))



    def get_weight_grad(self, model_weights, num_data=None):
        """ 
        Gets gradient of model for given inputs and outputs for all weights

        Parameters
        ----------
        model_weights: List of arrays
            Weights of the model
        num_data: Int
            Number of data observations used to train in a batch

        Returns
        ----------
        output_grad: List of arrays
            Calculated gradients
        """
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
