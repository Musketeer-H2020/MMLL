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

from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master, POM3_CommonML_Worker



class model():
    """
    This class contains the neural network model
    """

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




class NN_Master(POM3_CommonML_Master):
    """
    This class implements Neural nets, run at Master node. It inherits from POM3_CommonML_Master.
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

        self.name = 'POM3_NN_Master'                          # Name of the class
        self.platform = comms.name                            # Type of comms to use: either 'pycloudmessenger' or 'localflask'
        self.workers_addresses = comms.workers_ids            # Addresses of the workers
        self.Nworkers = len(self.workers_addresses)           # Number of workers
        super().__init__(self.workers_addresses, comms, logger, verbose)
        model = model_from_json(model_architecture)           # Keras model initialization
        self.display(self.name + ': Model architecture:')
        model.summary(print_fn=self.display)                  # Print the model architecture
        self.model_weights = model.get_weights()              # Weights of the initial model
        self.iter = 0                                         # Number of iterations already executed
        self.is_trained = False                               # Flag to know if the model has been trained
        self.public_keys = {}                                 # Dictionary to store public keys from all workers
        self.encrypted_Xi = {}                                # Dictionary to store encrypted Xi from all workers
        self.state_dict = {}                                  # Dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})



    def train_Master(self):
        """
        This is the main training loop, it runs the following actions until the stop condition is met:
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
        encrypted_weights = self.encrypt_list(self.model_weights, self.public_keys[self.workers_addresses[0]]) # Encrypt weights using worker 0 public key
        while self.iter != self.Nmaxiter:
            for index_worker, worker in enumerate(self.workers_addresses): 
                action = 'UPDATE_MODEL'
                data = {'model_weights': encrypted_weights}
                packet = {'action': action, 'data': data}
                
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
            packet = {'action': action, 'data': data}

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
            self.state_dict['CN'] = 'SET_PRECISION'

        if self.checkAllStates('ACK_SET_PRECISION', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SEND_PUBLIC_KEY'
            
        if self.checkAllStates('SEND_PUBLIC_KEY', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'INIT_MODEL'
            
        if self.checkAllStates('ACK_INIT_MODEL', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SET_NUM_WORKERS'
        
        if self.checkAllStates('ACK_SET_NUM_WORKERS', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'SET_LEARNING_RATE'
        
        if self.checkAllStates('ACK_SET_LEARNING_RATE', self.state_dict):
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

        Parameters
        ----------
        None
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

        # Send model to all workers
        if self.state_dict['CN'] == 'INIT_MODEL':
            action = 'INIT_MODEL'
            data = {'model_json': self.model_architecture}
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
        
        # Send the number of centroids to all workers
        if self.state_dict['CN'] == 'SET_LEARNING_RATE':
            action = 'SET_LEARNING_RATE'
            data = {'learning_rate': self.learning_rate}
            packet = {'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'WAIT'
        
        # Checking public keys received from workers
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

        sender: Strings
            Id of the sender
        """
        if packet['action'][0:3] == 'ACK':
            self.state_dict[sender] = packet['action']

        if self.state_dict['CN'] == 'WAIT_PUBLIC_KEYS':
            if packet['action'] == 'SEND_PUBLIC_KEY':
                self.public_keys[sender] = packet['data']['public_key']    # Store all public keys
                self.state_dict[sender] = packet['action']                 # Update state of sender

        if self.state_dict['CN'] == 'WAIT_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
            if packet['action'] == 'SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE':
                self.encrypted_Xi[sender] = packet['data']['encrypted_Xi'] # Store all encrypted Xi
                self.state_dict[sender] = packet['action']                 # Update state of sender
  
    
    

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
        self.master_address = master_address
        self.comms = comms
        self.logger = logger
        self.verbose = verbose
        self.Xtr_b = Xtr_b
        self.ytr = ytr

        super().__init__(logger, verbose)
        self.name = 'POM3_NN_Worker'                           # Name of the class
        self.worker_address = comms.id                         # Id identifying the current worker
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
            
        if packet['action'] == 'SET_NUM_WORKERS':
            self.display(self.name + ' %s: Storing number of workers' %self.worker_address)
            self.num_workers = packet['data']['num_workers'] # Store the number of centroids
            action = 'ACK_SET_NUM_WORKERS'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
        
        if packet['action'] == 'SET_LEARNING_RATE':
            self.display(self.name + ' %s: Storing learning rate' %self.worker_address)
            self.learning_rate = packet['data']['learning_rate'] # Store the learning rate
            action = 'ACK_SET_LEARNING_RATE'
            packet = {'action': action}
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
