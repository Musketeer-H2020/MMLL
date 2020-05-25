# -*- coding: utf-8 -*-
'''
Neural Network model
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "May 2019"


# Code to ensure reproducibility in the results
from numpy.random import seed
seed(1)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(2)

import numpy as np
from keras import backend as K
from keras import losses
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
# Disables the warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA", doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master, POM1_CommonML_Worker



class model():

    def __init__(self, model_architecture):
        """
        Initializes keras model

        Parameters
        ----------
        model_architecture: JSON
            JSON containing the neural network architecture as defined by Keras (in model.to_json())
        """
        self.keras_model = model_from_json(model_architecture)
        self.keras_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) # Compile the model



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
        preds = np.argmax(preds, axis=-1) # Labels

        return preds




class NN_Master(POM1_CommonML_Master):
    """
    This class implements Neural nets, run at Master node. It inherits from POM1_CommonML_Master.
    """
    def __init__(self, comms, logger, verbose=False, model_architecture=None, Nmaxiter=None, learning_rate=None, Xval_b=None, yval=None, Xtest_b=None, ytest=None):
        """
        Create a :class:`NN_Master` instance.

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

        model_architecture: JSON
            JSON containing the neural network architecture as defined by Keras (in model.to_json())

        Nmaxiter: int
            Maximum number of iterations

        learning_rate: float
            Learning rate for training

        Xval_b: np.ndarray
            Validation data inputs

        yval: np.ndarray
            Validation data labels

        Xtest_b: np.ndarray
            Test data inputs

        ytest: np.ndarray
            Test data inputs
        """
        #super().__init__(workers_addresses, platform, comms, logger, verbose)
        self.name = 'NN_Master'                     # Name

        self.comms = comms    
        self.logger = logger
        self.verbose = verbose
        self.model_architecture = model_architecture
        self.Nmaxiter = Nmaxiter
        self.learning_rate = learning_rate
        self.Xval_b = Xval_b
        self.yval = yval
        self.Xtest_b = Xtest_b
        self.ytest = ytest

        self.platform = comms.name                  # Type of comms to use: either 'pycloudmessenger' or 'localflask'
        self.workers_addresses = comms.workers_ids  # Addresses of the workers
        self.Nworkers = len(self.workers_addresses) # Nworkers
        self.reset()                                # Reset local data
        self.model = model(model_architecture)      # Keras model initialization
        self.display(self.name + ': Model architecture:')
        self.model.keras_model.summary(print_fn=self.display)
        self.iter = 0                               # Number of iterations
        self.is_trained = False                     # Flag to know if the model has been trained
        self.state_dict = {}                        # Dictionary storing the execution state
        for worker in self.workers_addresses:
            self.state_dict.update({worker: ''})

        '''
        # Data dimension checking
        input_model_shape = self.model.keras_model.input.shape
        Xval_input_shape = self.Xval_b.shape[1:] # Discard the number of observations
        Xtest_input_shape = self.Xtest_b.shape[1:]
        print('input_model_shape: ', input_model_shape)
        print('Xval_input_shape: ', Xval_input_shape)
        print('Xtest_input_shape: ', Xtest_input_shape)
        
        if Xval_input_shape!=input_model_shape:
            raise Exception('Shape of input validation data %s do not match input model architecture %s' %(Xval_input_shape, input_model_shape))
        elif Xtest_input_shape!=input_model_shape:
            raise Exception('Shape of input validation data %s do not match input model architecture %s' %(Xtest_input_shape, input_model_shape))'''



    def Update_State_Master(self):
        '''
        We update the state of execution.
        We control from here the data flow of the training process
        ** By now there is only one implemented option: direct transmission **

        This code needs some improvement...
        '''

        if self.state_dict['CN'] == 'START_TRAIN':
            self.state_dict['CN'] = 'INIT_MODEL'
            
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
            for index_layer in range(len(self.model.keras_model.get_weights())):
                layer_gradients = []
                for worker in range(self.Nworkers):
                    layer_gradients.append(self.list_gradients[worker][index_layer])                 
                mean_weights = np.mean(layer_gradients, axis=0) # Average layer gradients for all workers
                K.set_value(self.model.keras_model.trainable_weights[index_layer], K.get_value(self.model.keras_model.trainable_weights[index_layer]) - self.learning_rate*mean_weights) # Update model weights
                
            self.reset()
            self.state_dict['CN'] = 'CHECK_TERMINATION'
            self.iter += 1

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            if self.iter == self.Nmaxiter:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                [_, accuracy] = self.model.keras_model.evaluate(self.Xtest_b, self.ytest, verbose=self.verbose)
                self.display(self.name + ': Iteration %d, test accuracy: %0.4f' %(self.iter, accuracy))
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')
            else:
                self.state_dict['CN'] = 'COMPUTE_GRADIENTS'               
                [loss, accuracy] = self.model.keras_model.evaluate(self.Xval_b, self.yval, verbose=self.verbose)
                self.display(self.name + ': Iteration %d, loss: %0.4f, accuracy: %0.4f' %(self.iter, loss, accuracy))

        # Asking the workers to compute local gradients
        if self.state_dict['CN'] == 'COMPUTE_GRADIENTS':
            action = 'COMPUTE_LOCAL_GRADIENTS'
            data = {'model_weights': self.model.keras_model.get_weights()}
            packet = {'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_gradients'
        
        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_FINAL_MODEL':
            action = 'SEND_FINAL_MODEL'
            data = {'model_weights': self.model.keras_model.get_weights()}
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

        if self.state_dict['CN'] == 'wait_gradients':
            if packet['action'] == 'UPDATE_GRADIENTS':
                self.list_gradients.append(packet['data']['gradients'])
                self.state_dict[sender] = packet['action']
  
    
    

#===============================================================
#                 Worker   
#===============================================================

class NN_Worker(POM1_CommonML_Worker):
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

        Xtr_b: np.ndarray
            2-D numpy array containing the inputs for training

        ytr: np.ndarray
            2-D numpy array containing the labels for training
        """
        self.name = 'POM1_NN_Worker'                           # Name

        self.master_address = master_address
        self.comms = comms
        self.logger = logger
        self.verbose = verbose
        self.Xtr_b = Xtr_b
        self.ytr = ytr

        self.worker_address = comms.id
        self.platform = comms.name
        self.num_classes = ytr.shape[1]
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
            self.display(self.name + ' %s: Computing local gradients' %self.worker_address)
            gradients = self.get_weight_grad(packet['data']['model_weights'], num_data=500)
            action = 'UPDATE_GRADIENTS'
            data = {'gradients': gradients}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'SEND_FINAL_MODEL':            
            model_weights = packet['data']['model_weights']
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
