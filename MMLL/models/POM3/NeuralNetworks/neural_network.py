# -*- coding: utf-8 -*-
'''
Neural Network model under POM3.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "February 2021"


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disables tensorflow warnings

# Code to ensure reproducibility in the results
from numpy.random import seed
seed(1)
from tensorflow.compat.v1 import set_random_seed
set_random_seed(2)

import numpy as np
import tensorflow as tf
import time

from sklearn.utils import shuffle
from MMLL.models.POM3.CommonML.POM3_CommonML import POM3_CommonML_Master, POM3_CommonML_Worker
from MMLL.models.Common_to_models import Common_to_models

RESUME = False
TO_ADV_TRAIN = True

class NN_model(Common_to_models):
    """
    This class contains the neural network model in the format defined by Keras.
    """

    def __init__(self, logger, model_architecture, optimizer='Adam', loss='categorical_crossentropy', metric='accuracy'):
        """
        Create a :class:`NN_model` instance.

        Parameters
        ----------
        logger: :class:`mylogging.Logger`
            Logging object instance.

        model_architecture: JSON
            Neural network architecture as defined by Keras (in model.to_json()).

        optimizer: string
            Type of optimizer to use (must be one from https://keras.io/api/optimizers/).

        loss: string
            Type of loss to use (must be one from https://keras.io/api/losses/).

        metric: string
            Type of metric to use (must be one from https://keras.io/api/metrics/).
        """
        self.logger = logger
        self.is_trained = False
        self.supported_formats = ['h5', 'onnx', 'Tensorflow SavedModel']
        self.name = 'NN'

        if RESUME:
            self.keras_model = tf.keras.models.load_model('results/models/tmp_worker_user_1')  # load a worker who last
                                                                                               # had global model
        else:
            self.keras_model = tf.keras.models.model_from_json(model_architecture)  # Store the model architecture
            self.keras_model.compile(optimizer=optimizer,
                                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                     metrics=[metric])  # Compile the model

        self.pgd_params = {'random_start': True,
                           'eta': 0.3,
                           'steps': 40,
                           'step_size': 0.01}

    def predict(self, X_b):
        """
        Uses the neural network model to predict new outputs given the inputs.

        Parameters
        ----------
        X_b: ndarray
            Array containing the input patterns.

        Returns
        -------
        preds: ndarray
            Array containing the predictions.
        """
        preds = self.keras_model.predict(X_b)
        return preds

    def adv_step(self, x, y, loss_function):
        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self.keras_model(x, training=False)
            loss = loss_function(y, y_pred)
        return tape.gradient(loss, x)

    def make_adversarial_example(self, x, y):
        clip_min_value = x - self.pgd_params['eta']
        clip_max_value = x + self.pgd_params['eta']

        if self.pgd_params['random_start']:
            x = x + tf.random.uniform(x.shape, minval=-self.pgd_params['eta'], maxval=self.pgd_params['eta'], dtype='float64')
            x = tf.clip_by_value(x, 0, 1)  # ensure valid pixel range

        lf = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        for _ in range(self.pgd_params['steps']):
            grads = self.adv_step(x, y, lf)

            x += self.pgd_params['step_size'] * tf.math.sign(grads)

            x = tf.clip_by_value(x, clip_min_value, clip_max_value)
            x = tf.clip_by_value(x, 0, 1)

        return x

    def adversarial_training(self, x, y, epochs, batch_size):
        loss_list = []
        acc_list = []
        adv_loss_list = []
        adv_acc_list = []
        num_of_batches = int(len(x) / batch_size)

        x, y = shuffle(x, y)

        for _ in range(epochs):
            for bnum in range(num_of_batches):

                x_batch = np.copy(x[bnum * batch_size:(bnum + 1) * batch_size])
                y_batch = np.copy(y[bnum * batch_size:(bnum + 1) * batch_size])

                start = time.time()

                loss, acc = self.keras_model.test_on_batch(x_batch, y_batch)
                loss_list.append(loss)
                acc_list.append(acc)

                adv_data = self.make_adversarial_example(x_batch, y_batch)

                adv_loss, adv_acc = self.keras_model.train_on_batch(adv_data, y_batch)
                adv_loss_list.append(adv_loss)
                adv_acc_list.append(adv_acc)
                end = time.time()
                if bnum % 10 == 0:
                    print('Batch {}. Loss {}. Acc {}. Adv Loss {}. Adv Acc {}. taking {}'.format(
                        bnum, loss, acc, adv_loss, adv_acc, end - start))
        print('Batch {}. Loss {}. Acc {}. Adv Loss {}. Adv Acc {}. taking {}'.format(bnum, loss, acc, adv_loss, adv_acc,
                                                                                     end - start))



class NN_Master(POM3_CommonML_Master):
    """
    This class implements Neural Networks, run at Master node. It inherits from :class:`POM3_CommonML_Master`.
    """
    def __init__(self, comms, logger, verbose=False, model_architecture=None, Nmaxiter=10, learning_rate=0.0001, model_averaging='True', optimizer='adam', loss='categorical_crossentropy', metric='accuracy', batch_size=32, num_epochs=1):
        """
        Create a :class:`NN_Master` instance.

        Parameters
        ----------
        comms: :class:`Comms_master`
            Object providing communication functionalities.

        logger: :class:`mylogging.Logger`
            Logging object instance.

        verbose: boolean
            Indicates whether to print messages on screen nor not.

        model_architecture: JSON
            JSON containing the neural network architecture as defined by Keras (in model.to_json()).

        Nmaxiter: int
            Maximum number of communication rounds.

        learning_rate: float
            Learning rate for training.

        model_averaging: boolean
            Whether to use model averaging (True) or gradient averaging (False).

        optimizer: string
            Type of optimizer to use (should be one from https://keras.io/api/optimizers/).

        loss: string
            Type of loss to use (should be one from https://keras.io/api/losses/).

        metric: string
            Type of metric to use (should be one from https://keras.io/api/metrics/).

        batch_size: int
            Size of the batch to use for training in each worker locally.

        num_epochs: int
            Number of epochs to train in each worker locally before sending the result to the master.
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
        #self.Init_Environment()                                     # Send initialization messages common to all algorithms
        ml_model = NN_model(logger, model_architecture, self.optimizer, self.loss, self.metric)   # Keras model initialization
        self.model_weights = ml_model.keras_model.get_weights()      # Model weights
        self.display(self.name + ': Model architecture:')
        ml_model.keras_model.summary(print_fn=self.display)
        self.model = ml_model
        self.iter = 0


        self.display(self.name + ': Model architecture:')
        self.model.keras_model.summary(print_fn=self.display)  # Print model architecture
        self.is_trained = False                                      # Flag to know if the model has been trained



    def train_Master_(self):
        """
        Main loop controlling the training of the algorithm.

        Parameters
        ----------
        None
        """        
        ml_model = NN_model(self.logger, self.model_architecture, self.optimizer, self.loss, self.metric)
        self.model_weights = ml_model.keras_model.get_weights()
        # self.iter = 0
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
        Function to control the state of the execution.

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
            
        if self.checkAllStates('SEND_ENCRYPTED_PSEUDO_RANDOM_SEQUENCE', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'COMPILE_INIT'

        if self.checkAllStates('ACK_COMPILE_INIT', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'FIT_INIT'

        if self.model_averaging == 'true':
            if self.checkAllStates('ACK_FIT_INIT', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'TRAINING_READY'

        else:
            if self.checkAllStates('ACK_FIT_INIT', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'SET_LEARNING_RATE'
        
            if self.checkAllStates('ACK_SET_LEARNING_RATE', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'TRAINING_READY'
    
    
    
    def TakeAction_Master(self):
        """
        Function to take actions according to the state.

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
    Class implementing Neural Networks, run at Worker node. It inherits from :class:`POM3_CommonML_Worker`.
    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`NN_Worker` instance.

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
        self.ytr = ytr

        super().__init__(master_address, comms, logger, verbose)      # Initialize common class for POM3
        self.name = 'POM3_NN_Worker'                                  # Name of the class
        self.is_trained = False                                       # Flag to know if the model has been trained
        
        

    def ProcessReceivedPacket_Worker(self, packet):
        """
        Process the received packet at worker.

        Parameters
        ----------
        packet: dictionary
            Packet received from the master.
        """        
        if packet['action'] == 'INIT_MODEL':
            self.display(self.name + ' %s: Initializing local model' %self.worker_address)
            model_json = packet['data']['model_json']
            self.current_index = 0
            # Initialize local model
            self.model = NN_model(self.logger, model_json)
            self.display(self.name + ': Model architecture:')
            self.model.keras_model.summary(print_fn=self.display)

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

            if RESUME:
                self.model.keras_model = tf.keras.models.load_model('results/models/tmp_worker_' + self.worker_address)
            else:
                self.model.keras_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                               optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                               metrics=[metric])

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

            if TO_ADV_TRAIN:
                self.model.adversarial_training(self.Xtr_b, self.ytr, epochs=self.num_epochs, batch_size=self.batch_size)
            else:
                self.model.keras_model.fit(self.Xtr_b, self.ytr, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)

            self.model.keras_model.save('results/models/tmp_worker_' + self.worker_address)

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
            gradients = self.get_weight_grad(model_weights, num_data=self.batch_size)

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
            self.model.is_trained = True
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
        Get gradients of the model.

        Parameters
        ----------
        model_weights: list of numpy arrays
            Actual weights of the model as returned by Keras model.get_weights().
        num_data: int 
            Size of the batch.

        Returns
        -------
        output_grad: list of numpy arrays
            Gradients of the model.
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

        with tf.GradientTape(persistent=True) as tape:
            pred_y = self.model.keras_model(x_batch, training=False) # Make prediction
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
            model_loss = self.model.keras_model.compiled_loss(y_batch, pred_y)

        output_grad = tape.gradient(model_loss, self.model.keras_model.trainable_weights)
        output_grad = [layer_grad.numpy() for layer_grad in output_grad] # Convert each layer from EagerTensor to numpy

        return output_grad

