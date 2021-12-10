# -*- coding: utf-8 -*-
'''
Neural Network model under POM1.
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
from MMLL.models.POM1.CommonML.POM1_CommonML import POM1_CommonML_Master, POM1_CommonML_Worker
from MMLL.models.Common_to_models import Common_to_models
from MMLL.aggregators.aggregator import SGD, ModelAveraging

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

        self.keras_model = tf.keras.models.model_from_json(model_architecture)        # Store the model architecture
        self.keras_model.compile(optimizer=optimizer,
                                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                 metrics=[metric])    # Compile the model
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

class NN_Master(POM1_CommonML_Master):
    """
    This class implements Neural Networks, run at Master node. It inherits from :class:`POM1_CommonML_Master`.
    """

    def __init__(self, comms, logger, verbose=False, model_architecture=None, Nmaxiter=10, learning_rate=0.0001, momentum=0, nesterov=False, model_averaging='True', optimizer='SGD', loss='categorical_crossentropy', metric='accuracy', batch_size=32, num_epochs=1, Tmax=None, aggregator = None):
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

        momentum: float
            Optimizer momentum.

        Nesterov: boolean
            Flag indicating if the momentum optimizer is Nesterov or not.

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

        Tmax: float
            Maximum execution time in seconds.

        aggregator: Aggregator
            Rule to aggregate gradients or models.

        """
        self.model_architecture = model_architecture
        self.Nmaxiter = Nmaxiter
        self.learning_rate = learning_rate
        self.model_averaging = model_averaging.lower()          # Convert to lowercase
        try:
            self.Tmax = float(Tmax)
        except:
            self.Tmax = None

        if self.model_averaging=='true':
            self.optimizer = optimizer
            if aggregator is not None:
                self.aggregator= aggregator
            else:
                self.aggregator=ModelAveraging()
        else:
            self.optimizer = optimizer

            if aggregator is not None:
                self.aggregator= aggregator
            elif optimizer == 'SGD' or optimizer is None:
                self.aggregator=SGD(learning_rate, momentum, nesterov)
            else:
                raise NotImplementedError("This optimizer has not been implemented for gradient descent")

        self.loss = loss
        self.metric = metric
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        super().__init__(comms, logger, verbose)                                             # Initialize common class for POM1
        self.name = 'POM1_NN_Master'                                                         # Name
        self.model = NN_model(logger, model_architecture, optimizer, self.loss, self.metric) # Keras model initialization
        self.iter = 0                                                                        # Number of iterations

        if RESUME:
            self.model.keras_model = tf.keras.models.load_model('results/models/tmp_master')

        self.display(self.name + ': Model architecture:')
        self.model.keras_model.summary(print_fn=self.display)                                # Print model architecture
        self.is_trained = False



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
            self.state_dict['CN'] = 'COMPILE_INIT'
        if self.checkAllStates('ACK_COMPILE_INIT', self.state_dict):
            for worker in self.workers_addresses:
                self.state_dict[worker] = ''
            self.state_dict['CN'] = 'FIT_INIT'

        if self.model_averaging == 'true':
            if self.checkAllStates('ACK_FIT_INIT', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'LOCAL_TRAIN'
            if self.checkAllStates('LOCAL_UPDATE', self.state_dict):
                for worker in self.workers_addresses:
                    self.state_dict[worker] = ''
                self.state_dict['CN'] = 'MODEL_AVERAGING'

        else:            
            if self.checkAllStates('ACK_FIT_INIT', self.state_dict):
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

        # Send model to all workers
        if self.state_dict['CN'] == 'INIT_MODEL':
            action = 'INIT_MODEL'
            data = {'model_json': self.model_architecture}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'
            self.t_ini = time.time()

        # Compute average of gradients and update model
        if self.state_dict['CN'] == 'UPDATE_MODEL':
            new_weights = self.aggregator.aggregate(self.model, self.dict_gradients)
            self.model.keras_model.set_weights(new_weights)        
            self.model.keras_model.save('results/models/tmp_master')
            self.reset()
            self.state_dict['CN'] = 'CHECK_TERMINATION'
            self.iter += 1

        # Compute model averaging
        if self.state_dict['CN'] == 'MODEL_AVERAGING':
            new_weights = self.aggregator.aggregate(self.model, self.dict_weights)
            self.model.keras_model.set_weights(new_weights)        
            self.model.keras_model.save('results/models/tmp_master')
            self.reset()
            self.state_dict['CN'] = 'CHECK_TERMINATION'
            self.iter += 1

        # Check for termination of the training
        if self.state_dict['CN'] == 'CHECK_TERMINATION':
            if self.Xval is not None and self.yval is not None:
                [loss, accuracy] = self.model.keras_model.evaluate(self.Xval, self.yval, verbose=self.verbose)
                self.display(self.name + ': Iteration %d, loss: %0.4f val accuracy: %0.4f' %(self.iter, loss, accuracy))
            else:
                self.display(self.name + ': Iteration %d' %self.iter)

            if self.model_averaging == 'true':
                self.state_dict['CN'] = 'LOCAL_TRAIN' 
            else:
                self.state_dict['CN'] = 'COMPUTE_GRADIENTS'  

            if self.iter == self.Nmaxiter:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Stopping training, maximum number of iterations reached!')
            
            if self.Tmax is not None and time.time() - self.t_ini > self.Tmax:
                self.state_dict['CN'] = 'SEND_FINAL_MODEL'
                self.display(self.name + ': Stopping training, maximum Time reached!')                

        # Asking the workers to compute local gradients
        if self.state_dict['CN'] == 'COMPUTE_GRADIENTS':
            action = 'COMPUTE_LOCAL_GRADIENTS'
            data = {'model_weights': self.model.keras_model.get_weights()}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_gradients'

        # Asking the workers to compile the model
        if self.state_dict['CN'] == 'COMPILE_INIT':
            action = 'COMPILE_INIT'
            data = {'optimizer': self.optimizer, 'loss': self.loss, 'metric': self.metric}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Asking the workers to initialize fit parameters
        if self.state_dict['CN'] == 'FIT_INIT':
            action = 'FIT_INIT'
            data = {'batch_size': self.batch_size, 'num_epochs': self.num_epochs}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait'

        # Asking the workers to update model with local data
        if self.state_dict['CN'] == 'LOCAL_TRAIN':
            action = 'LOCAL_TRAIN'
            data = {'model_weights': self.model.keras_model.get_weights()}
            packet = {'to': to, 'action': action, 'data': data}
            self.comms.broadcast(packet, self.workers_addresses)
            self.display(self.name + ': Sent ' + action + ' to all workers')
            self.state_dict['CN'] = 'wait_weights'
        
        # Send final model to all workers
        if self.state_dict['CN'] == 'SEND_FINAL_MODEL':
            action = 'SEND_FINAL_MODEL'
            data = {'model_weights': self.model.keras_model.get_weights()}
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
                self.dict_gradients[packet['id']]=packet['data']['gradients']
                self.state_dict[sender] = packet['action']

        if self.state_dict['CN'] == 'wait_weights':
            if packet['action'] == 'LOCAL_UPDATE':
                self.dict_weights[packet['id']]=packet['data']['weights']
                self.state_dict[sender] = packet['action']
  
    
    

#===============================================================
#                 Worker   
#===============================================================

class NN_Worker(POM1_CommonML_Worker):
    '''
    Class implementing Neural Networks, run at Worker node. It inherits from :class:`POM1_CommonML_Worker`.
    '''

    def __init__(self, master_address, comms, logger, verbose=False, Xtr_b=None, ytr=None, worker_operations = None):
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

        self.worker_operations = None

        if worker_operations is not None:
            Xtr_b, ytr = worker_operations.preprocess(Xtr_b,ytr)
            self.worker_operations = worker_operations

        self.Xtr_b = Xtr_b
        self.ytr = ytr
        self.worker_address = comms.id

        super().__init__(master_address, comms, logger, verbose)    # Initialize common class for POM1
        self.name = 'POM1_NN_Worker'                                # Name
        self.is_trained = False                                     # Flag to know if the model has been trained
        
        

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
            action = 'ACK_INIT_MODEL'
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
            self.display(self.name + ' %s: Sworker_addresstoring batch size and number of epochs' %self.worker_address)
            self.batch_size = packet['data']['batch_size']
            self.num_epochs =  packet['data']['num_epochs']
            action = 'ACK_FIT_INIT'
            packet = {'action': action}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))

        if packet['action'] == 'LOCAL_TRAIN':
            self.display(self.name + ' %s: Updating model locally' %self.worker_address)
            weights = packet['data']['model_weights']
            if self.worker_operations is not None:
                self.worker_operations.process(self.model, weights, self.Xtr_b, self.ytr, epochs=self.num_epochs, batch_size=self.batch_size)
            else:
                self.model.keras_model.set_weights(weights)
                if TO_ADV_TRAIN:
                    self.model.adversarial_training(self.Xtr_b, self.ytr, epochs=self.num_epochs, batch_size=self.batch_size)
                    self.model.keras_model.save('results/models/tmp_worker_' + self.worker_address)
                else:
                    self.model.keras_model.fit(self.Xtr_b, self.ytr, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)
            action = 'LOCAL_UPDATE'
            data = {'weights': self.model.keras_model.get_weights()}
            packet = {'action': action,'id': self.worker_address, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'COMPUTE_LOCAL_GRADIENTS':
            self.display(self.name + ' %s: Computing local gradients' %self.worker_address)
            gradients = self.get_weight_grad(packet['data']['model_weights'], num_data=self.batch_size)
            action = 'UPDATE_GRADIENTS'
            data = {'gradients': gradients}
            packet = {'action': action,'id': self.worker_address ,'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))
            
        if packet['action'] == 'SEND_FINAL_MODEL':            
            model_weights = packet['data']['model_weights']
            self.model.keras_model.set_weights(model_weights)
            self.display(self.name + ' %s: Final model stored' %self.worker_address)
            [_, accuracy] = self.model.keras_model.evaluate(self.Xtr_b, self.ytr, verbose=self.verbose)
            self.display(self.name + ' %s: Accuracy in training set: %0.4f' %(self.worker_address, accuracy))
            self.model.is_trained = True
            self.is_trained = True
            action = 'ACK_FINAL_MODEL'
            packet = {'action': action}            
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

        return output_grad
