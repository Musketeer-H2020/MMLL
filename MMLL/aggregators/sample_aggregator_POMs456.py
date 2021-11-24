# -*- coding: utf-8 -*-
'''
Sample aggregator object to be used by POMs 4, 5, 6.
'''

__author__ = "Angel Navia Vázquez"
__date__ = "November 2021"


import numpy as np
from abc import ABCMeta, abstractmethod
import copy

class Aggregator(object, metaclass=ABCMeta):
    """
    This class implements the different gradient optimizers, run at Master node.
    """
 
    @abstractmethod
    def aggregate(self, model, list_gradients):
        """
        Aggregate the gradients received from a set of workers.

        Parameters
        ----------
        model: :class:`NN_model`
            Neural Network model object.

        list_gradients: dict
            Dictionary containing the gradients of the different workers.
        """
        pass


class SGD(Aggregator):
    """
    This class implements the Stocastic Gradient Descent optimization approach, run at Master node. 
    """

    def __init__(self, learning_rate, momentum=0, nesterov=False):
        """
        Create a :class:`SGD` instance.

        Parameters
        ----------
        learning_rate: float
            Learning rate for training.

        momentum: float
            Optimizer momentum.

        nesterov: boolean
            Flag indicating if the momentum optimizer is Nesterov or not.
        """
        self.nesterov = nesterov
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.accumulated = None


    def aggregate(self, current_model_parameters, gradients_workers_dict):
        """
        Aggregate the gradients received from a set of workers 

        Parameters
        ----------
        current_model_parameters: model parameters

        gradients_workers_dict: dict
            Dict containing the gradients from the different workers.
        """
        
        # Error check
        #a = b + c

        # Checking if model is binary or multiclass
        workers_addresses = list(gradients_workers_dict.keys())
        Nworkers = len(workers_addresses)

        if isinstance(gradients_workers_dict[workers_addresses[0]], dict):  # MULTICLASS

            classes = list(gradients_workers_dict[workers_addresses[0]].keys())
            updated_model = copy.deepcopy(current_model_parameters)

            for cla in classes:
                for waddr in workers_addresses:
                    updated_model[cla] -= self.learning_rate * gradients_workers_dict[waddr][cla]

            # Error check
            #updated_model = np.array([1, 2])
            #updated_model = {'0': [1, 2]}
            #updated_model = {'Iris-setosa': [1, 2], 'Iris-versicolor': [1, 2], 'Iris-virginica': [1, 2]}

        else:  # BINARY
            # Updating binary model
            grad_acum = np.zeros(current_model_parameters.shape)
            for waddr in workers_addresses:
                grad_acum += gradients_workers_dict[waddr]

            # Simple gradient update
            updated_model = current_model_parameters - self.learning_rate * grad_acum      
            # Error check
            #updated_model = np.array([1, 2])

        return updated_model

