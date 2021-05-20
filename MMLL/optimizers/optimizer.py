# -*- coding: utf-8 -*-
'''
Available optimization strategies for gradient averaging for Neural Networks in Master node.
'''

__author__ = "Roberto Díaz Morales and Marcos Fernández Díaz"
__date__ = "January 2021"


import numpy as np
from abc import ABCMeta, abstractmethod
from tensorflow.keras import backend as K



class GradientOptimizer(object, metaclass=ABCMeta):
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

        list_gradients: list
            List containing the gradients of the different workers.
        """
        pass



class SGD(GradientOptimizer):
    """
    This class implements the Stocastic Gradient Descent optimization approach, run at Master node. It inherits from :class:`GradientOptimizer`.
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


    def aggregate(self, model, list_gradients):
        """
        Aggregate the gradients received from a set of workers.

        Parameters
        ----------
        model: :class:`NN_model`
            Neural Network model object.

        list_gradients: list
            List containing the gradients of the different workers.
        """
        if self.accumulated is None and self.momentum>0:
            self.accumulated=[]
            for index_layer in range(len(model.keras_model.get_weights())):
                self.accumulated.append(np.zeros_like(model.keras_model.trainable_weights[index_layer]))

        new_weights = []

        for index_layer in range(len(model.keras_model.get_weights())):
            layer_gradients = []

            for worker in range(len(list_gradients)):
                layer_gradients.append(list_gradients[worker][index_layer])                 
            mean_weights = np.mean(layer_gradients, axis=0)

            if self.momentum>0:
                self.accumulated[index_layer] = self.accumulated[index_layer] * self.momentum - self.learning_rate*mean_weights
                if self.nesterov:
                    layer_weights = model.keras_model.trainable_weights[index_layer] + self.accumulated[index_layer] * self.momentum - self.learning_rate*mean_weights
                else:
                    layer_weights = model.keras_model.trainable_weights[index_layer] + self.accumulated[index_layer]
            else:
                layer_weights = model.keras_model.trainable_weights[index_layer] - self.learning_rate*mean_weights

            new_weights.append(layer_weights)
        model.keras_model.set_weights(new_weights)


