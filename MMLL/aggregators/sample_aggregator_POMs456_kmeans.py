# -*- coding: utf-8 -*-
'''
Sample aggregator object to be used by POMs 4, 5, 6 for the kmeans case
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
    def aggregate(self, model, contributions_from_workers):
        """
        Update the centroids using the contributions received from a set of workers.

        Parameters
        ----------
        model: :class:`NN_model`
            Neural Network model object.

        contributions_from_workers: list
            List containing the gradients of the different workers.
        """
        pass


class Kmeans(Aggregator):
    """
    This class implements the Kmeans algorithm. 
    """

    def __init__(self):
        """
        Create a :class:`Kmeans` instance.

        Parameters
        ----------
        None
        """

    def aggregate(self, current_model_parameters, contributions_from_workers):
        """
        Update centroids using Kmeans procedure 

        Parameters
        ----------
        current_model_parameters: model parameters

        contributions_from_workers: list of dicts
            Dicts containing the updates from the different workers.
        """
        
        # Error check
        #a = b + c

        sumX_dict = contributions_from_workers[0]
        N_dict = contributions_from_workers[1]
        workers_addresses = list(sumX_dict.keys())
        Nworkers = len(workers_addresses)

        NC, NI = current_model_parameters.shape

        newC = np.zeros((NC, NI))
        TotalP = np.zeros((NC, 1))
        
        for waddr in workers_addresses:
            for kc in range(NC):
                try:  # Some contributions may be empty
                    newC[kc, :] += sumX_dict[waddr][kc]
                    TotalP[kc] += N_dict[waddr][kc]
                except:
                    pass
        
        for kc in range(NC):
            if TotalP[kc] > 0:
                newC[kc, :] = newC[kc, :] / TotalP[kc]

        return newC

