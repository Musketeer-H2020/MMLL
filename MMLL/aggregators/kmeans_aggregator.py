# -*- coding: utf-8 -*-
'''
K means aggregation.
'''

__author__ = "Roberto Díaz Morales"
__date__ = "September 2021"


import numpy as np
from abc import ABCMeta, abstractmethod
from tensorflow.keras import backend as K




class Kmeans_Aggregator(object, metaclass=ABCMeta):
    """
    This class implements an aggregator for Kmeans algorithm.
    """
    initialization = "Default"



    @abstractmethod
    def get_initialization_method(self):
        """
        Returns the centroid initialization method that must be sent to the workers.

        """
        pass

    @abstractmethod
    def initial_aggregate(self, dict_centroids):
        """
        Aggregates the centroids sent at the initialization (no information about counts).

        Parameters
        ----------
        dict_centroids: dict
            Dict containing the centroids initialized by the workers.
        """
        pass


    @abstractmethod
    def aggregate(self, dict_centroids):
        """
        Aggregate the gradients received from a set of workers in every iteration of the Kmeans algorithm.

        Parameters
        ----------

        dict_centroids: dict

            Dict containing the centroids, counts and mean_distance of the different workers.
        """
        pass

 

class Default_Kmeans(Kmeans_Aggregator):
    """
    This class implements the default aggregation method for Kmeans
    """

    def __init__(self, num_centroids):
        """
        Create a :class:`Default_Kmeans` instance.

        Parameters
        ----------
        num_centroids: int
            Number of centroids for kmeans.

        num_features: int
            Number of features in input data.

        """
        self.initialization = "Naive"
        self.num_centroids = num_centroids

    def get_initialization_method(self):
        """
        Returns the centroid initialization method that must be sent to the workers.

        """
        return self.initialization

    def initial_aggregate(self, dict_centroids):
        """
        Aggregates the centroids sent at the initialization (no information about counts).

        Parameters
        ----------
        dict_centroids: dict
            Dict containing the centroids initialized by the workers.
        """

        list_centroids = []

        for key in dict_centroids:
            list_centroids.append(dict_centroids[key]["centroids"])

        list_centroids = np.array(list_centroids)
        centroids = np.sum(list_centroids, axis=0) / len(dict_centroids.keys())
        self.centroids = centroids
        return centroids


    def aggregate(self, dict_centroids):
        """
        Aggregate the centroids received from a set of workers.

        Parameters
        ----------

        dict_centroids: dict
            Dict containing diccionaries with centroids, counts and mean distance.
        """
        list_centroids = []
        list_counts = []
        list_dists = []

        for key in dict_centroids:
            list_centroids.append(dict_centroids[key]["centroids"])
            list_counts.append(dict_centroids[key]["counts"])
            list_dists.append(dict_centroids[key]["mean_dist"])
        
        list_centroids = np.array(list_centroids) # Array of shape (num_dons x num_centroids x num_features)
        list_counts = np.array(list_counts) # Array of shape (num_dons x num_centroids)
        list_dists = np.array(list_dists) # Array of shape (num_dons x 1)
            
        # Average all mean distances received from each DON according to total number of observations per DON with respect to the total 
        # observations including all DONs
        new_mean_dist = np.dot(list_dists.T, np.sum(list_counts, axis=1)) / np.sum(list_counts[:,:])

        # Average centroids taking into account the number of observations of the training set in each DON with respect to the total
        # including the training observations of all DONs
        if np.all(np.sum(list_counts, axis=0)): # If all centroids have at least one observation in one of the DONs
            centroids = np.sum((list_centroids.T * (np.ones(list_counts.shape) / np.sum(list_counts, axis=0)).T).T, axis=0) # Shape (num_centroids x num_features)
        else: # Modify only non-empty centroids
            centroids = self.centroids.copy()
            for i in range(self.num_centroids):
                if np.sum(list_counts[:,i])>0:
                    centroids[i,:] = np.zeros_like(list_centroids[0, i])
                    for kdon in range(len(dict_centroids.keys())):
                        centroids[i,:] = centroids[i,:]+list_centroids[kdon,i,:]/np.sum(list_counts[:,i])

        self.centroids = centroids
        return centroids, new_mean_dist


