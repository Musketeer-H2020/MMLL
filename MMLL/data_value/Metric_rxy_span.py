# coding: utf-8

# -*- coding: utf-8 -*-
'''

Class that defines the metric to be used in the A priori Shapley estimation

Uses span subspace for convex combination of label proportions.

@author:  Angel Navia Vázquez
May. 2021

'''

import numpy as np

class Metric():
    
    def __init__(self):
        self.name = 'rxy_span'

    def get_stats(self, X, y):
        '''
        Computes the statistics from X, y data

        Parameters
        ----------
        X: array
            input data matrix

        y: array
            targets vector

        Returns
        -------
        stats : dict of statistical measures
        '''

        X = X.astype(float)
        # We map targets to (-1, 1)
        y = np.array(y).astype(float).reshape(-1, 1) * 2 - 1

        rxy = np.dot(X.T, y)
        stats = {self.name: rxy}
        return stats

    def get_ref_stats(self, X, y):
        '''
        Computes the statistics from X, y data

        Parameters
        ----------
        X: array
            input data matrix

        y: array
            targets vector

        Returns
        -------
        stats : dict of statistical measures
        '''

        X = X.astype(float)
        # We map targets to (-1, 1)
        y = np.array(y).astype(float).reshape(-1, 1) * 2 - 1

        which0 = (y == -1).ravel()
        which1 = (y == 1).ravel()

        X0 = X[which0, :]
        X1 = X[which1, :]
        y0 = y[which0]
        y1 = y[which1]

        rxy0 = np.dot(X0.T, y0)
        rxy1 = np.dot(X1.T, y1)

        stats = {self.name: [rxy0, rxy1]}
        return stats

    def sim_cosine(self, v1, v2):
        '''
        Computes the cosine similarity between two vectors v1 and v2

        Parameters
        ----------
        v1: array
            vector 1

        v2: array
            vector 2

        Returns
        -------
        similarity: float value in (0,1)
        '''
        d = np.dot(v1.reshape(-1, 1).T, v2.reshape(-1, 1))
        v1Tv1 = np.sqrt(np.dot(v1.reshape(-1, 1).T, v1.reshape(-1, 1)))
        if v1Tv1 > 0:
            d = d / v1Tv1
        v2Tv2 = np.sqrt(np.dot(v2.reshape(-1, 1).T, v2.reshape(-1, 1)))
        if v2Tv2 > 0:
            d = d / v2Tv2
        return d.ravel()[0]

    def combine_statistics(self, stats_list):
        """
        Computes a single statistics dictionary from a list of statistics from different workers

        Parameters
        ----------
        stats_list: list of dicts
            list of data statistics

        Returns
        -------
        stats : dict
        """
        try:

            # We receive a single stats dict not in a list
            stats_list[0]
            if len(stats_list) == 1:
                return stats_list[0]
            else:
                stats = {}
                keys = stats_list[0].keys()
                for key in keys:
                    stats.update({key: 0})
                
                for stats_ in stats_list:
                    for key in keys:
                        stats[key] += stats_[key]

                return stats                
        except:
            return stats_list

    def get_vector(self, stats):  
        """
        Computes a vector from the stats, to also be used in external dot products

        Parameters
        ----------
        stats: dict
            data statistics

        Returns
        -------
        vector : array
        """
        return stats[self.name]

    def get_vectors(self, stats):  
        """
        Computes a vector from the stats, to also be used in external dot products

        Parameters
        ----------
        stats: dict
            data statistics

        Returns
        -------
        vector : array
        """
        return [stats[self.name][0], stats[self.name][1]]

    def S(self, stats1, stats2):
        """
        Computes the Similarity between stats1 and stats2

        Parameters
        ----------
        stats1: dict
            statistics values 1

        stats2: dict
            statistics values 2

        Returns
        -------
        Similarity : float

        """
        s1 = self.combine_statistics(stats1)
        s2 = self.combine_statistics(stats2)

        v1 = self.get_vector(s1)
        v2 = self.get_vector(s2)

        return self.sim_cosine(v1, v2).ravel()[0]


    def S_span(self, stats1, stats2):
        """
        Computes the Similarity between stats1 and stats2
        stats2 is divided in 0 and 1

        Parameters
        ----------
        stats1: dict
            statistics values 1

        stats2: dict
            statistics values 2

        Returns
        -------
        Similarity : float

        """

        s1 = self.combine_statistics(stats1)
        #s2 = self.combine_statistics(stats2)
        s2 = stats2

        v1 = self.get_vector(s1)

        [v2_0, v2_1] = self.get_vectors(s2)

        # computing maximum similitud
        paso = 0.01
        alfas = np.arange(0, 1 + paso, paso)
        sims = []
        for alfa in alfas: 
            v2 = alfa * v2_0 + (1 - alfa) * v2_1
            sim_alfa = self.sim_cosine(v1, v2).ravel()[0]
            sims.append(sim_alfa)

        max_sim = np.max(sims)

        return max_sim
