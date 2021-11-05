# coding: utf-8

# -*- coding: utf-8 -*-
'''

Class that estimates the Shapley values given the list of permutations and their utilities

@author:  Angel Navia Vázquez
May. 2021

'''

import numpy as np
from itertools import chain, combinations
from math import factorial
import math

class Shapley():
    
    def __init__(self, V0):
        self.V0 = V0

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def compute_scores(self, which_workers, models, verbose=True):
        """
        Computes the phi scores

        Parameters
        ----------
        models: list of tuples
            workers combinations and their utilities 

        Returns
        -------
        phis : the Shapley scores

        """
        NW = len(which_workers)
        phis = np.zeros(NW)
        self.contribs = []

        for kselected in range(NW):
            selected = [which_workers[kselected]]
            remaining = list(set(which_workers) - set(selected))

            if verbose: 
                print('=' * 50)
            # emptyset
            w_ = np.copy(selected)
            key = [str(w) for w in w_]
            key = '_'.join(key)
            aux = models[key][1]
            if math.isnan(aux):
                aux = 0

            phi0 = factorial(NW - 1) * (aux - self.V0) 
            acum_fi = [phi0]
            if verbose: 
                print(selected, [None], phi0)
            self.contribs.append([selected, [None], phi0])

            powerset_remaining = list(self.powerset(remaining))[1:]
            L = len(powerset_remaining)
            for k in range(L):
                active_workers = list(powerset_remaining[k])
                w_ = np.copy(active_workers)
                w_.sort()
                key = [str(w) for w in w_]
                key = '_'.join(key)
                M_active_workers = models[key][1]

                active_workers_plus_selected = selected + active_workers
                w_ = np.copy(active_workers_plus_selected)
                w_.sort()
                key = [str(w) for w in w_]
                key = '_'.join(key)
                M_active_workers_plus_selected = models[key][1]

                if math.isnan(M_active_workers):
                    M_active_workers = 0

                if math.isnan(M_active_workers_plus_selected):
                    M_active_workers_plus_selected = 0

                # Contribution of selected in this subset:
                N_active = len(active_workers)        
                contrib = M_active_workers_plus_selected - M_active_workers
                contrib = contrib * factorial(N_active) * factorial(NW - 1 - N_active)
                if verbose:         
                    print(active_workers_plus_selected, active_workers, contrib)
                acum_fi.append(contrib)
                self.contribs.append([active_workers_plus_selected, active_workers, contrib])

            phis[kselected] = np.sum(acum_fi) / factorial(NW)

        return phis.ravel()

