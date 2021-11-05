# coding: utf-8

# -*- coding: utf-8 -*-
'''

Class that estimates the Shapley values using an online algorithm

@author:  Angel Navia Vázquez
May. 2021

'''

import numpy as np
from itertools import chain, combinations
from math import factorial
import math
from MMLL.data_value.Shapley import Shapley
from sklearn.metrics import r2_score

class DVE_online():
    
    def __init__(self, method, V0, model, which_workers, Xval, yval, metric='binclass'):
        # List of models, one per iteration
        self.method = method
        self.V0 = V0        
        self.model = model
        self.model_history = [model.get_parameters()] 
        # List of updates (gradients)
        self.incs_history = []
        self.which_workers = which_workers
        self.which_workers.sort()
        self.Xval = Xval
        self.yval = yval
        self.metric = metric

        # Init DVE, equal shares
        self.DVEs = []  # The history of DVE
        self.DVE = []
        self.dve_dict = {}
        for kworker in self.which_workers:
            self.dve_dict.update({kworker: 1/len(self.which_workers)})
            self.DVE.append(1/len(self.which_workers))

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def update(self, model, values_dict, type='incs'):
        """
        Updates the internal records and the DVE

        Parameters
        ----------
        model: the updated model

        values_dict: dict of changes per worker 

        type (string): type of updates:
            - "incs": every worker provides an increment
            - "model": every worker provides a new model, to finally be averaged (as in FedAvg)
        
        Returns
        -------
        Nothing, it simply updates the internal states

        """
        self.model_history.append(model.get_parameters()) 

        if type == 'models': # values_dict contains the model parameters to average, one by every worker
            self.incs_history.append(values_dict)

            if set(self.which_workers) != set(values_dict.keys()):
                print('ERROR: the list of workers is different')
                return

            if self.method == 'song_mr':
                # Computing the models for all possible combinations
                models_dict = {}
                combination_workers = list(self.powerset(self.which_workers))[1:]
                Ncomb = len(combination_workers)

                for kcomb in range(Ncomb):
                    workers = combination_workers[kcomb]
                    selected = list(workers)
                    
                    # Creating list of increments                   
                    w_tmp = []
                    for kworker in selected:
                        w_tmp.append(self.incs_history[-1][kworker])
                    
                    # Compute predictions with combination of workers (model average)
                    preds = self.model.predict_with(w_tmp, self.Xval)

                    if self.metric == 'binclass':
                        o = (preds > 0.5).astype(int)
                        M_selected = np.mean(o == self.yval)    # accuracy

                    if self.metric == 'multiclass':
                        if len(preds) == 2:
                            preds = np.array(preds[1]).ravel()

                        M_selected = np.mean(preds == self.yval)    # accuracy

                    if self.metric == 'regression':
                        #mse = np.mean((preds.ravel() - self.yval.ravel())**2)
                        #M_selected = -mse 
                        M_selected = r2_score(self.yval.ravel(), preds.ravel())

                    #print(f'{kcomb+1} of {Ncomb}, U={M_selected}, workers =', selected)
                    w_ = np.copy(workers)
                    w_.sort()
                    key = [str(w) for w in w_]
                    key = '_'.join(key)
                    models_dict.update({key: [selected, M_selected, w_tmp]})

                shapley = Shapley(self.V0)
                PHIs = shapley.compute_scores(self.which_workers, models_dict, verbose=False)

                # Check this
                if np.sum(PHIs > 0) == 0:
                    PHIs = PHIs - 2 * np.min(PHIs)

                dv = np.copy(PHIs)
                # Warning, some PHIs may be negative, we eliminate those values in dv
                dv[dv < 0] = 0
                # We normalize dv to sum 1
                if np.sum(dv) > 0:
                    dv = dv / np.sum(dv)
            else:
                dv = None

            self.DVEs.append(dv)

            aux = np.array(self.DVEs)
            Naux = aux.shape[0]
            # weighted average, more importance to initial values
            landa = 0.5
            self.DVE = aux[0, :]
            for k in range(1, Naux):
                self.DVE = self.DVE + landa**k * aux[k, :]

            self.DVE = self.DVE / np.sum(self.DVE)

            self.dve_dict = {}
            for index, worker in enumerate(self.which_workers):
                self.dve_dict[worker] = self.DVE[index]
                
        if type == 'incs':
            self.incs_history.append(values_dict)

            if set(self.which_workers) != set(values_dict.keys()):
                print('ERROR: the list of workers is different')
                return

            if self.method == 'song_mr':
                # We take the last available model
                current_model_parameters = self.model_history[-1]

                # Computing the models for all possible combinations
                models_dict = {}
                combination_workers = list(self.powerset(self.which_workers))[1:]
                Ncomb = len(combination_workers)

                for kcomb in range(Ncomb):
                    workers = combination_workers[kcomb]
                    selected = list(workers)
                    
                    # Creating list of increments                   
                    w_tmp = [current_model_parameters]
                    for kworker in selected:
                        w_tmp.append(self.incs_history[-1][kworker])
                    # Compute predictions with combination of workers

                    preds = self.model.predict_with(w_tmp, self.Xval)

                    if self.metric == 'binclass':
                        o = (preds > 0.5).astype(int)
                        M_selected = np.mean(o == self.yval)    # accuracy

                    if self.metric == 'multiclass':
                        if len(preds) == 2:
                            preds = np.array(preds[1]).ravel()

                        M_selected = np.mean(preds == self.yval)    # accuracy

                    if self.metric == 'regression':
                        #mse = np.mean((preds.ravel() - self.yval.ravel())**2)
                        #M_selected = -mse 
                        M_selected = r2_score(self.yval.ravel(), preds.ravel())

                    #print(f'{kcomb+1} of {Ncomb}, U={M_selected}, workers =', selected)
                    w_ = np.copy(workers)
                    w_.sort()
                    key = [str(w) for w in w_]
                    key = '_'.join(key)
                    models_dict.update({key: [selected, M_selected, w_tmp]})

                shapley = Shapley(self.V0)
                PHIs = shapley.compute_scores(self.which_workers, models_dict, verbose=False)

                # Check this
                if np.sum(PHIs > 0) == 0:
                    PHIs = PHIs - 2 * np.min(PHIs)

                dv = np.copy(PHIs)
                # Warning, some PHIs may be negative, we eliminate those values in dv
                dv[dv < 0] = 0
                # We normalize dv to sum 1
                if np.sum(dv) > 0:
                    dv = dv / np.sum(dv)
            else:
                dv = None

            self.DVEs.append(dv)

            aux = np.array(self.DVEs)
            Naux = aux.shape[0]
            # weighted average, more importance to initial values
            landa = 0.5
            self.DVE = aux[0, :]
            for k in range(1, Naux):
                self.DVE = self.DVE + landa**k * aux[k, :]

            self.DVE = self.DVE / np.sum(self.DVE)

            self.dve_dict = {}
            for index, worker in enumerate(self.which_workers):
                self.dve_dict[worker] = self.DVE[index]

        if type == 'corrs':
            # we do not save the history
            #self.incs_history.append(values_dict)
            corrs_dict = values_dict['corr']
            landa = values_dict['landa']
            Fregul = values_dict['Fregul']

            if set(self.which_workers) != set(corrs_dict.keys()):
                print('ERROR: the list of workers is different')
                return

            if self.method == 'song_mr':
                # We take the last available model
                current_model_parameters = self.model_history[-1]

                # Computing the models for all possible combinations
                models_dict = {}
                combination_workers = list(self.powerset(self.which_workers))[1:]
                Ncomb = len(combination_workers)

                for kcomb in range(Ncomb):
                    workers = combination_workers[kcomb]
                    selected = list(workers)
                    
                    # Creating list of increments                   
                    w_tmp = [current_model_parameters]
                    for kworker in selected:
                        tmp_dict = {}
                        tmp_dict.update({'Rxx': corrs_dict[kworker]['Rxx']})
                        tmp_dict.update({'rxy': corrs_dict[kworker]['rxy']})
                        w_tmp.append(tmp_dict)

                    # Compute predictions with combination of workers
                    params_dict = {}
                    params_dict.update({'w_tmp': w_tmp})
                    params_dict.update({'Fregul': Fregul})
                    params_dict.update({'landa': landa})

                    preds = self.model.predict_with(params_dict, self.Xval)

                    if self.metric == 'binclass':
                        o = (preds > 0.5).astype(int)
                        M_selected = np.mean(o == self.yval)    # accuracy

                    if self.metric == 'multiclass':
                        if len(preds) == 2:
                            preds = np.array(preds[1]).ravel()

                        M_selected = np.mean(preds == self.yval)    # accuracy

                    if self.metric == 'regression':
                        #mse = np.mean((preds.ravel() - self.yval.ravel())**2)
                        #M_selected = -mse 
                        M_selected = r2_score(self.yval.ravel(), preds.ravel())

                    print(selected, M_selected)
                    w_ = np.copy(workers)
                    w_.sort()
                    key = [str(w) for w in w_]
                    key = '_'.join(key)
                    models_dict.update({key: [selected, M_selected, w_tmp]})

                shapley = Shapley(self.V0)
                PHIs = shapley.compute_scores(self.which_workers, models_dict, verbose=False)

                # Check this
                if np.sum(PHIs > 0) == 0:
                    PHIs = PHIs - 2 * np.min(PHIs)

                dv = np.copy(PHIs)
                # Warning, some PHIs may be negative, we eliminate those values in dv
                dv[dv < 0] = 0
                # We normalize dv to sum 1
                if np.sum(dv) > 0:
                    dv = dv / np.sum(dv)
            else:
                dv = None

            self.DVEs.append(dv)

            aux = np.array(self.DVEs)
            Naux = aux.shape[0]
            # weighted average, more importance to initial values
            landa = 0.5
            self.DVE = aux[0, :]
            for k in range(1, Naux):
                self.DVE = self.DVE + landa**k * aux[k, :]

            self.DVE = self.DVE / np.sum(self.DVE)

            self.dve_dict = {}
            for index, worker in enumerate(self.which_workers):
                self.dve_dict[worker] = self.DVE[index]



        return
 
    def get(self):
        return self.dve_dict