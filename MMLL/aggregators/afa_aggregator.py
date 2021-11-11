# -*- coding: utf-8 -*-
"""
Adaptive Federated Averaging (AFA) algorithm for training Neural Networks
robust to data and model poisoning attacks.

"""

__author__ = "Alexander Matyasko"
__date__ = "November 2021"

import logging

import numpy as np
from MMLL.aggregators.aggregator import Aggregator
from MMLL.common.math_utils import cosine_similarity
from scipy.stats import beta

logger = logging.getLogger()

class AFAAveraging(Aggregator):
    """This class implements Adaptive Federated Averaging aggregation
    algorithm, which ran at Master node. It inherits from :class:`Aggregator`.

    Reference: https://arxiv.org/abs/1909.05125

    Parameters
    ----------
    slack0: float
        the initial value of the slack variable.
    slack_delta: float
        the increment of the slack variable.
    alpha0: int
        the prior alpha for Bernoulli distribution.
    beta0: int
        the prior beta for Bernoulli distribution.
    block_threshold: float
        the threshold to block bad workers.
    similarity_metric: str
        the similarity metric to use to compute the similarity of updates.

    """
    def __init__(
        self,
        slack0: float = 2.0,
        slack_delta: float = 0.5,
        alpha0: int = 3,
        beta0: int = 3,
        block_threshold: float = 0.95,
        similarity_metric: str = "cosine",
    ):
        super().__init__()
        self.slack0 = slack0
        self.slack_delta = slack_delta
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.block_threshold = block_threshold
        self.similarity_metric = similarity_metric
        # aggregator state
        self._init = False
        self._blocked_workers = []
        self._statistics = None

    def aggregate(self, model, dict_weights):
        """
        Aggregate the gradients received from a set of workers.

        Parameters
        ----------
        model: :class:`NN_model`
            Neural Network model object.

        list_gradients: list
            List containing the gradients of the different workers.
        """
        if not self._init:
            # information about each worker posterior probability (alpha_k, beta_k)
            self._statistics = {
                worker: {
                    "alpha": self.alpha0,
                    "beta": self.beta0
                }
                for worker in dict_weights
            }
            self._init = True

        def bernoulli_p(a, b):
            return a / (a + b)

        def compute_weighted_average(workers):
            new_weights = []
            for i in range(num_layers):
                layer_weights = []
                N = 0
                for worker in workers:
                    worker_weight = dict_weights[worker][i]
                    worker_stat = self._statistics[worker]
                    pk = bernoulli_p(worker_stat["alpha"], worker_stat["beta"])
                    layer_weights.append(pk * worker_weight)
                    N += pk
                average_weights = np.sum(layer_weights, axis=0) / N
                new_weights.append(average_weights)
            return new_weights

        def flatten_weights(layer_weights):
            return np.concatenate([
                np.reshape(layer_weight, (-1, ))
                for layer_weight in layer_weights
            ],
                                  axis=0)

        # remove bad workers
        dict_weights = {
            worker: worker_weights
            for worker, worker_weights in dict_weights.items()
            if worker not in set(self._blocked_workers)
        }

        num_layers = len(list(dict_weights.values())[0])
        all_workers = list(dict_weights.keys())
        good_workers = set(all_workers)
        bad_workers = set()
        new_bad_workers = {1}
        slack = self.slack0

        while len(new_bad_workers) != 0:
            new_bad_workers = set()
            # compute weighted average of good workers weights
            new_weights = compute_weighted_average(good_workers)
            # compute cosine similarity for all good workers
            new_weights_flat = flatten_weights(new_weights)
            workers_similarity = {}
            for worker, worker_weight in dict_weights.items():
                if worker not in good_workers:
                    continue
                workers_similarity[worker] = cosine_similarity(
                    new_weights_flat, flatten_weights(worker_weight))

            # filter using mean and average
            workers_similarity_arr = np.array(list(
                workers_similarity.values()))
            mean_s = np.mean(workers_similarity_arr)
            median_s = np.median(workers_similarity_arr)
            std_s = np.std(workers_similarity_arr)
            if mean_s < median_s:
                for worker in list(good_workers):
                    if workers_similarity[worker] < median_s - slack * std_s:
                        good_workers.remove(worker)
                        new_bad_workers.add(worker)
            else:
                for worker in list(good_workers):
                    if workers_similarity[worker] > median_s + slack * std_s:
                        good_workers.remove(worker)
                        new_bad_workers.add(worker)
            # update bad workers
            bad_workers |= new_bad_workers
            # increase slack to reduce false positives
            slack += self.slack_delta

        # update alpha and beta for all workers
        for worker in all_workers:
            if worker in good_workers:
                self._statistics[worker]["alpha"] += 1
            if worker in bad_workers:
                self._statistics[worker]["beta"] += 1

        # block bad workers
        for worker in all_workers:
            worker_stat = self._statistics[worker]
            th = beta.cdf(0.5, worker_stat["alpha"], worker_stat["beta"])
            if th >= self.block_threshold:
                self._blocked_workers.append(worker)
                logger.info("Blocking worker %s" % worker)

        # computer average using good workers and update model weights
        new_weights = compute_weighted_average(good_workers)
        return new_weights
