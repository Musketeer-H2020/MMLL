# -*- coding: utf-8 -*-
"""
This file implements attacks for Musketeer Federated Learning Library
"""
"""This file implements a base class for PGD or Basic Iterative attack."""

__author__ = "Alexander Matyasko"
__date__ = "November 2021"

from __future__ import absolute_import, division, print_function

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical


class WorkerAttack(ABC):
    """Base class for all worker's attacks.
    """
    def __init__(self, seed=123):
        self._rng = np.random.RandomState(seed)
        self._tf_rng = tf.random.Generator.from_seed(seed)

    @abstractmethod
    def preprocess(self, Xtr_b, ytr):
        """Preprocess the worker's data before the start of federated learning."""
        ...

    @abstractmethod
    def process(self, model, weights, Xtr_b, ytr, epochs=1, batch_size=128):
        """Process and update the worker's model."""
        ...


def random_label_flipping(y, num_labels):
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    p = np.random.uniform(size=(y.shape[0], num_labels))
    p[np.cast[np.bool](to_categorical(y, num_labels))] = -1
    y_new = np.argmax(p, axis=-1)
    assert np.all(y_new != y)
    return y_new


class WorkerLabelFlippingAttack(WorkerAttack):
    def __init__(self, num_labels, **kwargs):
        self.num_labels = num_labels
        super().__init__(**kwargs)

    def preprocess(self, Xtr_b, ytr):
        return Xtr_b, random_label_flipping(ytr, self.num_labels)

    def process(self, model, weights, Xtr_b, ytr, epochs=1, batch_size=128):
        model.keras_model.set_weights(weights)
        model.keras_model.fit(Xtr_b,
                              ytr,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=1)


class WorkerByzantineAttack(WorkerAttack):
    def __init__(self, strength=1.0, **kwargs):
        self.strength = strength
        super().__init__(**kwargs)

    def preprocess(self, Xtr_b, ytr):
        return Xtr_b, ytr

    def process(self, model, weights, Xtr_b, ytr, epochs=1, batch_size=128):
        weights = [
            self.strength * self._rng.normal(size=w.shape) for w in weights
        ]
        model.keras_model.set_weights(weights)
