# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
"""
This file implements attacks for Musketeer Federated Learning Library
"""
"""This file implements a base class for PGD or Basic Iterative attack."""

__author__ = "Alexander Matyasko"
__date__ = "November 2021"

from abc import ABC, abstractmethod

import numpy as np
import copy
import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.utils import to_categorical


def rebuild_model(model,
                  optimizer=None,
                  loss=None,
                  metrics=['accuracy'],
                  ref_model=None):
    if optimizer is None:
        assert ref_model is not None
        optimizer = ref_model.optimizer
    if loss is None:
        assert ref_model is not None
        loss = ref_model.loss
    model.build(model.input_shape)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def assign_layer_regularizer(layer, variable, regularizer):
    name_in_scope = variable.name[:variable.name.find(':')]
    layer._handle_weight_regularization(name_in_scope, variable, regularizer)


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
    def __init__(self, num_labels, fraction=1.0, target_class=None, **kwargs):
        self.num_labels = num_labels
        self.fraction = fraction
        self.target_class = target_class
        super().__init__(**kwargs)

    def preprocess(self, Xtr_b, ytr):
        indices = np.arange(ytr.shape[0])
        self._rng.shuffle(indices)
        attack_indices = indices[:int(ytr.shape[0] * self.fraction)]
        if self.target_class is not None:
            ytr[attack_indices] = to_categorical(
                self.target_class * np.ones(len(attack_indices)),
                self.num_labels)
        else:
            ytr[attack_indices] = to_categorical(
                random_label_flipping(ytr[attack_indices].argmax(-1),
                                      self.num_labels))
        return Xtr_b, ytr

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


@tf.keras.utils.register_keras_serializable(package='MMLL',
                                            name='StealthyL2Regularizer')
class StealthyL2Regularizer(Regularizer):
    def __init__(self, W0, ρ=1e-4):
        self.W0 = W0
        self.ρ = ρ

    def __call__(self, W):
        return self.ρ * tf.reduce_sum(tf.math.square(W - self.W0))

    def get_config(self):
        return {'W0': self.W0, 'ρ': float(self.ρ)}


def max_crossentropy(y_true, y_pred):
    return -tf.losses.categorical_crossentropy(y_true, y_pred)


class WorkerStealthyAttack(WorkerAttack):
    def __init__(self, ρ=1e-4, max_loss=1e3, **kwargs):
        self.ρ = ρ
        self.max_loss = max_loss
        super().__init__(**kwargs)
        # private fields
        self._init = False

    def preprocess(self, Xtr_b, ytr):
        return Xtr_b, ytr

    def process(self, model, weights, Xtr_b, ytr, epochs=1, batch_size=128):
        model.keras_model.set_weights(weights)
        model.keras_model.fit(Xtr_b,
                              ytr,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=1)
        benign_weights = model.keras_model.get_weights()

        def stealthy_attack_loss(y_true, y_pred):
            nll = tf.minimum(
                tf.losses.categorical_crossentropy(y_true, y_pred),
                self.max_loss)
            stealthy = sum([
                tf.reduce_sum((w0 - w1)**2)
                for w0, w1 in zip(benign_weights, model.keras_model.weights)
            ])
            return -self.ρ * nll + stealthy

        opt = model.keras_model.optimizer
        model.keras_model.compile(optimizer=opt,
                                  loss=stealthy_attack_loss,
                                  metrics=['accuracy'])

        model.keras_model.set_weights(weights)
        model.keras_model.fit(Xtr_b,
                              ytr,
                              epochs=epochs,
                              batch_size=batch_size,
                              verbose=1)
