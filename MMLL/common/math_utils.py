# -*- coding: utf-8 -*-
"""
@author:  Alexander Matyasko
Nov. 2021

"""
import numpy as np


def cosine_similarity(a, b):
    """Compute the cosine similarity between two vectors."""
    a = np.squeeze(a)
    b = np.squeeze(b)
    assert a.shape[0] == b.shape[0]
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))
