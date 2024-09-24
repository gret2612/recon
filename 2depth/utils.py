# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:11:40 2020

@author: kaneko.naoshi
"""

import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd
import torch


# https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort  # NOQA
def _natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def natural_sort(l, key=_natural_sort_key):
    return sorted(l, key=key)


def set_random_seed(seed, use_cuda):
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


# https://stackoverflow.com/questions/3662361/fill-in-missing-values-with-nearest-neighbour-in-python-numpy-masked-arrays  # NOQA
def fill_invalid(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 True cells set where data value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    if invalid is None:
        invalid = np.isnan(data)

    ind = nd.distance_transform_edt(
        invalid, return_distances=False, return_indices=True)

    return data[tuple(ind)]


def visualize_depthmap(depth, cm='jet_r'):
    mask = depth != depth.min()
    masked = depth[mask]
    min_val, max_val = masked.min(), masked.max()

    vis = np.maximum(np.minimum(depth, max_val), min_val)
    vis = (vis - min_val) / (max_val - min_val)

    vis = plt.cm.get_cmap(cm)(vis)[..., :3]
    for c in range(3):
        vis[..., c][~mask] = 0.2

    vis = (vis * 255).astype(np.uint8)

    return vis


def visualize_errormap(depth1, depth2):
    error = np.abs(depth1 - depth2)

    return visualize_depthmap(error)


def input_to_bgr(image, dataset):
    dst = dataset.destandardize(image.transpose(1, 2, 0)[..., :3])
    dst = np.clip((dst * 255).astype(np.uint8), 0, 255)
    return cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
