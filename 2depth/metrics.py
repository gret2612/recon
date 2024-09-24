# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:27:52 2019

@author: kaneko.naoshi
"""

import numpy as np


def absolute_relative_diff(diff, target, valid, num_pixels):
    rel = (diff / np.maximum(target, np.finfo(np.float32).eps))
    rel[~valid] = 0
    return np.sum(rel) / num_pixels


def squared_relative_diff(diff, target, valid, num_pixels):
    rel = (np.power(diff, 2) / np.maximum(target, np.finfo(np.float32).eps))
    rel[~valid] = 0
    return np.sum(rel) / num_pixels


def log10_error(output, target, valid, num_pixels):
    y = np.maximum(output, np.finfo(np.float32).eps)
    t = np.maximum(target, np.finfo(np.float32).eps)
    diff = np.abs(np.log10(y) - np.log10(t))
    diff[~valid] = 0
    return np.sum(diff) / num_pixels


def thresholded_accuracy(output, target, valid, num_pixels, threshold):
    y = np.maximum(output, np.finfo(np.float32).eps)
    t = np.maximum(target, np.finfo(np.float32).eps)
    ratio = np.maximum(y / t, t / y)
    ratio[~valid] = np.finfo(ratio.dtype).max
    return np.sum(ratio <= threshold) / num_pixels


def compute_metric_scores(pred_depths, true_depths):
    if len(pred_depths) != len(true_depths):
        raise ValueError(
            'Different number of depth maps: pred_depths contains {} '
            'while true_depths contains {}'
            .format(len(pred_depths), len(true_depths)))

    output = np.array(pred_depths)
    target = np.array(true_depths)

    # Check validity
    valid = np.bitwise_and(np.isfinite(target), target > 0)
    num_pixels = np.count_nonzero(valid)

    # Set invalid pixels to zero
    output[~valid] = 0
    target[~valid] = 0

    # Difference between predicted and GT depths
    diff = np.abs(output - target)

    # Mean Squared Error
    mse = np.sum(np.power(diff, 2)) / num_pixels

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.sum(diff) / num_pixels

    # Mean Absolute Relative Error
    abs_rel = absolute_relative_diff(diff, target, valid, num_pixels)

    # Mean Squared Relative Error
    sqr_rel = squared_relative_diff(diff, target, valid, num_pixels)

    # log10 Error
    log10 = log10_error(output, target, valid, num_pixels)

    # Thresholded accuracy
    th1 = thresholded_accuracy(output, target, valid, num_pixels, 1.25)
    th2 = thresholded_accuracy(output, target, valid, num_pixels, 1.25 ** 2)
    th3 = thresholded_accuracy(output, target, valid, num_pixels, 1.25 ** 3)

    result = {
        'thresh1': th1, 'thresh2': th2, 'thresh3': th3,
        'abs_rel': abs_rel, 'sqr_rel': sqr_rel,
        'mae': mae, 'rmse': rmse, 'log10': log10
    }

    return result
