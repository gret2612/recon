# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:46:15 2019

@author: kaneko.naoshi
"""

import cv2
import numpy as np

# These augmentation scripts are partially adopted from
# https://github.com/fangchangma/sparse-to-dense/blob/master/datasets/transforms.lua


def to_grayscale(image, color='rgb'):
    dst = np.zeros(image.shape, dtype=np.float32)

    if color == 'rgb':
        r, g, b = cv2.split(image)
    elif color == 'bgr':
        b, g, r = cv2.split(image)
    else:
        raise ValueError('Unsupported color code: {}'.format(color))

    dst[..., 0] = 0.299 * r + 0.587 * g + 0.114 * b
    dst[..., 1] = dst[..., 0].copy()
    dst[..., 2] = dst[..., 0].copy()

    return dst


def change_brightness(image, brightness_range):
    dst = image.astype(np.float32)

    # Random brightness change
    if brightness_range[0] == 1 and brightness_range[1] == 1:
        factor = 1
    else:
        factor = np.random.uniform(brightness_range[0], brightness_range[1])

    dst = factor * dst

    return dst


def change_contrast(image, contrast_range, color='rgb'):
    dst = image.astype(np.float32)

    # Random contrast change
    if contrast_range[0] == 1 and contrast_range[1] == 1:
        factor = 1
    else:
        factor = np.random.uniform(contrast_range[0], contrast_range[1])

    gray = to_grayscale(image, color)
    mean = np.full(gray.shape, gray.mean(), np.float32)

    dst = factor * dst + (1 - factor) * mean

    return dst


def change_saturation(image, saturation_range, color='rgb'):
    dst = image.astype(np.float32)

    # Random saturation change
    if saturation_range[0] == 1 and saturation_range[1] == 1:
        factor = 1
    else:
        factor = np.random.uniform(saturation_range[0], saturation_range[1])

    gray = to_grayscale(image, color)

    dst = factor * dst + (1 - factor) * gray

    return dst


def change_lighting(image, alpha_std, eigen_values, eigen_vecs):
    if alpha_std < np.finfo(np.float32).eps:
        return image

    dst = image.astype(np.float32)

    alpha = np.random.normal(0.0, alpha_std, size=3)
    delta = np.dot(eigen_vecs, alpha * eigen_values)

    dst = dst + delta

    return dst


def compute_warp_matrix(image, rotation_deg, scaling_range, horizontal_flip):
    height, width = image.shape[0], image.shape[1]

    trans_to_origin = np.identity(3, np.float32)
    trans_to_origin[0, 2] = width // 2
    trans_to_origin[1, 2] = height // 2

    # Random rotation
    if rotation_deg:
        rotation = np.deg2rad(
            np.random.uniform(-rotation_deg, rotation_deg))
    else:
        rotation = 0.0

    # Random scaling
    if scaling_range[0] == 1 and scaling_range[1] == 1:
        scale = 1
    else:
        scale = np.random.uniform(scaling_range[0],
                                  scaling_range[1])

    # Horizontal flipping
    if horizontal_flip:
        flip = np.random.randint(0, 1)
    else:
        flip = 0

    affine_warp = np.identity(3, np.float32)

    a = scale * np.cos(rotation)
    b = scale * np.sin(rotation)

    if flip:
        affine_warp[0, 0] = -a
    else:
        affine_warp[0, 0] = a

    affine_warp[0, 1] = -b
    affine_warp[1, 0] = b
    affine_warp[1, 1] = a

    trans_to_center = np.identity(3, np.float32)
    trans_to_center[0, 2] = -(width / 2)
    trans_to_center[1, 2] = -(height / 2)

    warp = np.dot(trans_to_origin, affine_warp)
    warp = np.dot(warp, trans_to_center)

    return warp, scale


def get_crop_indices(image, target_size, random=False):
    if image.shape[0] < target_size[0] or image.shape[1] < target_size[1]:
        raise ValueError('image shape {} is smaller than target_size {}'
                         .format(image.shape[:2], target_size))

    if random:
        # Random crop
        range_y = image.shape[0] - target_size[0]
        range_x = image.shape[1] - target_size[1]

        y = np.random.randint(0, range_y + 1)
        x = np.random.randint(0, range_x + 1)
    else:
        # Center crop
        center_y = image.shape[0] // 2
        center_x = image.shape[1] // 2

        y = center_y - (target_size[0] // 2)
        x = center_x - (target_size[1] // 2)

    return y, x
