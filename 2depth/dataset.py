# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 19:49:26 2021

@author: kaneko.naoshi
"""

import glob
import os
import random

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data

import augmentation as ag
import utils


class DepthDataset(data.Dataset):
    stats = {'mean': np.array([0.485, 0.456, 0.406], np.float32),
             'std': np.array([0.229, 0.224, 0.225], np.float32)}

    pca = {'eigval': np.array([0.2175, 0.0188, 0.0045], np.float32),
           'eigvec': np.array([[-0.5675,  0.7192,  0.4009],
                               [-0.5808, -0.0045, -0.8140],
                               [-0.5836, -0.6948,  0.4203]], np.float32)}

    train_size = {'image': (228, 304), 'depth': (114, 152)}
    test_size = (228, 304)

    def __init__(self, dataset_dir,
                 rotation_deg=5.,
                 scaling_range=0.0,
                 brightness_range=0.4,
                 contrast_range=0.4,
                 saturation_range=0.4,
                 lighting_std=0.1,
                 horizontal_flip=True,
                 train=True):

        self.rotation_deg = rotation_deg
        self.lighting_std = lighting_std
        self.horizontal_flip = horizontal_flip

        self.scaling_range = self.check_range(scaling_range)
        self.brightness_range = self.check_range(brightness_range)
        self.contrast_range = self.check_range(contrast_range)
        self.saturation_range = self.check_range(saturation_range)

        self.train = train

        self.dataset = utils.natural_sort(
            glob.glob(os.path.join(dataset_dir, '**', '*.h5'), recursive=True))
        if not self.dataset:
            raise ValueError('No .h5 files found in ' + dataset_dir)

    def __len__(self):
        return len(self.dataset)

    def check_range(self, arg):
        if np.isscalar(arg):
            arg = [1 - arg, 1 + arg]
        elif len(arg) == 2:
            arg = [arg[0], arg[1]]
        else:
            raise ValueError('Range should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: {}'.format(arg))
        return arg

    def get_images(self, idx, downsample=False):
        image_path = self.dataset[idx]

        # Load images
        with h5py.File(image_path, 'r') as f:
            image = np.array(f['rgb'])
            depth = np.array(f['depth'])

        # Transpose axis to (rows, cols, channels)
        image = image.transpose(1, 2, 0).astype(np.float32)
        depth = depth.astype(np.float32)

        # Normalize
        image = image / 255

        if downsample:
            # Resize to 1/2
            dsize = (image.shape[1] // 2, image.shape[0] // 2)
            image = cv2.resize(image, dsize,
                               interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, dsize,
                               interpolation=cv2.INTER_NEAREST)

        return image, depth

    def standardize(self, image):
        return (image - DepthDataset.stats['mean']) / DepthDataset.stats['std']

    def destandardize(self, image):
        return image * DepthDataset.stats['std'] + DepthDataset.stats['mean']

    def center_crop(self, src, size):
        y, x = ag.get_crop_indices(src, size)
        dst = src[y:y + size[0], x:x + size[1]]
        return dst

    def quantize_depth(self, depth):
        # Quantize depth into [0, 255] with maximum distance of 0.6m
        depth_u8 = np.rint(depth / 0.6 * 255).astype(np.uint8)
        # Then back to actual distance
        return (depth_u8 / 255 * 0.6).astype(np.float32)

    def apply_augmentation(self, sample):
        image = sample['image']
        depth = sample['depth']

        # Color and contrast (in random order)
        color_aug = [
            (ag.change_brightness, self.brightness_range),
            (ag.change_contrast, self.contrast_range),
            (ag.change_saturation, self.saturation_range)
        ]
        random.shuffle(color_aug)

        # Apply color augmentations
        for aug in color_aug:
            image = aug[0](image, aug[1])

        # PCA color augmentation
        image = ag.change_lighting(
            image, self.lighting_std,
            DepthDataset.pca['eigval'], DepthDataset.pca['eigvec'])

        # Compute warp matrix for rotation, scaling, and flip
        warp, scale = ag.compute_warp_matrix(
            image, self.rotation_deg,
            self.scaling_range, self.horizontal_flip)

        image_warped = image.astype(np.float32)
        depth_warped = depth.astype(np.float32)

        # Apply geometric transformations
        affine = warp[:2]
        dsize = (image.shape[1], image.shape[0])
        image_warped = cv2.warpAffine(
            image_warped, affine, dsize, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        depth_warped = cv2.warpAffine(
            depth_warped, affine, dsize, flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Divide depth values by the scale
        depth_warped = depth_warped / scale

        sample_warped = {'image': image_warped, 'depth': depth_warped}

        return sample_warped

    def __getitem__(self, idx):
        image, depth = self.get_images(idx)

        # Fill outlier pixel (farther than 1m) with its neighbor
        depth = utils.fill_invalid(depth, depth > 1.0)

        if self.train:
            sample = {'image': image, 'depth': depth}
            sample_warped = self.apply_augmentation(sample)

            image = sample_warped['image']
            depth = sample_warped['depth']

            image = self.center_crop(image, DepthDataset.train_size['image'])
            depth = self.center_crop(depth, DepthDataset.train_size['image'])

            # OpenCV has (width, height) ordering
            dsize = DepthDataset.train_size['depth'][::-1]
            depth = cv2.resize(depth, dsize,
                               interpolation=cv2.INTER_NEAREST)

            depth = self.quantize_depth(depth)
        else:
            image = self.center_crop(image, DepthDataset.test_size)
            depth = self.center_crop(depth, DepthDataset.test_size)

        # Standardize
        image = self.standardize(image)

        # To tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).clone()
        depth = torch.from_numpy(np.expand_dims(depth, axis=0)).clone()

        return {'image': image, 'depth': depth}


class AugmentedDepthDataset(DepthDataset):
    def __init__(self, dataset_dir,
                 rotation_deg=5.,
                 scaling_range=0.0,
                 brightness_range=0.4,
                 contrast_range=0.4,
                 saturation_range=0.4,
                 lighting_std=0.1,
                 horizontal_flip=True,
                 train=True):

        super(AugmentedDepthDataset, self).__init__(
            dataset_dir=dataset_dir,
            rotation_deg=rotation_deg,
            scaling_range=scaling_range,
            brightness_range=brightness_range,
            contrast_range=contrast_range,
            saturation_range=saturation_range,
            lighting_std=lighting_std,
            horizontal_flip=horizontal_flip,
            train=train
        )

    def get_images(self, idx, downsample=False):
        image_path = self.dataset[idx]

        # Load images
        with h5py.File(image_path, 'r') as f:
            image = np.array(f['rgb'])
            depth = np.array(f['depth'])
            mask = np.array(f['mask'])

        # Transpose axis to (rows, cols, channels)
        image = image.transpose(1, 2, 0).astype(np.float32)
        depth = depth.astype(np.float32)
        mask = mask.astype(np.float32)

        # Normalize
        image = image / 255

        if downsample:
            # Resize to 1/2
            dsize = (image.shape[1] // 2, image.shape[0] // 2)
            image = cv2.resize(image, dsize,
                               interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, dsize,
                               interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, dsize,
                              interpolation=cv2.INTER_NEAREST)

        return image, depth, mask

    def apply_augmentation(self, sample):
        image = sample['image']
        depth = sample['depth']
        mask = sample['mask']

        # Color and contrast (in random order)
        color_aug = [
            (ag.change_brightness, self.brightness_range),
            (ag.change_contrast, self.contrast_range),
            (ag.change_saturation, self.saturation_range)
        ]
        random.shuffle(color_aug)

        # Apply color augmentations
        for aug in color_aug:
            image = aug[0](image, aug[1])

        # PCA color augmentation
        image = ag.change_lighting(
            image, self.lighting_std,
            DepthDataset.pca['eigval'], DepthDataset.pca['eigvec'])

        # Compute warp matrix for rotation, scaling, and flip
        warp, scale = ag.compute_warp_matrix(
            image, self.rotation_deg,
            self.scaling_range, self.horizontal_flip)

        image_warped = image.astype(np.float32)
        depth_warped = depth.astype(np.float32)
        mask_warped = mask.astype(np.float32)

        # Apply geometric transformations
        affine = warp[:2]
        dsize = (image.shape[1], image.shape[0])
        image_warped = cv2.warpAffine(
            image_warped, affine, dsize, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        depth_warped = cv2.warpAffine(
            depth_warped, affine, dsize, flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        mask_warped = cv2.warpAffine(
            mask_warped, affine, dsize, flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Divide depth values by the scale
        depth_warped = depth_warped / scale

        sample_warped = {'image': image_warped,
                         'depth': depth_warped,
                         'mask': mask_warped}

        return sample_warped

    def __getitem__(self, idx):
        image, depth, mask = self.get_images(idx)

        # Fill outlier pixel (farther than 1m) with its neighbor
        depth = utils.fill_invalid(depth, depth > 1.0)

        if self.train:
            sample = {'image': image, 'depth': depth, 'mask': mask}
            sample_warped = self.apply_augmentation(sample)

            image = sample_warped['image']
            depth = sample_warped['depth']
            mask = sample_warped['mask']

            image = self.center_crop(image, DepthDataset.train_size['image'])
            depth = self.center_crop(depth, DepthDataset.train_size['image'])
            mask = self.center_crop(mask, DepthDataset.train_size['image'])

            # Apply mask to depth
            mask = mask.astype(np.bool)
            depth[mask] = 0

            # OpenCV has (width, height) ordering
            dsize = DepthDataset.train_size['depth'][::-1]
            depth = cv2.resize(depth, dsize,
                               interpolation=cv2.INTER_NEAREST)

            depth = self.quantize_depth(depth)
        else:
            image = self.center_crop(image, DepthDataset.test_size)
            depth = self.center_crop(depth, DepthDataset.test_size)
            mask = self.center_crop(mask, DepthDataset.test_size)

            # Apply mask to depth
            mask = mask.astype(np.bool)
            depth[mask] = 0

        # Standardize
        image = self.standardize(image)

        # To tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).clone()
        depth = torch.from_numpy(np.expand_dims(depth, axis=0)).clone()

        return {'image': image, 'depth': depth}
