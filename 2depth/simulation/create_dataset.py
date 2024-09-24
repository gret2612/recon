# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:54:52 2019

@author: kaneko.naoshi
"""

import argparse
import glob
import os
import re
import subprocess

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


# https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort  # NOQA
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def natural_sort(l, key=natural_sort_key):
    return sorted(l, key=key)


def visualize_depthmap(depth):
    mask = depth != depth.min()
    masked = depth[mask]
    min_val, max_val = masked.min(), masked.max()

    vis = np.maximum(np.minimum(depth, max_val), min_val)
    vis = (vis - min_val) / (max_val - min_val)

    vis = plt.cm.jet_r(vis)[..., :3]
    for c in range(3):
        vis[..., c][~mask] = 0.2

    vis = (vis * 255).astype(np.uint8)

    return vis


def visualize_normalmap(normal):
    def normalize(normal):
        l2 = np.linalg.norm(normal, axis=2)
        l2[l2 < np.finfo(np.float).eps] = 1.0

        return normal / np.expand_dims(l2, axis=2)

    vis = normalize(normal)

    # Offset and rescale values to be in 0-255
    vis = ((vis * 0.5 + 0.5) * 255).astype(np.uint8)

    return vis


def create_dataset(food_dirs, out_dir):
    for food_dir in food_dirs:
        print('Create data from {}...'.format(food_dir))

        mask_paths = natural_sort(
            glob.glob(os.path.join(food_dir, 'mask', '*.png')))
        render_paths = natural_sort(
            glob.glob(os.path.join(food_dir, 'render', '*.png')))
        depth_paths = natural_sort(
            glob.glob(os.path.join(food_dir, 'depth', '*.exr')))
        normal_paths = natural_sort(
            glob.glob(os.path.join(food_dir, 'normal', '*.exr')))

        if not (len(mask_paths) == len(render_paths) == len(depth_paths) == len(normal_paths)):
            raise ValueError('Different number of files in subdirectories')

        foodname = os.path.basename(os.path.dirname(food_dir))        
        food_out_dir = os.path.join(out_dir, foodname)
        if not os.path.isdir(food_out_dir):
            os.makedirs(food_out_dir)

        num_frames = len(mask_paths)

        for i in range(num_frames):
            # Load rendered image
            render = cv2.imread(render_paths[i], cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)

            # Load depth map and split it to be one channel image
            depth = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)[..., 0]
            depth[depth > 100] = 0  # Ignore 'too far' pixels

            # Load normal map and negate it to be in the correct camera space
            normal = cv2.imread(normal_paths[i], cv2.IMREAD_UNCHANGED) * -1

            # Load mask
            mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
            
            # Expand mask to exclude object boundaries
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask, kernel)
            mask = ~(mask.astype(np.bool))  # Make food region be 'black'
            
            # Write data to a file
            name = os.path.splitext(os.path.basename(mask_paths[i]))[0]
            filename = os.path.join(food_out_dir, '{}.h5'.format(name))
            with h5py.File(filename, 'w') as f:
                f.create_dataset('rgb', data=rgb.transpose(2, 0, 1))
                f.create_dataset('depth', data=depth)
                f.create_dataset('normal', data=normal.transpose(2, 0, 1))
                f.create_dataset('mask', data=mask)


def main():
    parser = argparse.ArgumentParser(
        description='Create virtual food depth estimation dataset')
    parser.add_argument('--foods', '-f', default='models/foods',
                        help='Directory stores source food 3d models')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--out', '-o', default='../data/synthetic',
                        help='Directory that generated dataset will be stored')
    args = parser.parse_args()

    print('Dataset will be stored in: {}'.format(args.out))
    print('')

    tmp_dir = os.path.join(args.out, 'tmp')

    # Blender simulation
    cmd = 'blender\\blender.exe'
    opt = (' --background --python blender_export.py'
           ' -- --foods {0} --seed {1} --out {2}'.format(args.foods, args.seed, tmp_dir))
    redr = ' 2>&1 | findstr /v /b "Info Fra"'
    subprocess.run(cmd + opt + redr, shell=True, check=True)

    # Training set
    train_food_dirs = glob.glob(os.path.join(tmp_dir, 'train', '*', ''))
    train_out_dir = os.path.join(args.out, 'train')
    create_dataset(train_food_dirs, train_out_dir)

    # Validation set
    val_food_dirs = glob.glob(os.path.join(tmp_dir, 'val', '*', ''))
    val_out_dir = os.path.join(args.out, 'val')
    create_dataset(val_food_dirs, val_out_dir)


if __name__ == '__main__':
    main()
