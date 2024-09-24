# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:01:14 2021

@author: kaneko.naoshi
"""

import argparse
import os
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm

from dataset import AugmentedDepthDataset, DepthDataset
from metrics import compute_metric_scores
from train import build_model
import utils


def test_network(arch, device, test, out, model_path, parallel):
    # Set up a model to test
    if parallel:
        model = torch.nn.DataParallel(build_model(arch)).to(device)
    else:
        model = build_model(arch).to(device)

    # Load model parameters
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Set up a data loader
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    # Make output directories
    if not os.path.isdir(out):
        os.makedirs(out)

    out_img_dir = os.path.join(out, 'images')
    if not os.path.isdir(out_img_dir):
        os.makedirs(out_img_dir)

    # Prediction results
    outputs, targets = [], []

    # Predict depths from single RGB images
    with torch.no_grad():
        for i, sample in enumerate(tqdm.tqdm(test_loader)):
            # Get input images and target depth maps
            image, target = sample['image'], sample['depth']

            # Get target size
            target_size = (target.shape[2], target.shape[3])

            # Send input tensor to the specified device
            image = image.to(device)

            # Inference
            output = model(image)
            output = F.interpolate(output, size=target_size,
                                   mode='bilinear', align_corners=True)

            # Copy to the CPU and squeeze to 2D image
            output = np.squeeze(output.to('cpu').numpy().copy())
            target = np.squeeze(target.numpy().copy())

            # Get the original image
            image = np.squeeze(image.to('cpu').numpy().copy())
            image = utils.input_to_bgr(image, test)

            outputs.append(output)
            targets.append(target)

            # Colorize depth maps
            output_vis = utils.visualize_depthmap(output)
            target_vis = utils.visualize_depthmap(target)

            # Get error map
            # error_map = np.abs(output - target)

            # Visualize error map
            error_vis = utils.visualize_errormap(output, target)

            # Save visualized depth maps
            filename = os.path.join(out_img_dir, '{:04d}'.format(i))
            cv2.imwrite('{}_input.png'.format(filename), image)
            cv2.imwrite('{}_output.png'.format(filename), output_vis)
            cv2.imwrite('{}_target.png'.format(filename), target_vis)
            cv2.imwrite('{}_error.png'.format(filename), error_vis)

            # Save predicted depth and error map
            # np.save('{}_output.npy'.format(filename), output)
            # np.save('{}_error.npy'.format(filename), error_map)

    # Compute evaluation metrics
    metrics = compute_metric_scores(outputs, targets)

    # Print metrics
    print('{:<13} {}'.format('Metrics', 'Values'))
    for key, value in metrics.items():
        print('{:<13} {:.3f}'.format(key, value))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Test a depth prediction network')
    parser.add_argument('--gpu', '-g', nargs='+', default=['-1'],
                        help='GPU IDs (negative value indicates CPU)')
    parser.add_argument('--model', '-m', required=True,
                        help='Path to the model file to test')
    args = parser.parse_args()

    gpus = ','.join(args.gpu)

    # Negative -> CPU
    if '-' not in gpus:
        device = torch.device('cuda')
        parallel = True if ',' in gpus else False
    else:
        gpus = '-1'
        device = torch.device('cpu')
        parallel = False

    print('GPU: {}'.format(gpus))
    print('Model: {}'.format(args.model))
    print('')

    # Get configurations
    model_dir = os.path.basename(os.path.dirname(args.model))
    dataset = model_dir.split('_')[0]
    configs = model_dir.split('_')[1:]
    configs = {conf.split('=')[0]: conf.split('=')[1] for conf in configs}

    mask = configs['mask']

    arch = configs['arch']

    dataset_dir = os.path.join('data', dataset)
    test_dir = os.path.join(dataset_dir, 'val')

    if dataset == 'synthetic':
        if mask == 'True':
            test = AugmentedDepthDataset(test_dir, train=False)
        else:
            test = DepthDataset(test_dir, train=False)
    else:
        raise NotImplementedError

    out = os.path.dirname(args.model)

    metrics = test_network(arch, device, test,
                           out, args.model, parallel)

    # Save resuls
    filename = os.path.join(out, 'metrics.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(metrics, f)


if __name__ == '__main__':
    main()
