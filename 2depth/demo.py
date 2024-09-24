# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:04:19 2021

@author: kaneko.naoshi
"""

import argparse
import glob
import itertools
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import tqdm

from augmentation import get_crop_indices
from train import build_model
import utils


class DemoDataset(data.Dataset):
    stats = {'mean': np.array([0.485, 0.456, 0.406], np.float32),
             'std': np.array([0.229, 0.224, 0.225], np.float32)}

    input_size = (228, 304)

    def __init__(self, input_dir):
        exts = ['*.png', '*.jpg']
        self.dataset = utils.natural_sort(list(itertools.chain.from_iterable(
            [glob.glob(os.path.join(input_dir, ext)) for ext in exts])))
        if not self.dataset:
            raise ValueError('No image files found in ' + input_dir)

    def __len__(self):
        return len(self.dataset)

    def get_image(self, idx, downsample=False):
        image_path = self.dataset[idx]

        # Load image and convert it to RGB
        image = cv2.imread(image_path).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize
        image = image / 255

        if downsample:
            # Resize to 1/2
            dsize = (image.shape[1] // 2, image.shape[0] // 2)
            image = cv2.resize(image, dsize,
                               interpolation=cv2.INTER_LINEAR)

        return image, image_path

    def standardize(self, image):
        return (image - DemoDataset.stats['mean']) / DemoDataset.stats['std']

    def destandardize(self, image):
        return image * DemoDataset.stats['std'] + DemoDataset.stats['mean']

    def center_crop(self, src, size):
        y, x = get_crop_indices(src, size)
        dst = src[y:y + size[0], x:x + size[1]]
        return dst

    def __getitem__(self, idx):
        image, image_path = self.get_image(idx, downsample=True)

        image = self.center_crop(image, DemoDataset.input_size)

        # Standardize
        image = self.standardize(image)

        # To tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).clone()

        return {'image': image, 'image_path': image_path}


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
    test_loader = data.DataLoader(test, batch_size=1, shuffle=False)

    # Make output directories
    if not os.path.isdir(out):
        os.makedirs(out)

    # Predict depths from single RGB images
    with torch.no_grad():
        for sample in tqdm.tqdm(test_loader):
            # Get input image
            image, image_path = sample['image'], sample['image_path'][0]

            # Get input size
            input_size = (image.shape[2], image.shape[3])

            # Send input tensor to the specified device
            image = image.to(device)

            # Inference
            output = model(image)
            output = F.interpolate(output, size=input_size,
                                   mode='bilinear', align_corners=True)

            # Copy to the CPU and squeeze to 2D image
            output = np.squeeze(output.to('cpu').numpy().copy())

            # Get the original image
            image = np.squeeze(image.to('cpu').numpy().copy())
            image = utils.input_to_bgr(image, test)

            # Colorize depth map
            output_vis = utils.visualize_depthmap(output)

            # Save visualized images
            srcname = os.path.splitext(os.path.basename(image_path))[0]
            filename = os.path.join(out, srcname)
            cv2.imwrite('{}_input.png'.format(filename), image)
            cv2.imwrite('{}_output.png'.format(filename), output_vis)

            # Save predicted depth
            np.save('{}_output.npy'.format(filename), output)


def main():
    parser = argparse.ArgumentParser(
        description='Test a depth prediction network')
    parser.add_argument('--input', '-i', required=True,
                        help='Input image directory')
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

    print('Input: {}'.format(args.input))
    print('GPU: {}'.format(gpus))
    print('Model: {}'.format(args.model))
    print('')

    # Get configurations
    model_dir = os.path.basename(os.path.dirname(args.model))
    configs = model_dir.split('_')[1:]
    configs = {conf.split('=')[0]: conf.split('=')[1] for conf in configs}

    arch = configs['arch']

    test = DemoDataset(args.input)

    out = os.path.join(os.path.dirname(args.model), 'demo_results')

    test_network(arch, device, test, out, args.model, parallel)


if __name__ == '__main__':
    main()
