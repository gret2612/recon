# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:28:01 2021

@author: kaneko.naoshi
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AugmentedDepthDataset, DepthDataset
import meters
from models import modules, network, resnet, densenet, senet
import sobel
import utils


def build_model(arch):
    if arch == 'resnet':
        original_model = resnet.resnet50(pretrained=True)
        encoder = modules.E_resnet(original_model)
        model = network.model(encoder, num_features=2048,
                              block_channel=[256, 512, 1024, 2048])
    elif arch == 'densenet':
        original_model = densenet.densenet161(pretrained=True)
        encoder = modules.E_densenet(original_model)
        model = network.model(encoder, num_features=2208,
                              block_channel=[192, 384, 1056, 2208])
    elif arch == 'senet':
        original_model = senet.senet154(pretrained=True)
        encoder = modules.E_senet(original_model)
        model = network.model(encoder, num_features=2048,
                              block_channel=[256, 512, 1024, 2048])
    else:
        raise NotImplementedError

    return model


class LossFunc:
    def __init__(self, device, bootstrap_ratio):
        self._device = device
        self._bootstrap_ratio = bootstrap_ratio
        self.grad_func = sobel.Sobel().to(device)
        self.cos = nn.CosineSimilarity(dim=1, eps=0)

    @property
    def device(self):
        return self._device

    @property
    def bootstrap_ratio(self):
        return self._bootstrap_ratio

    def get_gradient(self, x, size):
        grad = self.grad_func(x)
        grad_dx = grad[:, 0, :, :].contiguous().view(size)
        grad_dy = grad[:, 1, :, :].contiguous().view(size)
        return grad_dx, grad_dy

    def bootstrap(self, x):
        x_flat = torch.flatten(x, start_dim=1)
        topk, _ = torch.topk(x_flat, k=x_flat.shape[1] // self.bootstrap_ratio)
        return topk.mean()

    def __call__(self, input, target):
        input_grad_dx, input_grad_dy = self.get_gradient(input,
                                                         target.size())
        target_grad_dx, target_grad_dy = self.get_gradient(target,
                                                           target.size())

        assert target.size(1) == 1
        ones = torch.ones_like(target, dtype=torch.float).to(self.device)

        input_normal = torch.cat(
            (-input_grad_dx, -input_grad_dy, ones), dim=1)
        target_normal = torch.cat(
            (-target_grad_dx, -target_grad_dy, ones), dim=1)

        loss_depth = torch.log(torch.abs(input - target) + 0.5)
        loss_dx = torch.log(
            torch.abs(input_grad_dx - target_grad_dx) + 0.5)
        loss_dy = torch.log(
            torch.abs(input_grad_dy - target_grad_dy) + 0.5)
        loss_normal = torch.abs(
            1 - self.cos(input_normal, target_normal))

        if self.bootstrap_ratio > 1:
            loss_depth = self.bootstrap(loss_depth)
            loss_dx = self.bootstrap(loss_dx)
            loss_dy = self.bootstrap(loss_dy)
            loss_normal = self.bootstrap(loss_normal)
        else:
            loss_depth = loss_depth.mean()
            loss_dx = loss_dx.mean()
            loss_dy = loss_dy.mean()
            loss_normal = loss_normal.mean()

        return loss_depth + loss_normal + (loss_dx + loss_dy)


def train_epoch(train_loader, model, criterion,
                optimizer, epoch, device, print_freq):
    batch_time = meters.AverageMeter('Time', ':6.3f')
    data_time = meters.AverageMeter('Data', ':6.3f')
    losses = meters.AverageMeter('Loss', ':.3f')

    end = time.time()
    for i, sample in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Get input images and target depth maps
        image, target = sample['image'], sample['depth']

        # Send tensors to the specified device
        image = image.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward
        output = model(image)

        # Compute loss
        loss = criterion(output, target)

        # Record loss
        losses.update(loss.item(), image.shape[0])

        # Compute gradient and do optimizer step
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print statistics
        progress = meters.ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch + 1))

        if (i + 1) % print_freq == 0:
            progress.display(i)


def train_network(arch, batch_size, lr, weight_decay, epochs, bootstrap_ratio,
                  device, train, out, print_freq, parallel, model_path=None):

    # Set up a model to train
    if parallel:
        model = torch.nn.DataParallel(build_model(arch)).to(device)
    else:
        model = build_model(arch).to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # Set up an optimizer
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)

    # Define a loss function
    criterion = LossFunc(device, bootstrap_ratio)

    # Set up a learning rate scheduler
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.1 ** (epoch // 5))

    # Set up a data loader
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=4)

    for epoch in range(epochs):
        train_epoch(train_loader, model, criterion,
                    optimizer, epoch, device, print_freq)
        scheduler.step()

    # Make output directories
    if not os.path.isdir(out):
        os.makedirs(out)

    # Save the trained model
    filename = os.path.join(out, 'network.pt')
    torch.save(model.state_dict(), filename)

    return filename


def main():
    parser = argparse.ArgumentParser(
        description='Train a depth estimation network')
    parser.add_argument('--dataset', '-d', default='synthetic',
                        const='synthetic', nargs='?',
                        choices=['synthetic'],
                        help='Depth dataset to train')
    parser.add_argument('--mask', '-m', action='store_true',
                        help='Apply foreground mask')
    parser.add_argument('--arch', '-a', default='resnet',
                        const='resnet', nargs='?',
                        choices=['resnet', 'densenet', 'senet'],
                        help='Network architecture')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay rate')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--bootstrap_ratio', '-r', type=int, default=1,
                        help='1/ratio of pixels contribute to the loss')
    parser.add_argument('--gpu', '-g', nargs='+', default=['-1'],
                        help='GPU IDs (negative value indicates CPU)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    args = parser.parse_args()

    gpus = ','.join(args.gpu)

    # Negative -> CPU
    if '-' not in gpus:
        use_cuda = True
    else:
        use_cuda = False
        gpus = '-1'

    utils.set_random_seed(args.seed, use_cuda)

    if use_cuda:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        parallel = True if ',' in gpus else False
    else:
        device = torch.device('cpu')
        parallel = False

    dataset_dir = os.path.join('data', args.dataset)
    train_dir = os.path.join(dataset_dir, 'train')

    print('GPU: {}'.format(gpus))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# epochs: {}'.format(args.epochs))
    print('Dataset: {}'.format(args.dataset))
    print('Mask: {}'.format(args.mask))
    print('Arch: {}'.format(args.arch))
    print('')

    if args.dataset == 'synthetic':
        if args.mask:
            train = (DepthDataset(train_dir, train=True),
                     AugmentedDepthDataset(train_dir, train=True))
        else:
            train = DepthDataset(train_dir, train=True)
    else:
        raise NotImplementedError

    print_freq = 1

    dir_name = (
        '{}_mask={}_arch={}_bs={}_lr={}_ep={}_br={}_sd={}'
        ''.format(args.dataset, args.mask, args.arch, args.batch_size,
                  args.lr, args.epochs, args.bootstrap_ratio, args.seed)
    )

    out = os.path.join(args.out, dir_name)

    if isinstance(train, tuple):  # --mask
        print('Stage 1: train without mask')
        saved_path = train_network(
            args.arch, args.batch_size, args.lr,
            args.weight_decay, args.epochs // 2, args.bootstrap_ratio,
            device, train[0], out, print_freq, parallel
        )

        print('Stage 2: train with mask')
        train_network(
            args.arch, args.batch_size, args.lr * 0.1,
            args.weight_decay, args.epochs // 2, args.bootstrap_ratio,
            device, train[1], out, print_freq, parallel, saved_path
        )
    else:
        train_network(
            args.arch, args.batch_size, args.lr,
            args.weight_decay, args.epochs, args.bootstrap_ratio,
            device, train, out, print_freq, parallel
        )


if __name__ == '__main__':
    main()
