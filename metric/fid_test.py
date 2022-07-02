from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
import torchvision.models as models

import numpy as np
from scipy import linalg
import warnings
import torch
import torchvision.transforms as transforms

from metric.fid import InceptionV3, get_fid
from tqdm import tqdm
import argparse


def load_args():
    parser = argparse.ArgumentParser("FID test")
    
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    
    return parser.parse_args()


def load_dataset(args):
    train_transfrom = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    train_data = CIFAR10(root='./data', train=True, download=True, transform=train_transfrom)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    test_data = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return train_loader, test_loader


def get_feature(model, train_loader, test_loader, args):
    count, t1, t2 = 0, [], []
    
    pbar = tqdm(train_loader)
    for i, (d, l) in enumerate(pbar):
        if args.cuda:
            d = d.cuda()
        if count <= 10000:
            t1_out = model(d)[0].view(d.size(0), -1)
            t1.append(t1_out)
        count += d.size(0)
        if count == 10000:
            break
        pbar.set_description("Extract train data feature: {}".format(tmp))

    test_pbar = tqdm(test_loader)
    for i, (d, l) in enumerate(test_pbar):
        if args.cuda:
            d = d.cuda()
        t2_out = model(d)[0].view(d.size(0), -1)
        t2.append(t2_out)
        test_pbar.set_description("Extract test data feature: {}".format((i + 1) * d.size(0)))
    t1 = torch.cat(t1, dim=0)
    t2 = torch.cat(t2, dim=0)
    
    return t1, t2
    

def main(args):
    train_loader, test_loader = load_dataset(args)
    
    model = InceptionV3()
    if args.cuda:
        model = model.cuda()
    
    print('--- Collecting train, test features ---')
    train_feats, test_feats = get_feature(model, train_loader, test_loader, args)
    print('--- Done for train, test features ---')
    
    print('Get FID: {}'.format(get_fid(train_feats.cpu().data.numpy(), test_feats.cpu().data.numpy())))
    
    
if __name__ == '__main__':
    args = load_args()
    main(args)

