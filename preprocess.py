from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset


def load_data(args):
    train_transfrom = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (.5, .5, .5))
    ])
    train_data = CIFAR10('./data', train=True, download=True, transform=train_transfrom)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=args.num_workers)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (.5, .5, .5))
    ])
    test_data = CIFAR10('./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=400, shuffle=False, num_workers=0)

    return train_loader, test_loader
