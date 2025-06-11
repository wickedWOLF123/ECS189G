import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

def get_cifar10_loaders(batch_size: int = 128,
                         num_workers: int = 4,
                         data_dir: str = './data',
                         pin_memory: bool = False):
    """
    Returns train and test loaders for CIFAR-10 with standard augmentations.
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    # Download / load datasets
    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory,
                                               persistent_workers=(num_workers > 0))

    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=True,
                               transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              persistent_workers=(num_workers > 0))

    return train_loader, test_loader

class CIFAR10C(Dataset):
    """
    CIFAR-10-C loader: corruption_type is e.g. 'gaussian_noise', severity in [1,5].
    """
    def __init__(self, root='./data/CIFAR-10-C', corruption='gaussian_noise', severity=1, transform=None):
        # Load all 10 corruption types, pick one
        xs = np.load(f"{root}/{corruption}.npy")  # shape: (10000, 32, 32, 3)
        ys = np.load(f"{root}/labels.npy")        # shape: (10000,)
        # Select severity (0-indexed)
        xs = xs[(severity-1)*10000 : severity*10000]
        ys = ys[(severity-1)*10000 : severity*10000]
        self.xs = xs
        self.ys = ys
        self.transform = transform

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        img = self.xs[idx].astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        label = int(self.ys[idx])
        return img, label

def get_cifar10c_loader(batch_size=128, num_workers=4, 
                        corruption='gaussian_noise', severity=1, 
                        data_dir='./data/CIFAR-10-C'):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    dataset = CIFAR10C(root=data_dir, corruption=corruption, severity=severity, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader

