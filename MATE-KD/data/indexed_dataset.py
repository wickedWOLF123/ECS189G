"""
Indexed CIFAR-10 Dataset - Returns (image, label, original_index)
"""

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional

class CIFAR10WithIndex(Dataset):
    """CIFAR-10 dataset that also returns the original dataset index"""
    
    def __init__(self, root: str = './data', train: bool = True, 
                 transform=None, download: bool = True):
        self.cifar_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )
        self.transform = transform
        
    def __len__(self):
        return len(self.cifar_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        # Get original data
        image, label = self.cifar_dataset[idx]
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label, idx

def get_indexed_cifar10_loaders(batch_size: int = 128, 
                               num_workers: int = 4,
                               pin_memory: bool = True,
                               root: str = './data',
                               augment_train: bool = True,
                               shuffle: bool = True):
    """Get CIFAR-10 loaders that return indices"""
    
    # CIFAR-10 normalization
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    
    # Transforms
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Datasets
    train_dataset = CIFAR10WithIndex(root=root, train=True, 
                                   transform=train_transform, download=True)
    test_dataset = CIFAR10WithIndex(root=root, train=False, 
                                  transform=test_transform, download=True)
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, test_loader 