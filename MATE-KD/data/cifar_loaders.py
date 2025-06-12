"""
CIFAR-10 Data Loaders with Proper Normalization
Supports regular CIFAR-10 and CIFAR-10-C (corrupted) datasets
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
from typing import Tuple, Optional, List

# CIFAR-10 Statistics (computed from training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# CIFAR-10 Class Names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-10-C Corruption Types
CIFAR10C_CORRUPTIONS = [
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
    'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
    'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
    'snow', 'spatter', 'speckle_noise', 'zoom_blur'
]

def get_cifar10_transforms(training: bool = True, augment: bool = True) -> transforms.Compose:
    """
    Get CIFAR-10 transforms with proper normalization
    
    Args:
        training: Whether this is for training data
        augment: Whether to apply data augmentation (only for training)
    
    Returns:
        transforms.Compose: Transform pipeline
    """
    if training and augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        # Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    
    return transform

def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    root: str = './data',
    download: bool = True,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 train and test data loaders
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        root: Root directory for dataset
        download: Whether to download dataset if not found
        augment_train: Whether to apply augmentation to training data
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Get transforms
    train_transform = get_cifar10_transforms(training=True, augment=augment_train)
    test_transform = get_cifar10_transforms(training=False, augment=False)
    
    # Create datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=download, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=download, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, test_loader

def get_cifar10_test_loader(
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    root: str = './data',
    download: bool = True
) -> DataLoader:
    """
    Get CIFAR-10 test data loader only
    
    Args:
        batch_size: Batch size for data loader
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        root: Root directory for dataset
        download: Whether to download dataset if not found
    
    Returns:
        DataLoader: Test data loader
    """
    test_transform = get_cifar10_transforms(training=False, augment=False)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=download, transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return test_loader

def denormalize_cifar10(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize CIFAR-10 tensor for visualization
    
    Args:
        tensor: Normalized tensor with shape (..., 3, 32, 32)
    
    Returns:
        torch.Tensor: Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    
    # Move to same device as input tensor
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    denormalized = (tensor * std) + mean
    return torch.clamp(denormalized, 0, 1)

class CIFAR10CDataset(torch.utils.data.Dataset):
    """
    CIFAR-10-C Dataset for loading corrupted test images
    """
    
    def __init__(self, root: str, corruption_type: str, severity: int = 1, transform=None):
        """
        Args:
            root: Root directory containing CIFAR-10-C data
            corruption_type: Type of corruption (e.g., 'gaussian_noise')
            severity: Corruption severity (1-5)
            transform: Optional transform to apply
        """
        self.root = root
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        # Load data
        data_path = os.path.join(root, f'{corruption_type}.npy')
        labels_path = os.path.join(root, 'labels.npy')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CIFAR-10-C data not found: {data_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"CIFAR-10-C labels not found: {labels_path}")
        
        # Load all data and select severity level
        all_data = np.load(data_path)
        all_labels = np.load(labels_path)
        
        # Each severity has 10,000 images
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        self.data = all_data[start_idx:end_idx]
        self.labels = all_labels[start_idx:end_idx]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image if transform expects it
        if self.transform:
            # Ensure img is in uint8 format
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        else:
            # Convert to tensor manually
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1)  # HWC -> CHW
        
        return img, label

def get_cifar10c_loader(
    root: str,
    corruption_type: str,
    severity: int = 1,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Get CIFAR-10-C data loader for specific corruption and severity
    
    Args:
        root: Root directory containing CIFAR-10-C data
        corruption_type: Type of corruption
        severity: Corruption severity (1-5)
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        DataLoader: CIFAR-10-C data loader
    """
    transform = get_cifar10_transforms(training=False, augment=False)
    
    dataset = CIFAR10CDataset(
        root=root, 
        corruption_type=corruption_type, 
        severity=severity, 
        transform=transform
    )
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return loader

def get_all_cifar10c_loaders(
    root: str,
    severity: int = 1,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True
) -> dict:
    """
    Get data loaders for all CIFAR-10-C corruption types
    
    Args:
        root: Root directory containing CIFAR-10-C data
        severity: Corruption severity (1-5)
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
    
    Returns:
        dict: Dictionary mapping corruption names to data loaders
    """
    loaders = {}
    
    for corruption in CIFAR10C_CORRUPTIONS:
        try:
            loader = get_cifar10c_loader(
                root=root,
                corruption_type=corruption,
                severity=severity,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            loaders[corruption] = loader
            print(f"✅ Loaded {corruption} (severity {severity})")
        except FileNotFoundError:
            print(f"❌ Missing {corruption}")
    
    return loaders

# Convenience functions for backward compatibility
def cifar_loader_correct(batch_size=128, root='./data'):
    """Legacy function name for compatibility"""
    return get_cifar10_test_loader(batch_size=batch_size, root=root)

if __name__ == "__main__":
    # Test the data loaders
    print("🧪 Testing CIFAR-10 data loaders...")
    
    # Test regular CIFAR-10
    train_loader, test_loader = get_cifar10_loaders(batch_size=4)
    
    # Get a batch
    images, labels = next(iter(test_loader))
    print(f"✅ CIFAR-10 batch shape: {images.shape}")
    print(f"   Labels: {labels}")
    print(f"   Data range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    
    # Test denormalization
    denorm_images = denormalize_cifar10(images)
    print(f"   Denormalized range: [{denorm_images.min().item():.3f}, {denorm_images.max().item():.3f}]")
    
    print(f"\n📊 Dataset info:")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Classes: {len(CIFAR10_CLASSES)}")
    print(f"   Class names: {CIFAR10_CLASSES}") 