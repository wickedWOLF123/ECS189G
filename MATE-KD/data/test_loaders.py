#!/usr/bin/env python3
"""
Test script for CIFAR-10 data loaders
"""

import torch
import matplotlib.pyplot as plt
from cifar_loaders import (
    get_cifar10_loaders, 
    get_cifar10_test_loader, 
    denormalize_cifar10,
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD
)

def test_basic_loading():
    """Test basic CIFAR-10 loading functionality"""
    print("🧪 Testing basic CIFAR-10 loading...")
    
    # Test train and test loaders
    train_loader, test_loader = get_cifar10_loaders(batch_size=4, num_workers=2)
    
    # Get a batch from each
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))
    
    print(f"✅ Train batch shape: {train_images.shape}")
    print(f"✅ Test batch shape: {test_images.shape}")
    print(f"   Data range: [{test_images.min().item():.3f}, {test_images.max().item():.3f}]")
    
    # Test denormalization
    denorm_images = denormalize_cifar10(test_images)
    print(f"   Denormalized range: [{denorm_images.min().item():.3f}, {denorm_images.max().item():.3f}]")
    
    return train_loader, test_loader, test_images, test_labels

def test_visualization(images, labels):
    """Test visualization with denormalization"""
    print("\n🎨 Testing visualization...")
    
    # Denormalize for visualization
    denorm_images = denormalize_cifar10(images)
    
    # Create a simple plot
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    for i in range(4):
        img = denorm_images[i].permute(1, 2, 0).cpu().numpy()
        label = labels[i].item()
        class_name = CIFAR10_CLASSES[label]
        
        axes[i].imshow(img)
        axes[i].set_title(f'{class_name}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualization saved as 'test_visualization.png'")

def test_normalization_stats():
    """Test that normalization statistics are correct"""
    print("\n📊 Testing normalization statistics...")
    
    print(f"CIFAR-10 Mean: {CIFAR10_MEAN}")
    print(f"CIFAR-10 Std:  {CIFAR10_STD}")
    
    # Load a batch and check normalization
    test_loader = get_cifar10_test_loader(batch_size=100, num_workers=2)
    images, _ = next(iter(test_loader))
    
    # Calculate actual mean and std
    actual_mean = images.mean(dim=[0, 2, 3])
    actual_std = images.std(dim=[0, 2, 3])
    
    print(f"Actual Mean:   {tuple(actual_mean.tolist())}")
    print(f"Actual Std:    {tuple(actual_std.tolist())}")
    
    # Check if close to zero mean, unit std (normalized)
    mean_close = torch.allclose(actual_mean, torch.zeros(3), atol=0.1)
    std_close = torch.allclose(actual_std, torch.ones(3), atol=0.1)
    
    if mean_close and std_close:
        print("✅ Normalization is working correctly!")
    else:
        print("⚠️  Normalization might have issues")

def test_dataset_info():
    """Test dataset information"""
    print("\n📈 Testing dataset information...")
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    print(f"Train samples: {len(train_loader.dataset):,}")
    print(f"Test samples:  {len(test_loader.dataset):,}")
    print(f"Classes:       {len(CIFAR10_CLASSES)}")
    print(f"Class names:   {CIFAR10_CLASSES}")
    
    # Check class distribution in a batch
    images, labels = next(iter(test_loader))
    unique_labels, counts = torch.unique(labels, return_counts=True)
    
    print(f"\nSample batch class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {CIFAR10_CLASSES[label]}: {count.item()}")

def main():
    """Run all tests"""
    print("🚀 CIFAR-10 Data Loader Test Suite")
    print("=" * 50)
    
    try:
        # Test basic loading
        train_loader, test_loader, test_images, test_labels = test_basic_loading()
        
        # Test visualization
        test_visualization(test_images, test_labels)
        
        # Test normalization
        test_normalization_stats()
        
        # Test dataset info
        test_dataset_info()
        
        print("\n🎉 All tests passed successfully!")
        print("\n💡 Usage examples:")
        print("   from data.cifar_loaders import get_cifar10_loaders")
        print("   train_loader, test_loader = get_cifar10_loaders()")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 