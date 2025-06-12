#!/usr/bin/env python3
"""
Test script to verify all teacher models can be instantiated correctly
"""

import torch
import sys
import os

# Add current directory to path to import from training scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model(model_func, model_name):
    """Test if a model can be instantiated and get its parameter count"""
    try:
        model = model_func()
        param_count = sum(p.numel() for p in model.parameters())
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ {model_name:15} - Parameters: {param_count:,} - Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ {model_name:15} - Error: {e}")
        return False

def main():
    print("Testing Teacher Models for CIFAR-10")
    print("=" * 50)
    
    # Import model functions
    from train_resnet56 import ResNet56
    from train_resnet110 import ResNet110
    from train_densenet121 import create_densenet121_cifar10
    from train_vgg16 import create_vgg16_cifar10
    
    # Test all models
    models_to_test = [
        (ResNet56, "ResNet-56"),
        (ResNet110, "ResNet-110"),
        (create_densenet121_cifar10, "DenseNet-121"),
        (create_vgg16_cifar10, "VGG-16")
    ]
    
    results = []
    for model_func, model_name in models_to_test:
        success = test_model(model_func, model_name)
        results.append((model_name, success))
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    
    if successful == total:
        print("🎉 All models are ready for training!")
    else:
        print("⚠️  Some models have issues. Please check the errors above.")

if __name__ == '__main__':
    main() 