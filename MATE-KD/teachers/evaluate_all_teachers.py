#!/usr/bin/env python3
"""
Evaluation script for all trained teacher models
"""

import torch
import torch.nn as nn
import os
import glob
import sys
from utils import get_cifar10_dataloaders, test_epoch
from train_resnet56 import ResNet56
from train_resnet110 import ResNet110
from train_densenet121 import create_densenet121_cifar10
from train_vgg16 import create_vgg16_cifar10
from train_vgg19 import create_vgg19_cifar10

def load_best_checkpoint(model_name, checkpoint_dir='checkpoints'):
    """
    Load the best checkpoint for a given model
    """
    pattern = os.path.join(checkpoint_dir, f'{model_name}_best_*.pth')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        print(f"❌ No best checkpoint found for {model_name}")
        return None
    
    # Get the checkpoint with highest accuracy
    best_checkpoint = None
    best_acc = 0
    
    for checkpoint_path in checkpoints:
        # Extract accuracy from filename
        try:
            acc_str = checkpoint_path.split('_acc_')[1].split('.pth')[0]
            acc = float(acc_str)
            if acc > best_acc:
                best_acc = acc
                best_checkpoint = checkpoint_path
        except:
            continue
    
    if best_checkpoint:
        print(f"📁 Loading best checkpoint for {model_name}: {os.path.basename(best_checkpoint)}")
        return torch.load(best_checkpoint)
    else:
        print(f"❌ Could not parse checkpoint files for {model_name}")
        return None

def evaluate_model(model_func, model_name, device):
    """
    Evaluate a single model
    """
    print(f"\n{'='*60}")
    print(f"🔍 Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Create model
    model = model_func().to(device)
    
    # Load best checkpoint
    checkpoint = load_best_checkpoint(model_name.lower().replace('-', ''))
    if checkpoint is None:
        return None
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get data loader
    _, test_loader = get_cifar10_dataloaders(batch_size=128)
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    
    result = {
        'model_name': model_name,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'checkpoint_epoch': checkpoint['epoch'],
        'checkpoint_accuracy': checkpoint['accuracy'],
        'parameters': sum(p.numel() for p in model.parameters())
    }
    
    print(f"✅ {model_name} Evaluation Complete:")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Parameters: {result['parameters']:,}")
    print(f"   Checkpoint Epoch: {checkpoint['epoch']}")
    
    return result

def main():
    print("🎯 Evaluating All Trained Teacher Models")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define models to evaluate
    models_to_evaluate = [
        (ResNet56, "ResNet-56"),
        (ResNet110, "ResNet-110"),
        (create_densenet121_cifar10, "DenseNet-121"),
        (create_vgg16_cifar10, "VGG-16"),
        (create_vgg19_cifar10, "VGG-19")
    ]
    
    # Check if checkpoint directory exists
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory '{checkpoint_dir}' not found!")
        print("Please train the models first using train_all_sequential.py")
        return 1
    
    # List available checkpoints
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not checkpoints:
        print(f"❌ No checkpoints found in '{checkpoint_dir}'!")
        print("Please train the models first using train_all_sequential.py")
        return 1
    
    print(f"\nFound {len(checkpoints)} checkpoint files:")
    for checkpoint in sorted(checkpoints):
        print(f"  📁 {os.path.basename(checkpoint)}")
    
    # Evaluate each model
    results = []
    for model_func, model_name in models_to_evaluate:
        try:
            result = evaluate_model(model_func, model_name, device)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("📊 FINAL EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    if not results:
        print("❌ No models were successfully evaluated!")
        return 1
    
    # Sort by accuracy
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    print(f"{'Model':<15} {'Accuracy':<10} {'Loss':<8} {'Parameters':<12} {'Epoch':<6}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model_name']:<15} "
              f"{result['test_accuracy']:<10.2f}% "
              f"{result['test_loss']:<8.4f} "
              f"{result['parameters']:<12,} "
              f"{result['checkpoint_epoch']:<6}")
    
    # Performance analysis
    best_model = results[0]
    print(f"\n🏆 Best performing model: {best_model['model_name']} ({best_model['test_accuracy']:.2f}%)")
    
    # Expected vs actual performance
    expected_performance = {
        'ResNet-56': (93, 94),
        'ResNet-110': (94, 95),
        'DenseNet-121': (95, 96),
        'VGG-16': (92, 93),
        'VGG-19': (91, 92)
    }
    
    print(f"\n📈 Performance Analysis:")
    print(f"{'Model':<15} {'Actual':<8} {'Expected':<12} {'Status':<10}")
    print("-" * 50)
    
    for result in results:
        model_name = result['model_name']
        actual_acc = result['test_accuracy']
        
        if model_name in expected_performance:
            min_exp, max_exp = expected_performance[model_name]
            expected_str = f"{min_exp}-{max_exp}%"
            
            if actual_acc >= min_exp:
                status = "✅ Good"
            elif actual_acc >= min_exp - 2:
                status = "⚠️  Close"
            else:
                status = "❌ Low"
        else:
            expected_str = "Unknown"
            status = "❓ N/A"
        
        print(f"{model_name:<15} {actual_acc:<8.2f}% {expected_str:<12} {status:<10}")
    
    print(f"\n📁 All model checkpoints are saved in: {os.path.abspath(checkpoint_dir)}")
    print("🎓 These models are now ready to be used as teachers for knowledge distillation!")
    
    return 0

if __name__ == '__main__':
    exit(main()) 