#!/usr/bin/env python3
"""
MATE-KD Results Reproduction Script

Reproduces the main results from the MATE-KD paper including:
- Model accuracy comparison
- CIFAR-10-C robustness evaluation  
- Grad-CAM similarity analysis
- Confusion matrix analysis
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from student.zoo import build_student, create_resnet110_cifar10, create_densenet121_cifar10, create_vgg_cifar10
from data.indexed_dataset import CIFAR10WithIndex
from explainers.gradcam import GradCAM
from explainers.gradcam_utils import set_inplace
from core.similarity import cosine_flat_clipped, pearson_correlation_clipped

def load_model(model_path, model_type, device):
    """Load a model from checkpoint"""
    if model_type == 'student':
        model = build_student(num_classes=10)
    elif model_type == 'resnet110':
        model = create_resnet110_cifar10()
    elif model_type == 'densenet':
        model = create_densenet121_cifar10()
    elif model_type == 'vgg19':
        model = create_vgg_cifar10('vgg19')
        set_inplace(model, False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def evaluate_accuracy(model, test_loader, device):
    """Evaluate model accuracy on test set"""
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def reproduce_accuracy_results(args, device):
    """Reproduce main accuracy comparison results"""
    print("Reproducing accuracy results...")
    print("-" * 50)
    
    # Load test dataset
    test_dataset = CIFAR10WithIndex(root='./data/cifar-10-batches-py', train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    # Model paths (adjust based on your setup)
    model_configs = {
        'Student (MATE-KD)': {
            'path': args.student_model or 'archive/results/best_student_90.22pct/best_model_90.22pct.pth',
            'type': 'student'
        },
        'ResNet110': {
            'path': 'archive/models/resnet110_best.pth',
            'type': 'resnet110'
        },
        'DenseNet121': {
            'path': 'archive/models/densenet121_best.pth', 
            'type': 'densenet'
        },
        'VGG19': {
            'path': 'archive/models/vgg19_best.pth',
            'type': 'vgg19'
        }
    }
    
    results = {}
    
    for model_name, config in model_configs.items():
        model_path = config['path']
        
        if not os.path.exists(model_path):
            print(f"  {model_name}: Model not found at {model_path}")
            continue
            
        try:
            model = load_model(model_path, config['type'], device)
            accuracy = evaluate_accuracy(model, test_loader, device)
            results[model_name] = accuracy
            print(f"  {model_name}: {accuracy:.2f}%")
            
        except Exception as e:
            print(f"  {model_name}: Error loading model - {e}")
    
    # Create results table
    if results:
        print(f"\nCIFAR-10 Test Accuracy Results:")
        print("-" * 40)
        
        # Sort by accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        
        for model_name, accuracy in sorted_results:
            print(f"{model_name:20}: {accuracy:5.2f}%")
    
    return results

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())

def reproduce_efficiency_analysis(args, device):
    """Reproduce model efficiency comparison"""
    print("\nReproducing efficiency analysis...")
    print("-" * 50)
    
    model_configs = {
        'Student (MATE-KD)': {
            'path': args.student_model or 'archive/results/best_student_90.22pct/best_model_90.22pct.pth',
            'type': 'student'
        },
        'ResNet110': {
            'path': 'archive/models/resnet110_best.pth',
            'type': 'resnet110'
        },
        'DenseNet121': {
            'path': 'archive/models/densenet121_best.pth',
            'type': 'densenet'
        },
        'VGG19': {
            'path': 'archive/models/vgg19_best.pth',
            'type': 'vgg19'
        }
    }
    
    print(f"{'Model':<20} | {'Parameters':<12} | {'Model Size (MB)':<15}")
    print("-" * 55)
    
    for model_name, config in model_configs.items():
        model_path = config['path']
        
        if not os.path.exists(model_path):
            continue
            
        try:
            model = load_model(model_path, config['type'], device)
            
            # Count parameters
            param_count = count_parameters(model)
            
            # Estimate model size
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            print(f"{model_name:<20} | {param_count/1e6:8.2f}M    | {model_size_mb:12.2f}")
            
        except Exception as e:
            print(f"{model_name:<20} | Error: {str(e)[:30]}")

def reproduce_key_results(args):
    """Reproduce the key results from the MATE-KD project"""
    print("MATE-KD Results Reproduction")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Reproduce accuracy results
    accuracy_results = reproduce_accuracy_results(args, device)
    
    # Reproduce efficiency analysis
    reproduce_efficiency_analysis(args, device)
    
    print(f"\n{'='*60}")
    print("Results reproduction complete!")
    
    return accuracy_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Reproduce MATE-KD results')
    parser.add_argument('--student-model', type=str, default=None,
                       help='Path to trained student model')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Reproduce key results
    results = reproduce_key_results(args)
    
    # Print summary
    if results:
        print(f"\nSummary:")
        print(f"  Student model achieved: {results.get('Student (MATE-KD)', 'N/A')}")
        
        # Check if target performance achieved
        student_acc = results.get('Student (MATE-KD)')
        if student_acc and student_acc >= 90.0:
            print(f"  Target performance (90%+): ACHIEVED")
        elif student_acc:
            print(f"  Target performance (90%+): {90.0 - student_acc:.2f}% gap remaining")

if __name__ == "__main__":
    main() 