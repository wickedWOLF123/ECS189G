#!/usr/bin/env python3
"""
Sequential training script for all teacher models with optimal hyperparameters
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_training(script_name, model_name, **kwargs):
    """
    Run training script with specified parameters
    """
    cmd = [sys.executable, script_name]
    
    # Add arguments
    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting training: {model_name}")
    print(f"Script: {script_name}")
    print(f"Parameters: {kwargs}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Successfully completed: {model_name}")
        print(f"Training time: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n❌ Failed to train: {model_name}")
        print(f"Error: {e}")
        print(f"Time before failure: {duration/60:.1f} minutes")
        return False, duration

def main():
    print("🎯 Training All Teacher Models with Optimal Hyperparameters")
    print("=" * 80)
    print("This will train all 4 teacher models sequentially with hyperparameters")
    print("optimized for achieving the best known accuracy on CIFAR-10.")
    print()
    
    # Define optimal training configurations for each model
    training_configs = [
        {
            'script': 'train_resnet56.py',
            'name': 'ResNet-56',
            'params': {
                'epochs': 200,
                'batch_size': 128,
                'lr': 0.1,
                'weight_decay': 1e-4,
                'device': 'cuda'
            }
        },
        {
            'script': 'train_resnet110.py',
            'name': 'ResNet-110',
            'params': {
                'epochs': 200,
                'batch_size': 128,
                'lr': 0.1,
                'weight_decay': 1e-4,
                'device': 'cuda'
            }
        },
        {
            'script': 'train_densenet121.py',
            'name': 'DenseNet-121',
            'params': {
                'epochs': 200,
                'batch_size': 64,  # Reduced batch size for memory efficiency
                'lr': 0.1,
                'weight_decay': 1e-4,
                'device': 'cuda'
            }
        },
        {
            'script': 'train_vgg16.py',
            'name': 'VGG-16',
            'params': {
                'epochs': 200,
                'batch_size': 128,
                'lr': 0.01,  # Lower learning rate for VGG
                'weight_decay': 5e-4,
                'device': 'cuda'
            }
        }
    ]
    
    print("Training Schedule:")
    for i, config in enumerate(training_configs, 1):
        print(f"  {i}. {config['name']} - {config['params']['epochs']} epochs")
    print()
    
    # Train each model sequentially
    results = {}
    total_start_time = time.time()
    
    for config in training_configs:
        success, duration = run_training(
            config['script'], 
            config['name'], 
            **config['params']
        )
        results[config['name']] = {
            'success': success,
            'duration': duration
        }
        
        # Brief pause between trainings
        if success:
            print(f"\n⏸️  Pausing for 30 seconds before next training...")
            time.sleep(30)
    
    # Print final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*80}")
    print("🏁 TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total training time: {total_duration/3600:.2f} hours")
    print()
    
    successful_models = 0
    for model_name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration_str = f"{result['duration']/3600:.2f}h" if result['duration'] >= 3600 else f"{result['duration']/60:.1f}m"
        print(f"{model_name:15} : {status:10} ({duration_str})")
        if result['success']:
            successful_models += 1
    
    print(f"\nResults: {successful_models}/{len(results)} models trained successfully")
    
    if successful_models == len(results):
        print("\n🎉 All teacher models trained successfully!")
        print("\nNext steps:")
        print("1. Check the 'checkpoints/' directory for saved models")
        print("2. Run evaluation script to test all models")
        print("3. Use these models as teachers for knowledge distillation")
    else:
        print(f"\n⚠️  {len(results) - successful_models} model(s) failed to train.")
        print("Please check the error messages above and retry failed models.")
    
    return 0 if successful_models == len(results) else 1

if __name__ == '__main__':
    exit(main()) 