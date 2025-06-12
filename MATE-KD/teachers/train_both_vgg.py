#!/usr/bin/env python3
"""
Sequential training script for both VGG-16 and VGG-19 models
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
    print("🎯 Training Both VGG Models with Optimal Hyperparameters")
    print("=" * 80)
    print("This will train VGG-16 and VGG-19 sequentially with hyperparameters")
    print("optimized for achieving the best accuracy on CIFAR-10.")
    print()
    
    # Define optimal training configurations for both VGG models
    training_configs = [
        {
            'script': 'train_vgg16.py',
            'name': 'VGG-16',
            'params': {
                'epochs': 200,
                'batch_size': 128,
                'lr': 0.01,
                'weight_decay': 5e-4,
                'device': 'cuda'
            },
            'expected_time': '3-4 hours',
            'expected_acc': '92-93%'
        },
        {
            'script': 'train_vgg19.py',
            'name': 'VGG-19',
            'params': {
                'epochs': 200,
                'batch_size': 128,
                'lr': 0.01,
                'weight_decay': 5e-4,
                'device': 'cuda'
            },
            'expected_time': '4-5 hours',
            'expected_acc': '93-94%'
        }
    ]
    
    print("Training Schedule:")
    total_expected_time = 0
    for i, config in enumerate(training_configs, 1):
        print(f"  {i}. {config['name']} - {config['params']['epochs']} epochs")
        print(f"     Expected time: {config['expected_time']}")
        print(f"     Expected accuracy: {config['expected_acc']}")
        if i == 1:
            total_expected_time += 3.5  # Average of 3-4 hours
        else:
            total_expected_time += 4.5  # Average of 4-5 hours
    
    print(f"\nTotal expected time: ~{total_expected_time:.1f} hours")
    print("Perfect for your 8-hour window! 🕐")
    print()
    
    # Train each model sequentially
    results = {}
    total_start_time = time.time()
    
    for i, config in enumerate(training_configs):
        print(f"\n🔄 Training model {i+1}/2: {config['name']}")
        
        success, duration = run_training(
            config['script'], 
            config['name'], 
            **config['params']
        )
        results[config['name']] = {
            'success': success,
            'duration': duration
        }
        
        # Brief pause between trainings (except after last model)
        if success and i < len(training_configs) - 1:
            print(f"\n⏸️  Pausing for 30 seconds before next training...")
            time.sleep(30)
        
        # Show progress
        elapsed_total = time.time() - total_start_time
        remaining_models = len(training_configs) - (i + 1)
        print(f"\n📊 Progress: {i+1}/{len(training_configs)} models completed")
        print(f"⏱️  Total elapsed time: {elapsed_total/3600:.2f} hours")
        if remaining_models > 0:
            print(f"🔮 Estimated remaining time: ~{remaining_models * 4:.1f} hours")
    
    # Print final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*80}")
    print("🏁 VGG TRAINING SUMMARY")
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
    
    print(f"\nResults: {successful_models}/{len(results)} VGG models trained successfully")
    
    if successful_models == len(results):
        print("\n🎉 Both VGG models trained successfully!")
        print("\n📈 You now have 5 excellent teacher models:")
        print("  1. ResNet-56 (91.37%)")
        print("  2. ResNet-110 (92.47%)")
        print("  3. DenseNet-121 (95.54%)")
        print("  4. VGG-16 (training completed)")
        print("  5. VGG-19 (training completed)")
        print("\n🎓 Perfect teacher ensemble for knowledge distillation!")
        print("\nNext steps:")
        print("1. Run 'python evaluate_all_teachers.py' to see final accuracies")
        print("2. Begin knowledge distillation experiments")
        print("3. Train student models using these teachers")
    else:
        print(f"\n⚠️  {len(results) - successful_models} VGG model(s) failed to train.")
        print("Check the error messages above and retry failed models if needed.")
    
    return 0 if successful_models == len(results) else 1

if __name__ == '__main__':
    exit(main()) 