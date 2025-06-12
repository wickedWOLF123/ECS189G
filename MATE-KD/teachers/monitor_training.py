#!/usr/bin/env python3
"""
Monitor training progress by checking checkpoints and logs
"""

import os
import glob
import time
from datetime import datetime

def check_training_progress():
    """
    Check training progress by examining checkpoint files
    """
    checkpoint_dir = 'checkpoints'
    
    print("🔍 Training Progress Monitor")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not os.path.exists(checkpoint_dir):
        print("❌ Checkpoint directory not found. Training may not have started yet.")
        return
    
    # Check for checkpoint files
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    
    if not checkpoints:
        print("⏳ No checkpoints found yet. Training may be in progress...")
        return
    
    print(f"📁 Found {len(checkpoints)} checkpoint files:")
    print()
    
    # Group checkpoints by model
    models = ['resnet56', 'resnet110', 'densenet121', 'vgg16']
    model_status = {}
    
    for model in models:
        model_checkpoints = [cp for cp in checkpoints if model in os.path.basename(cp)]
        
        if not model_checkpoints:
            model_status[model] = "❌ Not started"
            continue
        
        # Find best checkpoint
        best_checkpoints = [cp for cp in model_checkpoints if 'best' in cp]
        regular_checkpoints = [cp for cp in model_checkpoints if 'best' not in cp]
        
        if best_checkpoints:
            # Extract accuracy from best checkpoint
            try:
                best_cp = best_checkpoints[-1]  # Latest best
                acc_str = best_cp.split('_acc_')[1].split('.pth')[0]
                accuracy = float(acc_str)
                
                # Check if training is complete (look for high epoch numbers)
                epoch_checkpoints = []
                for cp in regular_checkpoints:
                    try:
                        epoch_str = cp.split('_epoch_')[1].split('_acc_')[0]
                        epoch = int(epoch_str)
                        epoch_checkpoints.append(epoch)
                    except:
                        continue
                
                max_epoch = max(epoch_checkpoints) if epoch_checkpoints else 0
                
                if max_epoch >= 190:  # Near completion
                    model_status[model] = f"✅ Complete - Best: {accuracy:.2f}% (Epoch {max_epoch})"
                elif max_epoch >= 100:
                    model_status[model] = f"🔄 Training - Best: {accuracy:.2f}% (Epoch {max_epoch})"
                else:
                    model_status[model] = f"🚀 Started - Best: {accuracy:.2f}% (Epoch {max_epoch})"
                    
            except:
                model_status[model] = f"🔄 In progress - {len(model_checkpoints)} checkpoints"
        else:
            model_status[model] = f"🚀 Started - {len(model_checkpoints)} checkpoints"
    
    # Display status
    print("Model Training Status:")
    print("-" * 40)
    model_names = {
        'resnet56': 'ResNet-56',
        'resnet110': 'ResNet-110', 
        'densenet121': 'DenseNet-121',
        'vgg16': 'VGG-16'
    }
    
    for model, status in model_status.items():
        print(f"{model_names[model]:<15}: {status}")
    
    # Overall progress
    completed = sum(1 for status in model_status.values() if "Complete" in status)
    in_progress = sum(1 for status in model_status.values() if "Training" in status or "Started" in status)
    not_started = sum(1 for status in model_status.values() if "Not started" in status)
    
    print()
    print(f"Overall Progress: {completed}/4 completed, {in_progress} in progress, {not_started} not started")
    
    if completed == 4:
        print("\n🎉 All models training completed!")
        print("Run 'python evaluate_all_teachers.py' to evaluate the models.")
    elif in_progress > 0:
        print(f"\n⏳ Training in progress... Check back later for updates.")
    
    # Show recent checkpoint files
    if checkpoints:
        print(f"\n📋 Recent checkpoint files:")
        recent_checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)[:5]
        for cp in recent_checkpoints:
            mtime = datetime.fromtimestamp(os.path.getmtime(cp))
            print(f"  {mtime.strftime('%H:%M:%S')} - {os.path.basename(cp)}")

def main():
    check_training_progress()

if __name__ == '__main__':
    main() 