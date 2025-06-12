#!/usr/bin/env python3
"""
Demonstration Script: Precompute Grad-CAM Heatmaps for CIFAR-10

This script demonstrates how to precompute and cache Grad-CAM heatmaps
for efficient multi-teacher knowledge distillation training.

Usage:
    cd ECS189G/data
    python run_gradcam_precomputation.py
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run Grad-CAM precomputation for CIFAR-10')
    parser.add_argument('--models', nargs='+', 
                       choices=['densenet', 'vgg16', 'vgg19', 'resnet'],
                       default=['densenet', 'vgg16', 'resnet'],
                       help='Models to process (default: densenet vgg16 resnet)')
    parser.add_argument('--cache-dir', type=str, default='gradcam_cache',
                       help='Directory to store cached heatmaps (default: gradcam_cache)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the cache loader (skip precomputation)')
    
    args = parser.parse_args()
    
    print("🚀 GRAD-CAM PRECOMPUTATION FOR MULTI-TEACHER KNOWLEDGE DISTILLATION")
    print("=" * 70)
    
    if args.test_only:
        print("🧪 Testing cache loader only...")
        from gradcam_cache_loader import test_cache_loader
        test_cache_loader(args.cache_dir)
        return
    
    # Check if we're in the right directory
    if not Path('../cfg.yaml').exists():
        print("❌ Please run this script from the ECS189G/data directory")
        print("   Current directory:", os.getcwd())
        print("   Expected files: ../cfg.yaml, cifar_loaders.py")
        return
    
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🎯 Models to process: {', '.join(args.models)}")
    print(f"💾 Cache directory: {args.cache_dir}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🖥️  Device: {args.device}")
    
    # Import and run the precomputation
    try:
        from precompute_gradcam_cache import main as precompute_main
        
        print("\n" + "=" * 50)
        print("🔥 STARTING GRAD-CAM PRECOMPUTATION...")
        print("=" * 50)
        
        # Override sys.argv to pass arguments to the precomputation script
        original_argv = sys.argv.copy()
        sys.argv = [
            'precompute_gradcam_cache.py',
            '--config', '../cfg.yaml',
            '--cache-dir', args.cache_dir,
            '--batch-size', str(args.batch_size),
            '--device', args.device,
            '--models'
        ] + args.models
        
        # Run precomputation
        precompute_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print("\n" + "=" * 50)
        print("✅ PRECOMPUTATION COMPLETED!")
        print("=" * 50)
        
        # Test the cache loader
        print("\n🧪 Testing the cache loader...")
        from gradcam_cache_loader import test_cache_loader
        test_cache_loader(args.cache_dir)
        
        print("\n" + "=" * 70)
        print("🎉 ALL DONE! Your Grad-CAM cache is ready for training!")
        print("=" * 70)
        
        print("\n💡 NEXT STEPS:")
        print("1. Use the cache in your training loop:")
        print("   ```python")
        print("   from data.gradcam_cache_loader import create_cache_loader")
        print("   cache_loader = create_cache_loader('data/gradcam_cache')")
        print("   ```")
        print("")
        print("2. Load teacher heatmaps during training:")
        print("   ```python")
        print("   teacher_heatmaps = cache_loader.load_heatmaps_batch(")
        print("       'train', ['densenet', 'vgg16', 'resnet'], indices, labels)")
        print("   ```")
        print("")
        print("3. Compute adaptive teacher weights:")
        print("   ```python")
        print("   weights = cache_loader.get_adaptive_teacher_weights(")
        print("       student_heatmaps, teacher_heatmaps, temperature=0.5)")
        print("   ```")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the ECS189G/data directory and all dependencies are installed")
    except Exception as e:
        print(f"❌ Error during precomputation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 