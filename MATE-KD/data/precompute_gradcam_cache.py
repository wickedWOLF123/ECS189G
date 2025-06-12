#!/usr/bin/env python3
"""
Precompute Grad-CAM Heatmaps for CIFAR-10 Teacher Models

This script precomputes Grad-CAM attribution maps for all CIFAR-10 images
using the three teacher models (DenseNet-BC, VGG-16, ResNet-110) and caches
them for efficient multi-teacher knowledge distillation training.

Usage:
    python precompute_gradcam_cache.py
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
from typing import Dict, List, Tuple
import json
import hashlib
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../student'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../explainers'))

from cifar_loaders import get_cifar10_loaders, CIFAR10_CLASSES
from student.zoo import load_teacher
from explainers.gradcam import GradCAM


def get_target_layer(model, arch_key: str):
    """
    Get the target layer for Grad-CAM based on architecture.
    
    CRITICAL: All layers chosen to output ~8×8 feature maps for spatial alignment!
    This ensures cosine similarity works properly for teacher weighting.
    """
    target_layer_getters = {
        # DenseNet: Use block3 instead of block4 → 8×8, 512 channels (vs 4×4, 1024)
        'densenet': lambda model: list(model.features.denseblock3.children())[-1].conv2,
        
        # VGG: Use earlier layer before final pooling → 8×8, 256 channels (vs 2×2, 512)  
        'vgg16': lambda model: model.features[26],  # Conv before pool4 → 8×8
        'vgg19': lambda model: model.features[34],  # Conv before pool4 → 8×8, 256 ch
        
        # ResNet: Keep layer3 → 8×8, 64 channels ✅ Already correct
        'resnet': lambda model: model.layer3[-1].conv2,  # Last conv in layer3 block
    }
    
    if arch_key not in target_layer_getters:
        raise ValueError(f"Unknown architecture: {arch_key}")
    
    return target_layer_getters[arch_key](model)


def set_inplace_relu(model, inplace: bool = False):
    """Set inplace parameter for all ReLU layers in the model"""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = inplace


def create_cache_directory_structure(base_dir: str = "gradcam_cache") -> Path:
    """Create organized directory structure for caching heatmaps"""
    base_path = Path(base_dir)
    
    # Create main directories
    for split in ['train', 'test']:
        for arch in ['densenet', 'vgg16', 'vgg19', 'resnet']:
            arch_path = base_path / split / arch
            arch_path.mkdir(parents=True, exist_ok=True)
    
    # Create metadata directory
    (base_path / 'metadata').mkdir(parents=True, exist_ok=True)
    
    return base_path


def save_heatmap_batch(heatmaps: torch.Tensor, indices: List[int], labels: List[int], 
                      split: str, arch: str, base_path: Path):
    """Save a batch of heatmaps to disk"""
    arch_path = base_path / split / arch
    
    for i, (heatmap, idx, label) in enumerate(zip(heatmaps, indices, labels)):
        # Convert to numpy and save as compressed file
        heatmap_np = heatmap.cpu().numpy().astype(np.float16)  # Use float16 to save space
        
        filename = f"idx_{idx:05d}_class_{label}.npz"
        filepath = arch_path / filename
        
        np.savez_compressed(filepath, 
                          heatmap=heatmap_np,
                          index=idx,
                          label=label)


def load_heatmap(filepath: Path) -> Tuple[torch.Tensor, int, int]:
    """Load a single heatmap from disk"""
    data = np.load(filepath)
    heatmap = torch.from_numpy(data['heatmap'].astype(np.float32))
    index = int(data['index'])
    label = int(data['label'])
    return heatmap, index, label


def compute_gradcam_for_model(model, gradcam, dataloader, arch_key: str, 
                            split: str, base_path: Path, device: str = 'cuda'):
    """Compute and cache Grad-CAM heatmaps for one model"""
    print(f"\n🔥 Computing Grad-CAM for {arch_key.upper()} on {split} set...")
    
    model.eval()
    total_processed = 0
    batch_size = dataloader.batch_size
    
    # Process without torch.no_grad() since we need gradients for Grad-CAM
    pbar = tqdm(dataloader, desc=f"Processing {arch_key}", 
               total=len(dataloader), unit='batch')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        batch_heatmaps = []
        batch_indices = []
        batch_labels = []
        
        # Process each image in the batch individually for Grad-CAM
        for i in range(images.shape[0]):
            single_image = images[i:i+1]  # Keep batch dimension
            single_label = labels[i].item()
            
            try:
                # Handle VGG in-place ReLU issues (same as your working code)
                is_vgg = 'vgg' in arch_key
                if is_vgg:
                    set_inplace_relu(model, inplace=False)
                
                # Compute Grad-CAM heatmap using GROUND TRUTH label for semantic consistency
                # This provides more stable, semantically consistent attribution maps
                heatmap = gradcam(single_image, single_label)
                
                # Restore VGG model
                if is_vgg:
                    set_inplace_relu(model, inplace=True)
                
                # Resize heatmap to input size (32x32 for CIFAR-10)
                if heatmap.shape[-1] != 32:
                    heatmap = torch.nn.functional.interpolate(
                        heatmap, size=(32, 32), mode='bilinear', align_corners=False
                    )
                
                batch_heatmaps.append(heatmap.squeeze(0))  # Remove batch dim
                batch_indices.append(total_processed + i)
                batch_labels.append(single_label)
                
            except Exception as e:
                print(f"Error processing image {total_processed + i}: {e}")
                # Create zero heatmap as fallback
                zero_heatmap = torch.zeros(1, 32, 32, device=device)
                batch_heatmaps.append(zero_heatmap)
                batch_indices.append(total_processed + i)
                batch_labels.append(single_label)
                
                # Restore VGG model in case of error
                if 'vgg' in arch_key:
                    set_inplace_relu(model, inplace=True)
        
        # Save batch to disk
        if batch_heatmaps:
            heatmaps_tensor = torch.stack(batch_heatmaps)
            save_heatmap_batch(heatmaps_tensor, batch_indices, batch_labels,
                             split, arch_key, base_path)
        
        total_processed += len(images)
        pbar.set_postfix({
            'Processed': total_processed,
            'Batch': f"{batch_idx+1}/{len(dataloader)}"
        })
    
    print(f"✅ Completed {arch_key.upper()}: {total_processed} images processed")
    return total_processed


def save_metadata(base_path: Path, model_configs: Dict, dataset_info: Dict, 
                 processing_stats: Dict):
    """Save metadata about the cached heatmaps"""
    metadata = {
        'created_at': datetime.now().isoformat(),
        'dataset': 'CIFAR-10',
        'dataset_info': dataset_info,
        'models': model_configs,
        'processing_stats': processing_stats,
        'cache_format': {
            'file_format': 'npz (numpy compressed)',
            'heatmap_dtype': 'float16',
            'heatmap_shape': '[1, 32, 32]',
            'naming_convention': 'idx_{index:05d}_class_{label}.npz'
        },
        'directory_structure': {
            'train/': 'Training set heatmaps',
            'test/': 'Test set heatmaps',
            'metadata/': 'Cache metadata and statistics'
        }
    }
    
    metadata_path = base_path / 'metadata' / 'cache_info.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {metadata_path}")


def compute_dataset_hash(dataloader) -> str:
    """Compute a hash of the dataset to verify cache consistency"""
    hasher = hashlib.md5()
    
    # Sample a few batches to create a representative hash
    sample_count = 0
    for images, labels in dataloader:
        if sample_count >= 1000:  # Sample first 1000 examples for hash
            break
        for i in range(min(10, len(images))):  # Sample 10 per batch
            img_bytes = images[i].numpy().tobytes()
            label_bytes = labels[i].numpy().tobytes()
            hasher.update(img_bytes + label_bytes)
            sample_count += 1
    
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Precompute Grad-CAM heatmaps for CIFAR-10')
    parser.add_argument('--config', type=str, default='../cfg.yaml',
                       help='Config file path')
    parser.add_argument('--cache-dir', type=str, default='data/gradcam_cache',
                       help='Directory to store cached heatmaps')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--models', nargs='+', 
                       choices=['densenet', 'vgg16', 'vgg19', 'resnet'],
                       default=['densenet', 'vgg16', 'resnet'],
                       help='Models to process')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"🚀 Starting Grad-CAM cache generation")
    print(f"📱 Device: {device}")
    print(f"📁 Cache directory: {args.cache_dir}")
    print(f"🎯 Models to process: {', '.join(args.models)}")
    
    # Create cache directory structure
    cache_path = create_cache_directory_structure(args.cache_dir)
    print(f"📂 Created cache structure at: {cache_path}")
    
    # Load CIFAR-10 datasets
    print("\n📊 Loading CIFAR-10 datasets...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        root='.',
        augment_train=False  # No augmentation for consistent heatmaps
    )
    
    dataset_info = {
        'train_samples': len(train_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'num_classes': 10,
        'class_names': CIFAR10_CLASSES,
        'dataset_hash': compute_dataset_hash(train_loader)
    }
    
    print(f"📈 Dataset loaded: {dataset_info['train_samples']} train, {dataset_info['test_samples']} test")
    
    # Model configurations for different architectures
    model_configs = {
        'densenet': {
            'name': 'DenseNet-121',
            'path': cfg['teacher_paths']['densenet'],
            'description': 'Dense connections between layers'
        },
        'vgg16': {
            'name': 'VGG-16',
            'path': cfg['teacher_paths']['vgg16'],
            'description': '16-layer VGG with batch normalization'
        },
        'vgg19': {
            'name': 'VGG-19', 
            'path': cfg['teacher_paths']['vgg19'],
            'description': '19-layer VGG with batch normalization'
        },
        'resnet': {
            'name': 'ResNet-110',
            'path': cfg['teacher_paths']['resnet'],
            'description': '110-layer ResNet with residual connections'
        }
    }
    
    processing_stats = {}
    
    # Process each model
    for arch_key in args.models:
        if arch_key not in model_configs:
            print(f"⚠️ Skipping unknown model: {arch_key}")
            continue
        
        config = model_configs[arch_key]
        model_path = config['path']
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            continue
        
        print(f"\n🔧 Loading {config['name']} from {model_path}")
        
        try:
            # Load model
            model = load_teacher(model_path, arch_key)
            model.to(device)
            model.eval()
            
            # Get target layer for Grad-CAM
            target_layer = get_target_layer(model, arch_key)
            gradcam = GradCAM(model, target_layer)
            
            print(f"🎯 Target layer: {type(target_layer).__name__}")
            
            # Initialize processing stats for this model
            processing_stats[arch_key] = {}
            
            # Process training set
            train_processed = compute_gradcam_for_model(
                model, gradcam, train_loader, arch_key, 'train', cache_path, device
            )
            processing_stats[arch_key]['train_processed'] = train_processed
            
            # Process test set
            test_processed = compute_gradcam_for_model(
                model, gradcam, test_loader, arch_key, 'test', cache_path, device
            )
            processing_stats[arch_key]['test_processed'] = test_processed
            
            # Clean up GPU memory
            del model, gradcam
            if device == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"✅ {config['name']} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error processing {config['name']}: {e}")
            processing_stats[arch_key] = {'error': str(e)}
    
    # Save metadata
    save_metadata(cache_path, model_configs, dataset_info, processing_stats)
    
    # Print summary
    print(f"\n🎉 GRAD-CAM CACHE GENERATION COMPLETE!")
    print(f"📁 Cache location: {cache_path.absolute()}")
    print(f"📊 Summary:")
    
    for arch_key, stats in processing_stats.items():
        if 'error' in stats:
            print(f"  ❌ {arch_key}: {stats['error']}")
        else:
            train_count = stats.get('train_processed', 0)
            test_count = stats.get('test_processed', 0)
            print(f"  ✅ {arch_key}: {train_count} train + {test_count} test = {train_count + test_count} total")
    
    print(f"\n💡 Usage: Load heatmaps using the cache loader utilities!")


if __name__ == "__main__":
    main() 