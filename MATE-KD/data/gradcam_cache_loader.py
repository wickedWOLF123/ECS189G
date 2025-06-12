"""
Grad-CAM Cache Loader Utilities

This module provides efficient utilities for loading precomputed Grad-CAM heatmaps
during multi-teacher knowledge distillation training. It includes functions for
batch loading, cosine similarity computation, and memory-efficient heatmap access.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings


class GradCAMCacheLoader:
    """
    Efficient loader for cached Grad-CAM heatmaps with memory management
    and batch processing capabilities.
    """
    
    def __init__(self, cache_dir: str, device: str = 'cuda'):
        """
        Initialize the cache loader
        
        Args:
            cache_dir: Directory containing cached heatmaps
            device: Device to load tensors to ('cuda' or 'cpu')
        """
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.metadata = self._load_metadata()
        self.available_models = self._get_available_models()
        
        # Cache for frequently accessed heatmaps (LRU-style)
        self._memory_cache = {}
        self._cache_access_order = []
        self.max_cache_size = 1000  # Max number of heatmaps to keep in memory
        
        print(f"📋 Loaded Grad-CAM cache from: {self.cache_dir}")
        print(f"🎯 Available models: {', '.join(self.available_models)}")
        if self.metadata:
            stats = self.metadata.get('processing_stats', {})
            for model, stat in stats.items():
                if 'error' not in stat:
                    train_count = stat.get('train_processed', 0)
                    test_count = stat.get('test_processed', 0)
                    print(f"  📊 {model}: {train_count} train + {test_count} test")
    
    def _load_metadata(self) -> Optional[Dict]:
        """Load cache metadata"""
        metadata_path = self.cache_dir / 'metadata' / 'cache_info.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            warnings.warn(f"No metadata found at {metadata_path}")
            return None
    
    def _get_available_models(self) -> List[str]:
        """Get list of available model architectures"""
        models = []
        for split in ['train', 'test']:
            split_path = self.cache_dir / split
            if split_path.exists():
                for model_dir in split_path.iterdir():
                    if model_dir.is_dir() and model_dir.name not in models:
                        models.append(model_dir.name)
        return sorted(models)
    
    def _get_cache_key(self, split: str, model: str, index: int) -> str:
        """Generate cache key for a heatmap"""
        return f"{split}_{model}_{index}"
    
    def _manage_memory_cache(self, key: str):
        """Manage memory cache size using LRU eviction"""
        if key in self._cache_access_order:
            self._cache_access_order.remove(key)
        self._cache_access_order.append(key)
        
        # Evict oldest entries if cache is full
        while len(self._memory_cache) > self.max_cache_size:
            oldest_key = self._cache_access_order.pop(0)
            if oldest_key in self._memory_cache:
                del self._memory_cache[oldest_key]
    
    def load_heatmap(self, split: str, model: str, index: int, label: int) -> torch.Tensor:
        """
        Load a single heatmap from cache
        
        Args:
            split: 'train' or 'test'
            model: Model architecture ('densenet', 'vgg16', 'vgg19', 'resnet')
            index: Sample index in dataset
            label: Ground truth label
            
        Returns:
            torch.Tensor: Heatmap tensor of shape (1, 32, 32)
        """
        cache_key = self._get_cache_key(split, model, index)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            self._manage_memory_cache(cache_key)
            return self._memory_cache[cache_key].clone()
        
        # Load from disk
        filename = f"idx_{index:05d}_class_{label}.npz"
        filepath = self.cache_dir / split / model / filename
        
        if not filepath.exists():
            # warnings.warn(f"Heatmap not found: {filepath}")  # Disable warnings for now
            # Return zero heatmap as fallback
            return torch.zeros(1, 32, 32, device=self.device)
        
        try:
            data = np.load(filepath)
            heatmap = torch.from_numpy(data['heatmap'].astype(np.float32))
            heatmap = heatmap.to(self.device)
            
            # Add to memory cache
            self._memory_cache[cache_key] = heatmap.clone()
            self._manage_memory_cache(cache_key)
            
            return heatmap
            
        except Exception as e:
            warnings.warn(f"Error loading heatmap {filepath}: {e}")
            return torch.zeros(1, 32, 32, device=self.device)
    
    def load_heatmaps_batch(self, split: str, models: List[str], 
                           indices: List[int], labels: List[int]) -> Dict[str, torch.Tensor]:
        """
        Load heatmaps for multiple models and samples in batch
        
        Args:
            split: 'train' or 'test'
            models: List of model architectures
            indices: List of sample indices
            labels: List of ground truth labels
            
        Returns:
            Dict mapping model names to batched heatmap tensors of shape (batch_size, 1, 32, 32)
        """
        batch_size = len(indices)
        result = {}
        
        for model in models:
            if model not in self.available_models:
                warnings.warn(f"Model {model} not available in cache")
                continue
            
            heatmaps = []
            for idx, label in zip(indices, labels):
                heatmap = self.load_heatmap(split, model, idx, label)
                heatmaps.append(heatmap)
            
            if heatmaps:
                result[model] = torch.stack(heatmaps, dim=0)
        
        return result
    
    def compute_cosine_similarities(self, student_heatmaps: torch.Tensor, 
                                  teacher_heatmaps: Dict[str, torch.Tensor],
                                  normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute cosine similarities between student and teacher heatmaps
        
        Args:
            student_heatmaps: Student heatmaps of shape (batch_size, 1, 32, 32)
            teacher_heatmaps: Dict of teacher heatmaps, each of shape (batch_size, 1, 32, 32)
            normalize: Whether to normalize heatmaps before computing similarity
            
        Returns:
            Dict mapping teacher names to similarity scores of shape (batch_size,)
        """
        if normalize:
            # Normalize student heatmaps
            student_flat = student_heatmaps.view(student_heatmaps.shape[0], -1)
            student_norm = F.normalize(student_flat, p=2, dim=1)
        else:
            student_norm = student_heatmaps.view(student_heatmaps.shape[0], -1)
        
        similarities = {}
        
        for teacher_name, teacher_maps in teacher_heatmaps.items():
            if normalize:
                # Normalize teacher heatmaps
                teacher_flat = teacher_maps.view(teacher_maps.shape[0], -1)
                teacher_norm = F.normalize(teacher_flat, p=2, dim=1)
            else:
                teacher_norm = teacher_maps.view(teacher_maps.shape[0], -1)
            
            # Compute cosine similarity
            similarity = torch.sum(student_norm * teacher_norm, dim=1)
            similarities[teacher_name] = similarity
        
        return similarities
    
    def compute_teacher_weights(self, similarities: Dict[str, torch.Tensor], 
                              temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Convert cosine similarities to teacher weights using temperature-controlled softmax
        
        Args:
            similarities: Dict of similarity scores for each teacher
            temperature: Temperature parameter for softmax (lower = more focused)
            
        Returns:
            Dict mapping teacher names to normalized weights of shape (batch_size,)
        """
        if not similarities:
            return {}
        
        # Stack similarities and apply temperature
        teacher_names = list(similarities.keys())
        sim_tensor = torch.stack([similarities[name] for name in teacher_names], dim=1)
        
        # Apply temperature and softmax
        weights_tensor = F.softmax(sim_tensor / temperature, dim=1)
        
        # Convert back to dict
        weights = {}
        for i, name in enumerate(teacher_names):
            weights[name] = weights_tensor[:, i]
        
        return weights
    
    def get_adaptive_teacher_weights(self, student_heatmaps: torch.Tensor,
                                   teacher_heatmaps: Dict[str, torch.Tensor],
                                   temperature: float = 1.0,
                                   normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive teacher weights based on heatmap similarities (end-to-end)
        
        Args:
            student_heatmaps: Student heatmaps of shape (batch_size, 1, 32, 32)
            teacher_heatmaps: Dict of teacher heatmaps
            temperature: Softmax temperature
            normalize: Whether to normalize heatmaps
            
        Returns:
            Dict mapping teacher names to weights of shape (batch_size,)
        """
        similarities = self.compute_cosine_similarities(
            student_heatmaps, teacher_heatmaps, normalize=normalize
        )
        return self.compute_teacher_weights(similarities, temperature=temperature)
    
    def clear_memory_cache(self):
        """Clear the memory cache to free up memory"""
        self._memory_cache.clear()
        self._cache_access_order.clear()
        print("🧹 Memory cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache"""
        stats = {
            'cache_dir': str(self.cache_dir),
            'available_models': self.available_models,
            'memory_cache_size': len(self._memory_cache),
            'max_cache_size': self.max_cache_size,
        }
        
        if self.metadata:
            stats.update({
                'created_at': self.metadata.get('created_at'),
                'dataset_info': self.metadata.get('dataset_info'),
                'processing_stats': self.metadata.get('processing_stats')
            })
        
        return stats


def create_cache_loader(cache_dir: str, device: str = 'cuda') -> GradCAMCacheLoader:
    """
    Convenience function to create a cache loader
    
    Args:
        cache_dir: Directory containing cached heatmaps
        device: Device to load tensors to
        
    Returns:
        GradCAMCacheLoader instance
    """
    return GradCAMCacheLoader(cache_dir, device)


def batch_load_teacher_heatmaps(cache_loader: GradCAMCacheLoader,
                               split: str, indices: List[int], labels: List[int],
                               teacher_models: List[str] = None) -> Dict[str, torch.Tensor]:
    """
    Convenience function to load teacher heatmaps for a batch
    
    Args:
        cache_loader: Cache loader instance
        split: 'train' or 'test'
        indices: List of sample indices
        labels: List of ground truth labels
        teacher_models: List of teacher models to load (None = all available)
        
    Returns:
        Dict mapping teacher names to batched heatmap tensors
    """
    if teacher_models is None:
        teacher_models = cache_loader.available_models
    
    return cache_loader.load_heatmaps_batch(split, teacher_models, indices, labels)


# Example usage and testing functions
def test_cache_loader(cache_dir: str = "gradcam_cache"):
    """Test the cache loader functionality"""
    print("🧪 Testing Grad-CAM Cache Loader...")
    
    # Create loader
    loader = create_cache_loader(cache_dir)
    
    # Test single heatmap loading
    print("\n📄 Testing single heatmap loading...")
    heatmap = loader.load_heatmap('test', 'densenet', 0, 0)
    print(f"Loaded heatmap shape: {heatmap.shape}")
    
    # Test batch loading
    print("\n📦 Testing batch loading...")
    indices = [0, 1, 2, 3, 4]
    labels = [0, 1, 2, 3, 4]
    batch_heatmaps = loader.load_heatmaps_batch('test', ['densenet', 'resnet'], indices, labels)
    
    for model, heatmaps in batch_heatmaps.items():
        print(f"{model} batch shape: {heatmaps.shape}")
    
    # Test similarity computation
    if len(batch_heatmaps) >= 2:
        print("\n🔍 Testing similarity computation...")
        models = list(batch_heatmaps.keys())
        student_maps = batch_heatmaps[models[0]]  # Use first as "student"
        teacher_maps = {models[1]: batch_heatmaps[models[1]]}  # Use second as "teacher"
        
        similarities = loader.compute_cosine_similarities(student_maps, teacher_maps)
        weights = loader.compute_teacher_weights(similarities, temperature=0.5)
        
        for teacher, sims in similarities.items():
            print(f"Similarities with {teacher}: {sims[:3].tolist()}")
        for teacher, w in weights.items():
            print(f"Weights for {teacher}: {w[:3].tolist()}")
    
    # Print cache stats
    print("\n📊 Cache Statistics:")
    stats = loader.get_cache_stats()
    for key, value in stats.items():
        if key not in ['dataset_info', 'processing_stats']:
            print(f"  {key}: {value}")
    
    print("✅ Cache loader test completed!")


if __name__ == "__main__":
    # Run test if executed directly
    test_cache_loader() 