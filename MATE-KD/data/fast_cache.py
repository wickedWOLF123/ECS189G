"""
Fast Memory-Resident Cache for Teacher Artifacts
Preloads everything into GPU memory at startup
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import glob

class FastTeacherCache:
    """Memory-resident cache for teacher logits and Grad-CAM maps"""
    
    def __init__(self, cache_dir: str, device: str = 'cuda'):
        self.cache_dir = Path(cache_dir)
        self.device = device
        
        # Memory storage: teacher_name -> {logits: tensor, heatmaps: tensor}
        self.teacher_artifacts = {}
        
        # Available teachers
        self.available_teachers = self._discover_teachers()
        
        print(f"🔥 FastTeacherCache initializing...")
        print(f"📁 Cache dir: {cache_dir}")
        print(f"🎯 Available teachers: {self.available_teachers}")
        
        # Preload everything
        self._preload_all_artifacts()
        
    def _discover_teachers(self) -> List[str]:
        """Discover available teachers from cache directory"""
        teachers = []
        train_dir = self.cache_dir / 'train'
        if train_dir.exists():
            for teacher_dir in train_dir.iterdir():
                if teacher_dir.is_dir():
                    teachers.append(teacher_dir.name)
        return sorted(teachers)
    
    def _preload_all_artifacts(self):
        """Preload all teacher artifacts into GPU memory"""
        total_memory = 0
        
        for teacher_name in self.available_teachers:
            print(f"  📥 Loading {teacher_name}...")
            
            # Load heatmaps
            heatmaps = self._load_teacher_heatmaps(teacher_name)
            
            # Store in memory
            self.teacher_artifacts[teacher_name] = {
                'heatmaps': heatmaps
            }
            
            # Calculate memory usage
            if heatmaps is not None:
                memory_mb = heatmaps.numel() * heatmaps.element_size() / (1024 * 1024)
                total_memory += memory_mb
                print(f"    💾 Heatmaps: {heatmaps.shape} ({memory_mb:.1f} MB)")
        
        print(f"🚀 Cache loaded: {total_memory:.1f} MB total")
    
    def _load_teacher_heatmaps(self, teacher_name: str) -> Optional[torch.Tensor]:
        """Load all heatmaps for a teacher into a single tensor"""
        teacher_dir = self.cache_dir / 'train' / teacher_name
        
        if not teacher_dir.exists():
            print(f"⚠️  Teacher directory not found: {teacher_dir}")
            return None
        
        # Find all cache files using glob pattern
        cache_files = list(teacher_dir.glob("idx_*_class_*.npz"))
        
        if not cache_files:
            print(f"⚠️  No cache files found for {teacher_name}")
            return None
        
        print(f"    📊 Found {len(cache_files)} cache files")
        
        # Create tensor for 50K samples (CIFAR-10 train set size)
        # Initialize with zeros, fill what we have
        heatmaps = torch.zeros(50000, 32, 32, device=self.device, dtype=torch.float16)
        loaded_count = 0
        
        for cache_file in cache_files:
            try:
                # Extract index from filename: idx_12345_class_7.npz
                filename = cache_file.name
                if filename.startswith('idx_') and '_class_' in filename:
                    idx_str = filename.split('_')[1]
                    idx = int(idx_str)
                    
                    if 0 <= idx < 50000:
                        # Load heatmap
                        data = np.load(cache_file)
                        heatmap = torch.from_numpy(data['heatmap'].astype(np.float16))
                        heatmaps[idx] = heatmap.squeeze().to(self.device)
                        loaded_count += 1
                        
            except Exception as e:
                print(f"⚠️  Error loading {cache_file}: {e}")
                continue
        
        print(f"    ✅ Loaded {loaded_count}/{len(cache_files)} heatmaps")
        return heatmaps
    
    def get_teacher_heatmaps_batch(self, teacher_name: str, 
                                 indices: torch.Tensor) -> torch.Tensor:
        """Get teacher heatmaps for a batch of indices"""
        if teacher_name not in self.teacher_artifacts:
            # Return zeros if teacher not available
            batch_size = len(indices)
            return torch.zeros(batch_size, 32, 32, device=self.device)
        
        heatmaps = self.teacher_artifacts[teacher_name]['heatmaps']
        if heatmaps is None:
            # Return zeros if heatmaps failed to load
            batch_size = len(indices)
            return torch.zeros(batch_size, 32, 32, device=self.device)
        
        return heatmaps[indices].float()  # Direct indexing - very fast! Convert to float32
    
    def get_all_teacher_heatmaps_batch(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get heatmaps for all teachers for a batch"""
        result = {}
        for teacher_name in self.available_teachers:
            # Only include teachers that actually have valid cache data
            if (teacher_name in self.teacher_artifacts and 
                self.teacher_artifacts[teacher_name]['heatmaps'] is not None):
                result[teacher_name] = self.get_teacher_heatmaps_batch(teacher_name, indices)
        return result
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'num_teachers': len(self.available_teachers),
            'teachers': self.available_teachers,
            'memory_usage_mb': 0
        }
        
        for teacher_name, artifacts in self.teacher_artifacts.items():
            if 'heatmaps' in artifacts and artifacts['heatmaps'] is not None:
                tensor = artifacts['heatmaps']
                memory_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
                stats['memory_usage_mb'] += memory_mb
        
        return stats 