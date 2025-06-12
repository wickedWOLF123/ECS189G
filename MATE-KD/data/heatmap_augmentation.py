"""
Heatmap Augmentation Utilities

This module provides functions to apply geometric transformations to cached
Grad-CAM heatmaps, ensuring that augmentation strength is maintained while
using pre-computed attribution maps.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, Optional, Union

def apply_horizontal_flip(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Apply horizontal flip to heatmap
    
    Args:
        heatmap: Input heatmap [H, W] or [1, H, W]
    
    Returns:
        Horizontally flipped heatmap
    """
    return torch.flip(heatmap, dims=[-1])

def apply_vertical_flip(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Apply vertical flip to heatmap
    
    Args:
        heatmap: Input heatmap [H, W] or [1, H, W]
    
    Returns:
        Vertically flipped heatmap
    """
    return torch.flip(heatmap, dims=[-2])

def apply_rotation(heatmap: torch.Tensor, angle: float, 
                  interpolation_mode: str = 'bilinear') -> torch.Tensor:
    """
    Apply rotation to heatmap
    
    Args:
        heatmap: Input heatmap [H, W] or [1, H, W]  
        angle: Rotation angle in degrees
        interpolation_mode: Interpolation mode for rotation
    
    Returns:
        Rotated heatmap
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0)  # Add channel dimension
    
    # Convert to PIL format range and apply rotation
    heatmap_pil = (heatmap * 255).byte()
    rotated = TF.rotate(heatmap_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
    
    # Convert back to float
    rotated_float = rotated.float() / 255.0
    
    return rotated_float.squeeze(0) if rotated_float.shape[0] == 1 else rotated_float

def apply_crop_and_resize(heatmap: torch.Tensor, 
                         crop_params: Tuple[int, int, int, int],
                         target_size: Tuple[int, int] = (32, 32)) -> torch.Tensor:
    """
    Apply random crop and resize to heatmap
    
    Args:
        heatmap: Input heatmap [H, W] or [1, H, W]
        crop_params: (top, left, height, width) for cropping
        target_size: Target size after resize (H, W)
    
    Returns:
        Cropped and resized heatmap
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)  # Add batch dim
    
    top, left, height, width = crop_params
    
    # Crop
    cropped = heatmap[:, :, top:top+height, left:left+width]
    
    # Resize
    resized = F.interpolate(cropped, size=target_size, mode='bilinear', align_corners=False)
    
    # Remove batch dimension and return
    return resized.squeeze(0).squeeze(0)

def apply_translation(heatmap: torch.Tensor, 
                     translate_x: float, translate_y: float,
                     padding_mode: str = 'zeros') -> torch.Tensor:
    """
    Apply translation to heatmap using affine transformation
    
    Args:
        heatmap: Input heatmap [H, W] or [1, H, W]
        translate_x: Translation in x direction (fraction of width)
        translate_y: Translation in y direction (fraction of height)
        padding_mode: Padding mode for areas outside original bounds
    
    Returns:
        Translated heatmap
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)  # Add batch dim
    
    batch_size, channels, height, width = heatmap.shape
    
    # Create affine transformation matrix for translation
    # Translation matrix: [[1, 0, tx], [0, 1, ty]]
    tx = translate_x * width
    ty = translate_y * height
    
    theta = torch.tensor([
        [1, 0, 2 * tx / width],    # Normalize translation for grid_sample
        [0, 1, 2 * ty / height]
    ], dtype=heatmap.dtype, device=heatmap.device).unsqueeze(0)
    
    # Generate sampling grid
    grid = F.affine_grid(theta, heatmap.size(), align_corners=False)
    
    # Apply transformation
    translated = F.grid_sample(heatmap, grid, mode='bilinear', 
                              padding_mode=padding_mode, align_corners=False)
    
    # Remove batch dimension and return
    return translated.squeeze(0).squeeze(0)

def apply_augmentation_to_heatmap(heatmap: torch.Tensor, 
                                 augmentation_params: dict) -> torch.Tensor:
    """
    Apply a combination of augmentations to heatmap based on parameters
    
    Args:
        heatmap: Input heatmap [H, W]
        augmentation_params: Dictionary containing augmentation parameters
            Expected keys:
            - 'horizontal_flip': bool
            - 'vertical_flip': bool
            - 'rotation': float (degrees)
            - 'crop_params': tuple (top, left, height, width)
            - 'translate': tuple (tx, ty) as fractions
    
    Returns:
        Augmented heatmap [H, W]
    """
    result = heatmap.clone()
    
    # Apply horizontal flip
    if augmentation_params.get('horizontal_flip', False):
        result = apply_horizontal_flip(result)
    
    # Apply vertical flip
    if augmentation_params.get('vertical_flip', False):
        result = apply_vertical_flip(result)
    
    # Apply rotation
    rotation_angle = augmentation_params.get('rotation', 0)
    if rotation_angle != 0:
        result = apply_rotation(result, rotation_angle)
    
    # Apply translation
    translate_params = augmentation_params.get('translate')
    if translate_params is not None:
        tx, ty = translate_params
        if tx != 0 or ty != 0:
            result = apply_translation(result, tx, ty)
    
    # Apply crop and resize
    crop_params = augmentation_params.get('crop_params')
    if crop_params is not None:
        result = apply_crop_and_resize(result, crop_params, target_size=(32, 32))
    
    return result

def extract_augmentation_params_from_transforms(image_before: torch.Tensor,
                                               image_after: torch.Tensor) -> dict:
    """
    Extract augmentation parameters by comparing original and augmented images.
    This is a simplified version - in practice, you'd track the transforms directly.
    
    Args:
        image_before: Original image [C, H, W]
        image_after: Augmented image [C, H, W]
    
    Returns:
        Dictionary of detected augmentation parameters
    """
    # This is a placeholder - in the actual implementation, you would:
    # 1. Modify the DataLoader to track applied transformations
    # 2. Return the transformation parameters along with the augmented image
    # 3. Use those parameters to apply the same transforms to heatmaps
    
    params = {
        'horizontal_flip': False,
        'vertical_flip': False,
        'rotation': 0,
        'crop_params': None,
        'translate': (0, 0)
    }
    
    # Simple horizontal flip detection (very basic)
    if torch.sum(torch.abs(image_before - torch.flip(image_after, dims=[-1]))) < 0.1:
        params['horizontal_flip'] = True
    
    return params

class HeatmapAugmenter:
    """
    Class to handle heatmap augmentation consistently with image augmentation
    """
    
    def __init__(self, preserve_magnitude: bool = True):
        """
        Initialize heatmap augmenter
        
        Args:
            preserve_magnitude: Whether to preserve heatmap magnitude after augmentation
        """
        self.preserve_magnitude = preserve_magnitude
    
    def __call__(self, heatmap: torch.Tensor, augmentation_params: dict) -> torch.Tensor:
        """
        Apply augmentation to heatmap
        
        Args:
            heatmap: Input heatmap [H, W]
            augmentation_params: Augmentation parameters
        
        Returns:
            Augmented heatmap [H, W]
        """
        if self.preserve_magnitude:
            original_sum = torch.sum(heatmap)
        
        augmented = apply_augmentation_to_heatmap(heatmap, augmentation_params)
        
        if self.preserve_magnitude and torch.sum(augmented) > 0:
            # Rescale to preserve total attribution magnitude
            scale_factor = original_sum / torch.sum(augmented)
            augmented = augmented * scale_factor
        
        return augmented

def create_augmentation_aware_dataloader(original_dataloader, heatmap_cache_loader):
    """
    Create a wrapper that applies consistent augmentations to both images and heatmaps.
    This would integrate with your existing DataLoader structure.
    
    This is a conceptual implementation - the actual version would need to be
    integrated with your specific DataLoader and caching system.
    """
    # This would be implemented as part of the main training loop
    # where you load both images and cached heatmaps, apply the same
    # transformations to both, and return the consistent pair
    pass 