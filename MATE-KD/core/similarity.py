import torch
import torch.nn.functional as F

def _resize_to_match(a, b):
    """Resize tensor a to match the spatial dimensions of tensor b"""
    # Handle missing channel dimension in teacher cached heatmaps
    if len(a.shape) == 3 and len(b.shape) == 4:
        # Teacher cache is [B, H, W], need [B, 1, H, W]
        a = a.unsqueeze(1)
    elif len(b.shape) == 3 and len(a.shape) == 4:
        # Student is [B, 1, H, W], teacher cache is [B, H, W]
        b = b.unsqueeze(1)
    elif len(a.shape) == 3 and len(b.shape) == 3:
        # Both are 3D [B, H, W], add channel dimension to both
        a = a.unsqueeze(1)  # [B, 1, H, W]
        b = b.unsqueeze(1)  # [B, 1, H, W]
    
    # Now both should be 4D, check if resize is needed
    if a.shape[2:] != b.shape[2:]:
        # Resize a to match b's spatial dimensions
        target_size = b.shape[2:]
        a = F.interpolate(a, size=target_size, mode='bilinear', align_corners=False)
    
    # Return both tensors (b might have been modified too)
    return a, b

def cosine_flat(a, b):
    """Cosine similarity computation with automatic resizing"""
    a, b = _resize_to_match(a, b)
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)

def cosine_flat_clipped(a, b):
    """Cosine similarity with ReLU clipping to remove negative correlations."""
    a, b = _resize_to_match(a, b)
    cos_sim = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
    return torch.relu(cos_sim)  # Clip negatives to zero

def pearson_correlation(a, b):
    """Pearson correlation coefficient between flattened tensors.
    
    Args:
        a, b: Tensors of shape [B, C, H, W] or [B, H, W]
    
    Returns:
        correlation: Tensor of shape [B] with correlation coefficients in [-1, 1]
    """
    a, b = _resize_to_match(a, b)
    
    # Flatten spatial dimensions: [B, C, H, W] -> [B, C*H*W]
    a_flat = a.flatten(1)  # [B, N]
    b_flat = b.flatten(1)  # [B, N]
    
    # Compute means along feature dimension
    a_mean = a_flat.mean(dim=1, keepdim=True)  # [B, 1]
    b_mean = b_flat.mean(dim=1, keepdim=True)  # [B, 1]
    
    # Center the data (subtract means)
    a_centered = a_flat - a_mean  # [B, N]
    b_centered = b_flat - b_mean  # [B, N]
    
    # Compute numerator and denominators
    numerator = (a_centered * b_centered).sum(dim=1)  # [B]
    
    a_norm = torch.sqrt((a_centered ** 2).sum(dim=1) + 1e-8)  # [B]
    b_norm = torch.sqrt((b_centered ** 2).sum(dim=1) + 1e-8)  # [B]
    
    # Pearson correlation coefficient
    correlation = numerator / (a_norm * b_norm + 1e-8)  # [B]
    
    return correlation

def pearson_correlation_clipped(a, b):
    """Pearson correlation with ReLU clipping for positive-only similarities."""
    correlation = pearson_correlation(a, b)
    return torch.relu(correlation)  # Clip negatives to zero

def rescaled_similarity(a, b, similarity_fn=cosine_flat_clipped, max_normalize=True):
    """Enhanced similarity computation with rescaling for better teacher differentiation.
    
    Args:
        a, b: Input tensors
        similarity_fn: Function to compute similarity (cosine_flat_clipped, pearson_correlation_clipped)
        max_normalize: Whether to normalize by max value to [0, 1] range
    
    Returns:
        similarities: Rescaled similarity scores ready for softmax weighting
    """
    # Compute raw similarities
    similarities = similarity_fn(a, b)
    
    if max_normalize:
        # Normalize to [0, 1] by dividing by max
        max_sim = similarities.max(dim=0, keepdim=True).values
        similarities = similarities / (max_sim + 1e-6)
    
    return similarities 