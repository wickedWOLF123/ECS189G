import torch
import torch.nn.functional as F

def softmax_weights(sim_matrix, tau):
    """Original softmax weighting function"""
    # sim_matrix shape: (K, B)
    return torch.softmax(sim_matrix / tau, dim=0)  # over K

def enhanced_teacher_weighting(sim_matrix, tau, rescale=True, clip_negatives=True):
    """
    Enhanced teacher weighting with rescaling pipeline for better differentiation.
    
    Pipeline:
    1. Clip negative similarities (optional)
    2. Rescale similarities to [0, 1] by max-normalization  
    3. Apply softmax with low temperature τ
    
    This transforms similarities from [0, 0.05] to [0, 1.0], giving softmax
    real differences to work with instead of tiny variations.
    
    Args:
        sim_matrix: Similarity matrix [K, B] where K=teachers, B=batch
        tau: Temperature parameter (lower = more selective)
        rescale: Whether to apply max-normalization rescaling
        clip_negatives: Whether to clip negative similarities to zero
        
    Returns:
        weights: Teacher weights [K, B] summing to 1 along teacher dimension
    """
    similarities = sim_matrix.clone()
    
    if clip_negatives:
        similarities = torch.relu(similarities)
    
    if rescale:
        # Rescale to [0, 1] range by max-normalization
        max_sim = similarities.max(dim=0, keepdim=True).values  # [1, B]
        similarities = similarities / (max_sim + 1e-6)  # [K, B] in [0, 1]
    
    # Apply softmax with temperature
    weights = torch.softmax(similarities / tau, dim=0)  # [K, B]
    
    return weights

def compute_tau_schedule(epoch, total_epochs, tau_start=0.3, tau_min=0.05, warmup_epochs=50):
    """Adaptive temperature scheduling for teacher weighting.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        tau_start: Initial temperature (broader selection)
        tau_min: Minimum temperature (sharper selection)  
        warmup_epochs: Epochs over which to cool temperature
        
    Returns:
        tau: Current temperature value
    """
    if epoch >= warmup_epochs:
        return tau_min
    
    # Linear decay from tau_start to tau_min over warmup_epochs
    progress = epoch / warmup_epochs  # [0, 1]
    tau = tau_start * (1 - progress) + tau_min * progress
    
    return max(tau, tau_min)

def compute_teacher_weights_enhanced(student_attribution, teacher_attributions, 
                                   teacher_names, epoch=0, total_epochs=200,
                                   similarity_fn=None, use_tau_schedule=True,
                                   tau_start=0.3, tau_min=0.05):
    """
    Comprehensive teacher weighting with all enhancements:
    - Choice of similarity functions (cosine, Pearson)
    - Rescaling pipeline for better differentiation
    - Adaptive τ-schedule for training stability
    
    Args:
        student_attribution: Student Grad-CAM [B, H, W]
        teacher_attributions: Dict of teacher Grad-CAMs {name: [B, H, W]}
        teacher_names: List of teacher names
        epoch: Current training epoch
        total_epochs: Total training epochs
        similarity_fn: Similarity function to use (default: cosine_flat_clipped)
        use_tau_schedule: Whether to use adaptive temperature scheduling
        tau_start: Initial temperature (broader selection)
        tau_min: Minimum temperature (sharper selection)
        
    Returns:
        weights: Dict of teacher weights {name: [B]} 
        stats: Dict with weighting statistics
    """
    from .similarity import cosine_flat_clipped, rescaled_similarity
    
    if similarity_fn is None:
        similarity_fn = cosine_flat_clipped
    
    B = student_attribution.shape[0]
    device = student_attribution.device
    
    # Compute similarities for each teacher
    similarities = []
    for name in teacher_names:
        teacher_attr = teacher_attributions[name]
        
        # Use enhanced similarity computation with rescaling
        sim = rescaled_similarity(
            student_attribution, 
            teacher_attr,
            similarity_fn=similarity_fn,
            max_normalize=True  # Key rescaling step
        )
        similarities.append(sim)
    
    # Stack similarities: [K, B]
    sim_matrix = torch.stack(similarities, dim=0)
    
    # Compute adaptive temperature
    if use_tau_schedule:
        tau = compute_tau_schedule(epoch, total_epochs, tau_start=tau_start, tau_min=tau_min)
    else:
        tau = 0.1  # Fixed low temperature
    
    # Enhanced teacher weighting with rescaling
    weights_matrix = enhanced_teacher_weighting(
        sim_matrix, 
        tau=tau, 
        rescale=True,
        clip_negatives=False
    )
    
    # Convert to dictionary format
    weights = {}
    for i, name in enumerate(teacher_names):
        weights[name] = weights_matrix[i]  # [B]
    
    # Compute dominance stats
    dominant_teacher_counts = _compute_dominance_stats(weights_matrix, teacher_names)
    dominance_ratio = max(dominant_teacher_counts.values()) / sum(dominant_teacher_counts.values())
    
    # Compute statistics
    stats = {
        'tau': tau,
        'epoch': epoch,
        'mean_similarity': sim_matrix.mean().item(),
        'max_similarity': sim_matrix.max().item(),
        'min_similarity': sim_matrix.min().item(),
        'similarity_std': sim_matrix.std().item(),
        'entropy': _compute_weight_entropy(weights_matrix),
        'dominance': dominance_ratio,
        'teacher_weight_entropy': _compute_weight_entropy(weights_matrix),
        'dominant_teacher_counts': dominant_teacher_counts
    }
    
    return weights, stats

def _compute_weight_entropy(weights_matrix):
    """Compute entropy of teacher weight distribution (higher = more uniform)"""
    # weights_matrix: [K, B]
    mean_weights = weights_matrix.mean(dim=1)  # [K] - average weight per teacher
    entropy = -(mean_weights * torch.log(mean_weights + 1e-8)).sum()
    return entropy.item()

def _compute_dominance_stats(weights_matrix, teacher_names):
    """Compute which teacher dominates each sample"""
    # weights_matrix: [K, B]
    dominant_indices = weights_matrix.argmax(dim=0)  # [B]
    
    dominance_counts = {}
    for i, name in enumerate(teacher_names):
        count = (dominant_indices == i).sum().item()
        dominance_counts[name] = count
    
    return dominance_counts 