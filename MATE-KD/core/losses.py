import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_soft, T):
    """Original single-teacher KD loss"""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        teacher_soft,
        reduction='batchmean'
    ) * (T**2)

def multi_teacher_distillation_loss(s_logits, t_logits_list, teacher_weights, y, T_kd, ce_weight):
    """Compute weighted multi-teacher distillation loss with instance-wise teacher weighting
    
    Args:
        s_logits: Student logits [B, C]
        t_logits_list: List of teacher logits [B, C] for each teacher
        teacher_weights: Dict of teacher weights [B] per teacher name
        y: Ground truth labels [B]
        T_kd: KD temperature
        ce_weight: Cross entropy weight (alpha)
    
    Returns:
        tuple: (total_loss, loss_ce, loss_kd, loss_stats)
    """
    B, C = s_logits.shape
    device = s_logits.device
    
    # 1. Cross entropy loss with ground truth
    loss_ce = F.cross_entropy(s_logits, y)
    
    # 2. Weighted multi-teacher KL divergence
    teacher_names = list(teacher_weights.keys())
    
    # Compute soft targets for each teacher
    teacher_soft_targets = []
    for i, teacher_name in enumerate(teacher_names):
        soft_target = F.softmax(t_logits_list[i] / T_kd, dim=1)
        teacher_soft_targets.append(soft_target)
    
    # Stack teacher soft targets: [num_teachers, B, C]
    teacher_soft_stack = torch.stack(teacher_soft_targets, dim=0)
    
    # Stack teacher weights: [num_teachers, B]
    weight_values = torch.stack([teacher_weights[name] for name in teacher_names], dim=0)
    
    # Normalize weights per sample (they should already be normalized from softmax, but ensure)
    weight_values = weight_values / (weight_values.sum(dim=0, keepdim=True) + 1e-8)
    
    # Compute per-sample weighted ensemble of soft targets
    # weight_values: [num_teachers, B] -> [num_teachers, B, 1]
    # teacher_soft_stack: [num_teachers, B, C]
    weighted_soft_targets = (weight_values.unsqueeze(-1) * teacher_soft_stack).sum(dim=0)  # [B, C]
    
    # 3. KL divergence loss
    student_log_probs = F.log_softmax(s_logits / T_kd, dim=1)
    loss_kd = F.kl_div(student_log_probs, weighted_soft_targets, reduction='batchmean') * (T_kd ** 2)
    
    # 4. Combined loss
    total_loss = ce_weight * loss_ce + (1 - ce_weight) * loss_kd
    
    # 5. Compute loss statistics for monitoring
    loss_stats = {
        'loss_total': total_loss.item(),
        'loss_ce': loss_ce.item(),
        'loss_kd': loss_kd.item(),
        'teacher_weight_stats': {
            name: {
                'mean': teacher_weights[name].mean().item(),
                'std': teacher_weights[name].std().item(),
                'max': teacher_weights[name].max().item(),
                'min': teacher_weights[name].min().item()
            } for name in teacher_names
        }
    }
    
    return total_loss, loss_ce, loss_kd, loss_stats

def enhanced_multi_teacher_distillation_loss(s_logits, t_logits_list, teacher_weights, y, 
                                           s_attribution, t_attributions_list, 
                                           T_kd, ce_weight, map_weight):
    """Enhanced multi-teacher distillation loss with attribution map alignment
    
    Loss = α * L_CE + (1-α) * L_KD + β * L_map
    
    Where:
    - L_CE: Cross entropy loss with ground truth
    - L_KD: Weighted KL divergence with teacher ensemble 
    - L_map: MSE between student attribution and weighted teacher attribution ensemble
    
    Args:
        s_logits: Student logits [B, C]
        t_logits_list: List of teacher logits [B, C] for each teacher
        teacher_weights: Dict of teacher weights [B] per teacher name
        y: Ground truth labels [B]
        s_attribution: Student attribution map [B, H, W]
        t_attributions_list: List of teacher attribution maps [B, H, W] for each teacher
        T_kd: KD temperature
        ce_weight: Cross entropy weight (alpha)
        map_weight: Attribution map alignment weight (beta)
    
    Returns:
        tuple: (total_loss, loss_ce, loss_kd, loss_map, loss_stats)
    """
    B, C = s_logits.shape
    device = s_logits.device
    
    # 1. Cross entropy loss with ground truth
    loss_ce = F.cross_entropy(s_logits, y)
    
    # 2. Weighted multi-teacher KL divergence (existing logic)
    teacher_names = list(teacher_weights.keys())
    
    # Compute soft targets for each teacher
    teacher_soft_targets = []
    for i, teacher_name in enumerate(teacher_names):
        soft_target = F.softmax(t_logits_list[i] / T_kd, dim=1)
        teacher_soft_targets.append(soft_target)
    
    # Stack teacher soft targets: [num_teachers, B, C]
    teacher_soft_stack = torch.stack(teacher_soft_targets, dim=0)
    
    # Stack teacher weights: [num_teachers, B]
    weight_values = torch.stack([teacher_weights[name] for name in teacher_names], dim=0)
    
    # Normalize weights per sample (they should already be normalized from softmax, but ensure)
    weight_values = weight_values / (weight_values.sum(dim=0, keepdim=True) + 1e-8)
    
    # Compute per-sample weighted ensemble of soft targets
    # weight_values: [num_teachers, B] -> [num_teachers, B, 1]
    # teacher_soft_stack: [num_teachers, B, C]
    weighted_soft_targets = (weight_values.unsqueeze(-1) * teacher_soft_stack).sum(dim=0)  # [B, C]
    
    # 3. KL divergence loss
    student_log_probs = F.log_softmax(s_logits / T_kd, dim=1)
    loss_kd = F.kl_div(student_log_probs, weighted_soft_targets, reduction='batchmean') * (T_kd ** 2)
    
    # 4. Attribution map alignment loss with max-normalization
    # Stack teacher attribution maps: [num_teachers, B, H, W]
    teacher_attribution_stack = torch.stack(t_attributions_list, dim=0)
    
    # Compute weighted ensemble of teacher attribution maps
    # weight_values: [num_teachers, B] -> [num_teachers, B, 1, 1] for broadcasting
    weight_values_spatial = weight_values.unsqueeze(-1).unsqueeze(-1)
    
    # Weighted attribution ensemble: [B, H, W]
    weighted_attribution_target = (weight_values_spatial * teacher_attribution_stack).sum(dim=0)
    
    # First, resize to match dimensions if needed
    if s_attribution.shape != weighted_attribution_target.shape:
        # Resize weighted target to match student dimensions
        target_size = s_attribution.shape[2:] if s_attribution.dim() == 4 else s_attribution.shape[1:]
        if weighted_attribution_target.dim() == 3:
            # Add channel dimension: [B, H, W] -> [B, 1, H, W]
            weighted_attribution_target = weighted_attribution_target.unsqueeze(1)
        if s_attribution.dim() == 3:
            # Add channel dimension: [B, H, W] -> [B, 1, H, W]
            s_attribution = s_attribution.unsqueeze(1)
        
        # Resize to match student dimensions
        weighted_attribution_target = F.interpolate(
            weighted_attribution_target, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Remove channel dimension if student doesn't have it
        if s_attribution.dim() == 4 and s_attribution.shape[1] == 1:
            s_attribution = s_attribution.squeeze(1)
            weighted_attribution_target = weighted_attribution_target.squeeze(1)
    
    # Max-normalization instead of L2-normalization
    # This provides better loss visibility and meaningful gradients
    
    # Max-normalize student attribution: divide by global max per sample
    student_map_normalized = s_attribution / (
        s_attribution.view(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1) + 1e-6
    )
    
    # Max-normalize target attribution: divide by global max per sample
    target_map_normalized = weighted_attribution_target / (
        weighted_attribution_target.view(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1) + 1e-6
    )
    
    # MSE loss between max-normalized attribution maps
    loss_map = F.mse_loss(student_map_normalized, target_map_normalized, reduction='mean')
    
    # 5. Combined loss
    total_loss = ce_weight * loss_ce + (1 - ce_weight) * loss_kd + map_weight * loss_map
    
    # 6. Compute loss statistics for monitoring
    loss_stats = {
        'loss_total': total_loss.item(),
        'loss_ce': loss_ce.item(),
        'loss_kd': loss_kd.item(),
        'loss_map': loss_map.item(),
        'teacher_weight_stats': {
            name: {
                'mean': teacher_weights[name].mean().item(),
                'std': teacher_weights[name].std().item(),
                'max': teacher_weights[name].max().item(),
                'min': teacher_weights[name].min().item()
            } for name in teacher_names
        },
        'attribution_stats': {
            'student_attribution_mean': s_attribution.mean().item(),
            'student_attribution_std': s_attribution.std().item(),
            'target_attribution_mean': weighted_attribution_target.mean().item(),
            'target_attribution_std': weighted_attribution_target.std().item(),
            'attribution_mse': loss_map.item()
        }
    }
    
    return total_loss, loss_ce, loss_kd, loss_map, loss_stats 