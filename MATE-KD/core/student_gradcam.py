"""
Efficient Student Grad-CAM using single autograd.grad call
Based on your corrected recipe - ~5ms per batch
"""

import torch
import torch.nn.functional as F
from typing import Tuple

class EfficientStudentGradCAM:
    """Efficient student Grad-CAM using single autograd.grad call"""
    
    def __init__(self, model, target_layer_name: str = 'layer3'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.feature_maps = None
        self.hook_registered = False
        
    def _register_hook(self):
        """Register forward hook to capture feature maps"""
        if self.hook_registered:
            return
            
        # Get target layer
        target_layer = self._get_target_layer()
        
        def hook_fn(module, input, output):
            self.feature_maps = output
        
        target_layer.register_forward_hook(hook_fn)
        self.hook_registered = True
    
    def _get_target_layer(self):
        """Get the target layer for hooking"""
        # For ResNet20: layer3[-1].conv2 (last conv in layer3)
        if hasattr(self.model, self.target_layer_name):
            layer_group = getattr(self.model, self.target_layer_name)
            if hasattr(layer_group, '__getitem__'):
                # It's a sequential/list - get last block
                last_block = layer_group[-1]
                if hasattr(last_block, 'conv2'):
                    return last_block.conv2
                else:
                    return last_block
            else:
                return layer_group
        else:
            raise ValueError(f"Target layer {self.target_layer_name} not found")
    
    def compute_batch_gradcam(self, images: torch.Tensor, 
                            use_predicted_class: bool = True) -> torch.Tensor:
        """
        Compute Grad-CAM for entire batch efficiently
        
        Args:
            images: Input batch (B, 3, 32, 32)
            use_predicted_class: Use predicted class (True) or ground truth
            
        Returns:
            torch.Tensor: Grad-CAM maps (B, 32, 32)
        """
        # Register hook if not done
        self._register_hook()
        
        # Reset feature maps
        self.feature_maps = None
        
        # Ensure input requires grad for Grad-CAM
        images.requires_grad_(True)
        
        # Forward pass - hook will capture feature maps
        logits = self.model(images)
        
        # Ensure feature maps require grad
        if self.feature_maps is not None:
            self.feature_maps.requires_grad_(True)
        
        if self.feature_maps is None:
            raise RuntimeError("Feature maps not captured - check hook registration")
        
        # Get target classes (predicted)
        pred_classes = logits.argmax(dim=1)  # (B,)
        
        # Create one-hot encoding for predicted classes
        batch_size = logits.shape[0]
        one_hot = torch.zeros_like(logits)
        one_hot[range(batch_size), pred_classes] = 1.0
        
        # Single autograd.grad call - THE KEY EFFICIENCY GAIN
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=self.feature_maps,
            grad_outputs=one_hot,
            retain_graph=True,
            create_graph=False
        )[0]  # (B, C, H, W)
        
        # Global Average Pooling of gradients
        weights = grads.mean(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination of feature maps
        grad_cam = (weights * self.feature_maps).sum(dim=1, keepdim=False)  # (B, H, W)
        
        # ReLU and normalize
        grad_cam = F.relu(grad_cam)
        
        # Upsample to 32x32 to match teacher cache resolution
        grad_cam = F.interpolate(grad_cam.unsqueeze(1), size=(32, 32), mode='bilinear', align_corners=False)
        grad_cam = grad_cam.squeeze(1)  # Remove channel dimension
        
        # L2 normalization per sample
        grad_cam_flat = grad_cam.view(batch_size, -1)
        grad_cam_norm = grad_cam_flat / (grad_cam_flat.norm(p=2, dim=1, keepdim=True) + 1e-6)
        grad_cam = grad_cam_norm.view(batch_size, 32, 32)
        
        return grad_cam
    
    def forward_with_gradcam(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combined forward pass and Grad-CAM computation
        
        Returns:
            Tuple of (logits, grad_cam_maps)
        """
        grad_cam = self.compute_batch_gradcam(images)
        
        # Logits were computed during Grad-CAM, but let's be explicit
        logits = self.model(images)
        
        return logits, grad_cam 