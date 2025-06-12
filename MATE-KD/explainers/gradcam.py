import torch
import torch.nn.functional as F

class GradCAM:
    """Robust Grad-CAM implementation that avoids in-place operation issues."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.fmap = None
        self.grad = None

    def _save_fmap(self, module, input, output):
        # Store feature maps with explicit cloning to avoid view issues
        self.fmap = output.clone().detach()

    def _save_grad(self, module, grad_input, grad_output):
        # Store gradients with explicit cloning to avoid view issues
        if grad_output[0] is not None:
            self.grad = grad_output[0].clone().detach()

    def _normalize_cam(self, cam):
        """Normalize CAM to [0, 1] range with robust handling"""
        # Apply ReLU to keep only positive activations
        cam = F.relu(cam.clone())
        
        # Handle each sample in the batch
        batch_size = cam.shape[0]
        normalized_cams = []
        
        for i in range(batch_size):
            single_cam = cam[i:i+1].clone()
            
            # Flatten for normalization
            flat_cam = single_cam.view(-1)
            
            # Get min and max values
            cam_min = flat_cam.min()
            cam_max = flat_cam.max()
            
            # Avoid division by zero
            cam_range = cam_max - cam_min
            if cam_range < 1e-8:
                cam_range = torch.tensor(1.0, device=cam.device, dtype=cam.dtype)
            
            # Normalize to [0, 1]
            normalized_cam = (single_cam - cam_min) / cam_range
            normalized_cams.append(normalized_cam)
        
        return torch.cat(normalized_cams, dim=0)

    def __call__(self, x, class_idx):
        """Generate Grad-CAM heatmap with robust error handling."""
        # Reset and ensure clean state
        self.fmap = None
        self.grad = None
        
        # Store original training mode
        original_mode = self.model.training
        self.model.eval()
        
        # Register hooks temporarily
        forward_hook = self.target_layer.register_forward_hook(self._save_fmap)
        backward_hook = self.target_layer.register_full_backward_hook(self._save_grad)
        
        try:
            # Ensure clean gradients
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # Create input that requires gradients (avoid views)
            x_input = x.detach().clone().requires_grad_(True)
            
            # Forward pass with explicit gradient tracking
            logits = self.model(x_input)
            
            # Handle class index
            if isinstance(class_idx, torch.Tensor):
                if class_idx.dim() == 0:
                    class_idx = class_idx.unsqueeze(0)
                target_score = logits.gather(1, class_idx.view(-1, 1)).squeeze()
            else:
                target_score = logits[:, class_idx].squeeze()
            
            # Ensure target_score is a scalar for clean backward
            if target_score.dim() > 0:
                target_score = target_score.sum()
            
            # Backward pass
            target_score.backward(retain_graph=False)
            
            # Check if we captured the data
            if self.fmap is None or self.grad is None:
                print(f"Warning: Failed to capture data for target layer {type(self.target_layer).__name__}")
                return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], 
                                 device=x.device, dtype=x.dtype)
            
            # Compute weights (global average pooling)
            weights = torch.mean(self.grad, dim=(2, 3), keepdim=True)
            
            # Generate CAM
            cam = torch.sum(weights * self.fmap, dim=1, keepdim=True)
            
            # Normalize
            cam = self._normalize_cam(cam)
            
            return cam
            
        except Exception as e:
            print(f"Error in GradCAM computation: {e}")
            # Return zero heatmap as fallback
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], 
                             device=x.device, dtype=x.dtype)
            
        finally:
            # Always clean up
            forward_hook.remove()
            backward_hook.remove()
            self.model.train(original_mode)
            
            # Clear stored data
            self.fmap = None
            self.grad = None 