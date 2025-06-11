import torch
import torch.nn.functional as F

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        """
        model: PyTorch model in eval mode
        target_layer: nn.Module, last conv layer (e.g., model.layer3[-1].conv2)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, target_class=None):
        """
        input_tensor: (1, 3, 32, 32)
        target_class: int, class index to explain. If None, uses predicted class.
        Returns: heatmap (numpy array, shape [H,W])
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        loss = output[0, target_class]
        loss.backward(retain_graph=True)

        # Gradients & activations: (B,C,H,W)
        grads = self.gradients      # dScore/dActivation
        activations = self.activations

        # Grad-CAM++ weights
        grads2 = grads ** 2
        grads3 = grads2 * grads

        # Sum across spatial dims
        sum_activations = torch.sum(activations, dim=(2,3), keepdim=True)

        # Epsilon to avoid division by zero
        eps = 1e-8

        # alpha_ij
        alpha_numer = grads2
        alpha_denom = 2 * grads2 + sum_activations * grads3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.tensor(eps, device=alpha_denom.device))
        alpha = alpha_numer / alpha_denom

        # Positive gradients only
        relu_grad = F.relu(loss.exp() * grads)
        weights = (alpha * relu_grad).sum(dim=(2,3), keepdim=True)  # (B,C,1,1)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Remove negative values
        cam = cam[0,0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

