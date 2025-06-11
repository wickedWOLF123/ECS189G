# integrated_gradients.py
import torch

def integrated_gradients(
    model,
    input_tensor,
    target_label_idx=None,
    baseline=None,
    m_steps=50,
    cuda=True
):
    """
    Compute Integrated Gradients for a given model and input.
    Args:
        model: nn.Module (should be in eval mode)
        input_tensor: torch.Tensor, shape (1, C, H, W) or (B, C, H, W)
        target_label_idx: int or list of int (class index for which to explain)
        baseline: torch.Tensor, same shape as input_tensor, or None (defaults to all zeros)
        m_steps: int, number of interpolation steps
        cuda: bool, use CUDA
    Returns:
        integrated_gradients: torch.Tensor, same shape as input_tensor
    """
    model.eval()
    device = input_tensor.device
    if cuda:
        model = model.to(device)
    
    # Set baseline (reference point)
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    assert baseline.shape == input_tensor.shape

    # 1. Generate scaled inputs
    alphas = torch.linspace(0, 1, steps=m_steps+1).to(device)
    # shape: (m_steps+1, 1, 1, 1, 1)
    alphas = alphas.view(-1, *([1] * (input_tensor.dim())))

    # shape: (m_steps+1, B, C, H, W)
    input_expanded = baseline + alphas * (input_tensor - baseline)

    input_expanded = input_expanded.requires_grad_()

    # 2. Forward & collect gradients
    integrated_grads = torch.zeros_like(input_tensor).to(device)
    for i in range(m_steps):
        inp = baseline + alphas[i] * (input_tensor - baseline)
        inp = inp.clone().detach().requires_grad_(True).to(device)
        output = model(inp)
        # Handle batch or single input
        if target_label_idx is None:
            target_label = output.argmax(dim=1)
        else:
            target_label = target_label_idx
        # if multiple targets, loop
        if isinstance(target_label, list) or isinstance(target_label, torch.Tensor):
            grad_outputs = torch.zeros_like(output)
            for j, idx in enumerate(target_label):
                grad_outputs[j, idx] = 1.0
            loss = (output * grad_outputs).sum()
        else:
            loss = output[:, target_label].sum()
        model.zero_grad()
        loss.backward()
        integrated_grads += inp.grad.detach()

    # 3. Average and scale
    avg_grads = integrated_grads / m_steps
    integrated_grads = (input_tensor - baseline) * avg_grads

    return integrated_grads

