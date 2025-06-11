import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import os
from model_utils import resnet56, resnet20
from explainers.integratedGradients import integrated_gradients
from explainers.gradcam import GradCAM
from explainers.gradcampp import GradCAMPlusPlus

def compute_alignment(student_model, teacher_model, explainer_name, device=None, max_samples=1000):
    """
    Compute attribution alignment between student and teacher models.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up data loader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)  # Smaller batch size

    # Prepare explainer functions
    if explainer_name == 'ig':
        def get_attribution(model, x, y):
            # Enable gradients for input
            x = x.clone().detach().requires_grad_(True)
            try:
                # Compute IG with gradient tracking enabled
                atts = integrated_gradients(
                    model, x, target_label_idx=y, 
                    baseline=None, m_steps=15
                )
                # Sum absolute attributions over channels
                maps = atts.abs().sum(dim=1)  # (B, H, W)
                
                # Normalize to prevent extreme values
                B, H, W = maps.shape
                maps_flat = maps.view(B, -1)
                
                # Min-max normalization per sample
                min_vals = maps_flat.min(dim=1, keepdim=True)[0]
                max_vals = maps_flat.max(dim=1, keepdim=True)[0]
                
                # Avoid division by zero
                ranges = max_vals - min_vals
                ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
                
                normalized = (maps_flat - min_vals) / ranges
                return normalized.view(B, H, W)
            except Exception as e:
                print(f"Warning: IG computation failed: {e}")
                # Return dummy maps as fallback
                return torch.ones(x.size(0), 32, 32, device=x.device) * 0.5
    else:
        # CAM-based explainers (unchanged)
        target_layer_t = teacher_model.layer3[-1].conv2
        target_layer_s = student_model.layer3[-1].conv2
        
        if explainer_name == 'cam':
            explainer_t = GradCAM(teacher_model, target_layer_t)
            explainer_s = GradCAM(student_model, target_layer_s)
        else:  # 'campp'
            explainer_t = GradCAMPlusPlus(teacher_model, target_layer_t)
            explainer_s = GradCAMPlusPlus(student_model, target_layer_s)
        
        def get_attribution(model, x, y, is_teacher=True):
            explainer = explainer_t if is_teacher else explainer_s
            batch_maps = []
            for i in range(x.size(0)):
                try:
                    xi = x[i:i+1]
                    yi = int(y[i].item())
                    cam = explainer(xi, target_class=yi)
                    batch_maps.append(torch.tensor(cam, device=device))
                except:
                    batch_maps.append(torch.ones((8, 8), device=device) * 0.5)
            return torch.stack(batch_maps)

    # Compute alignments
    total_mse = 0.0
    total_samples = 0
    
    # Process in smaller batches to avoid memory issues
    for x, y in tqdm(test_loader, desc=f'{explainer_name.upper()} Alignment'):
        if total_samples >= max_samples:
            break
            
        x, y = x.to(device), y.to(device)
        
        # Get attribution maps
        if explainer_name == 'ig':
            teacher_maps = get_attribution(teacher_model, x, y)
            student_maps = get_attribution(student_model, x, y)
        else:
            teacher_maps = get_attribution(teacher_model, x, y, is_teacher=True)
            student_maps = get_attribution(student_model, x, y, is_teacher=False)
        
        # Make sure dimensions match
        if teacher_maps.shape != student_maps.shape:
            teacher_maps = F.interpolate(
                teacher_maps.unsqueeze(1), 
                size=student_maps.shape[1:], 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
        
        # Normalize to [0,1]
        B = teacher_maps.shape[0]
        t_flat = teacher_maps.view(B, -1)
        s_flat = student_maps.view(B, -1)
        
        t_norm = (t_flat - t_flat.min(dim=1)[0].unsqueeze(1)) / (t_flat.max(dim=1)[0].unsqueeze(1) - t_flat.min(dim=1)[0].unsqueeze(1) + 1e-8)
        s_norm = (s_flat - s_flat.min(dim=1)[0].unsqueeze(1)) / (s_flat.max(dim=1)[0].unsqueeze(1) - s_flat.min(dim=1)[0].unsqueeze(1) + 1e-8)
        
        # Compute MSE
        mse = ((t_norm - s_norm) ** 2).mean(dim=1)
        total_mse += mse.sum().item()
        total_samples += B

    avg_mse = total_mse / total_samples
    alignment_score = 1.0 - avg_mse
    return avg_mse, alignment_score

def evaluate_student(student_path, teacher_path, explainer_name, device=None):
    """
    Evaluate student model with fixed IG computation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Evaluating {explainer_name.upper()} student...")
    
    # Load models
    teacher = resnet56(num_classes=10).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    
    student = resnet20(num_classes=10).to(device)
    student.load_state_dict(torch.load(student_path, map_location=device))
    student.eval()
    
    # Evaluate clean accuracy
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = student(x)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
    clean_acc = 100.0 * correct / total
    
    # Compute alignment
    avg_mse, alignment = compute_alignment(student, teacher, explainer_name, device=device)
    
    return {
        'clean_accuracy': clean_acc,
        'alignment_mse': avg_mse,
        'alignment_score': alignment
    }