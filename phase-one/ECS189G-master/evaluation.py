import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# For student/teacher loading and attribution alignment
from model_utils import resnet56, resnet20
from explainers.integratedGradients import integrated_gradients
from explainers.gradcam import GradCAM
from explainers.gradcampp import GradCAMPlusPlus

# CIFAR-10 normalization constants
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std  = (0.2470, 0.2435, 0.2616)

def evaluate_cifar10(model, data_dir='./data', batch_size=128, device=None):
    """
    Simple CIFAR-10 evaluation. Returns (loss, accuracy).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    model.to(device).eval()

    total_loss = 0.0
    correct = 0
    count = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='CIFAR-10 Eval'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            count += labels.size(0)

    return total_loss / count, 100.0 * correct / count

def compute_alignment(student_model, teacher_model, explainer_name, device=None, max_samples=1000):
    """
    Simple attribution alignment computation.
    explainer_name: 'ig', 'cam', or 'campp'
    max_samples: number of test samples to use (default 1000 for speed)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    student_model.to(device).eval()
    teacher_model.to(device).eval()

    # Setup test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # Prepare explainer functions
    if explainer_name == 'ig':
        def get_attribution(model, x, y):
            atts = integrated_gradients(model, x, target_label_idx=y, m_steps=15)
            return atts.abs().sum(dim=1)  # (B, H, W)
    else:
        # Setup CAM explainers
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
                    # Simple fallback
                    batch_maps.append(torch.ones((8, 8), device=device) * 0.5)
            return torch.stack(batch_maps)

    # Compute alignments
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
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
    Simple function to evaluate one student model.
    Returns dict with clean accuracy and alignment score.
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
    
    # Evaluate
    _, clean_acc = evaluate_cifar10(student, device=device)
    avg_mse, alignment = compute_alignment(student, teacher, explainer_name, device=device)
    
    return {
        'clean_accuracy': clean_acc,
        'alignment_mse': avg_mse,
        'alignment_score': alignment
    }

# 2) CIFAR-10-C Evaluation (subset & full)
# --------------------------------------------------
def load_cifar10c(data_dir: str = './data/CIFAR-10-C'):
    """
    Loads all CIFAR-10-C corruptions and returns:
      - sorted list of corruption names
      - labels array (shape (50000,))
    """
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    corruptions = [f.replace('.npy','')
                   for f in os.listdir(data_dir)
                   if f.endswith('.npy') and f != 'labels.npy']
    return sorted(corruptions), labels

def evaluate_cifar10c_subset(model: nn.Module,
                             data_dir: str = './data/CIFAR-10-C',
                             batch_size: int = 256,
                             num_corruptions: int = 5,
                             num_workers: int = 2,
                             device: torch.device = None):
    """
    Evaluates model on the first `num_corruptions` types of CIFAR-10-C perturbations.
    Returns a dict: { corruption_name: accuracy }.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # Gather all corruption names and labels
    corruptions, labels = load_cifar10c(data_dir)
    selected = corruptions[:num_corruptions]

    mean = torch.tensor(cifar10_mean).view(1,3,1,1).to(device)
    std  = torch.tensor(cifar10_std).view(1,3,1,1).to(device)

    results = {}
    for corruption in selected:
        imgs = np.load(os.path.join(data_dir, f"{corruption}.npy"))
        imgs = imgs.astype(np.float32) / 255.0          # shape: (50000,32,32,3)
        imgs_tensor = torch.from_numpy(imgs).permute(0,3,1,2).to(device)
        labels_tensor = torch.from_numpy(labels).to(device)

        dataset = TensorDataset(imgs_tensor, labels_tensor)
        loader  = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

        correct = 0
        count = 0
        with torch.no_grad():
            for x, y in loader:
                x = (x - mean) / std
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(y).sum().item()
                count += y.size(0)
        results[corruption] = 100.0 * correct / count

    return results

def evaluate_cifar10c_full_avg(model: nn.Module,
                               data_dir: str = './data/CIFAR-10-C',
                               batch_size: int = 256,
                               num_workers: int = 2,
                               device: torch.device = None):
    """
    Evaluates model on all 15 CIFAR-10-C corruptions and returns the average accuracy.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    corruptions, labels = load_cifar10c(data_dir)
    mean = torch.tensor(cifar10_mean).view(1,3,1,1).to(device)
    std  = torch.tensor(cifar10_std).view(1,3,1,1).to(device)

    total_acc = 0.0
    for corruption in corruptions:
        imgs = np.load(os.path.join(data_dir, f"{corruption}.npy"))
        imgs = imgs.astype(np.float32) / 255.0
        imgs_tensor = torch.from_numpy(imgs).permute(0,3,1,2).to(device)
        labels_tensor = torch.from_numpy(labels).to(device)

        dataset = TensorDataset(imgs_tensor, labels_tensor)
        loader  = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

        correct = 0
        count = 0
        with torch.no_grad():
            for x, y in loader:
                x = (x - mean) / std
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(y).sum().item()
                count += y.size(0)
        total_acc += 100.0 * correct / count

    avg_acc = total_acc / len(corruptions)
    return avg_acc

def evaluate_model_comprehensive(model_path, model_type='resnet56', model_name='Model'):
    """
    Comprehensive evaluation on CIFAR-10 clean and CIFAR-10-C.
    
    Args:
        model_path: Path to model checkpoint
        model_type: 'resnet56' or 'resnet20'
        model_name: Name for printing results
    
    Returns:
        dict: Complete evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if model_type == 'resnet56':
        model = resnet56(num_classes=10)
    elif model_type == 'resnet20':
        model = resnet20(num_classes=10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name} ({model_type.upper()})")
    print(f"{'='*60}")
    
    # CIFAR-10 Clean
    print("\n🔹 CIFAR-10 Clean Test Set:")
    clean_loss, clean_acc = evaluate_cifar10(model, device=device)
    print(f"   Loss: {clean_loss:.4f} | Accuracy: {clean_acc:.2f}%")
    
    # CIFAR-10-C
    print("\n🔹 CIFAR-10-C Corruptions:")
    c10c_results = evaluate_cifar10c_subset(model, device=device, num_corruptions=15)
    
    print("\n   Individual Corruption Results:")
    for corruption, accuracy in c10c_results.items():
        print(f"     {corruption:20s}: {accuracy:.2f}%")
    
    # Calculate average
    c10c_accuracies = list(c10c_results.values())
    avg_c10c_acc = sum(c10c_accuracies) / len(c10c_accuracies)
    
    print(f"\n   📊 Average CIFAR-10-C Accuracy: {avg_c10c_acc:.2f}%")
    print(f"   📈 Clean vs Corrupted Gap: {clean_acc - avg_c10c_acc:.2f}%")
    
    return {
        'model_name': model_name,
        'model_type': model_type,
        'clean_accuracy': clean_acc,
        'clean_loss': clean_loss,
        'cifar10c_results': c10c_results,
        'cifar10c_average': avg_c10c_acc,
        'robustness_gap': clean_acc - avg_c10c_acc
    }

# Simple usage example
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example paths - update these to match your actual files
    teacher_path = 'Checkpoints/resnet56_teacher_best.pth'
    
    explainers = ['ig', 'cam', 'campp']
    results = {}
    
    for explainer in explainers:
        student_path = f'Checkpoints/student_resnet20_{explainer}_epoch_50.pth'
        results[explainer] = evaluate_student(student_path, teacher_path, explainer, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for explainer, metrics in results.items():
        print(f"\n{explainer.upper()}:")
        print(f"  Clean Accuracy: {metrics['clean_accuracy']:.2f}%")
        print(f"  Alignment Score: {metrics['alignment_score']:.4f}")
        print(f"  Attribution MSE: {metrics['alignment_mse']:.4f}")
