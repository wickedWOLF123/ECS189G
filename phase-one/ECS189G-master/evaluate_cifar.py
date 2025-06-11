# evaluation.py
# Utilities to evaluate a CIFAR model on CIFAR-10, CIFAR-10-C, and Attribution Alignment

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

# --------------------------------------------------
# 1) CIFAR-10 Evaluation
# --------------------------------------------------
def evaluate_cifar10(model: nn.Module,
                    data_dir: str = '/home/adi000001kmr/ECS189G/data',
                     batch_size: int = 128,
                     num_workers: int = 2,
                     device: torch.device = None):
    """
    Evaluates the given model on CIFAR-10 test set and returns (loss, accuracy).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=True,
                               transform=test_transform)
    loader = DataLoader(testset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers)

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


# --------------------------------------------------
# 2) CIFAR-10-C Evaluation (subset & full)
# --------------------------------------------------
def load_cifar10c(data_dir: str = '/home/adi000001kmr/ECS189G/data/CIFAR-10-C'):
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
                             data_dir: str = '/home/adi000001kmr/ECS189G/data/CIFAR-10-C',
                             batch_size: int = 256,
                             num_corruptions: int = 5,
                             num_workers: int = 0,
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
                               data_dir: str = '/home/adi000001kmr/ECS189G/data/CIFAR-10-C',
                               batch_size: int = 256,
                               num_workers: int = 0,
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


# --------------------------------------------------
# 3) Attribution Alignment (Teacher vs. Student)
# --------------------------------------------------
def compute_alignment(model_s: nn.Module,
                      model_t: nn.Module,
                      explainer_name: str,
                      device: torch.device = None,
                      batch_size: int = 128,
                      num_workers: int = 2,
                      m_steps: int = 50):
    """
    Computes average MSE and alignment score (1 - MSE) between
    teacher and student attribution maps on the CIFAR-10 test set.

    - explainer_name: 'ig', 'cam', or 'campp'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_s.to(device).eval()
    model_t.to(device).eval()

    # 1) Prepare explainer objects / functions
    if explainer_name == 'ig':
        def get_maps(model, x, y):
            atts = integrated_gradients(
                model, x, target_label_idx=y, baseline=None, m_steps=m_steps
            )  # (B, C, H, W)
            maps = atts.abs().sum(dim=1)  # (B, H, W)
            return maps.to(device)
        get_teacher_maps = lambda x, y: get_maps(model_t, x, y)
        get_student_maps = lambda x, y: get_maps(model_s, x, y)
    else:
        # Use last conv layer for CAM methods
        target_layer_t = model_t.layer3[-1].conv2
        target_layer_s = model_s.layer3[-1].conv2
        if explainer_name == 'cam':
            expl_t = GradCAM(model_t, target_layer_t)
            expl_s = GradCAM(model_s, target_layer_s)
        else:  # 'campp'
            expl_t = GradCAMPlusPlus(model_t, target_layer_t)
            expl_s = GradCAMPlusPlus(model_s, target_layer_s)

        def get_teacher_maps(x, y):
            batch_maps = []
            for i in range(x.size(0)):
                xi = x[i:i+1]
                yi = int(y[i].item())
                cam = expl_t(xi, target_class=yi)  # numpy (h, w)
                batch_maps.append(torch.tensor(cam, device=device))
            return torch.stack(batch_maps)  # (B, h, w)

        def get_student_maps(x, y):
            batch_maps = []
            for i in range(x.size(0)):
                xi = x[i:i+1]
                yi = int(y[i].item())
                cam = expl_s(xi, target_class=yi)
                batch_maps.append(torch.tensor(cam, device=device))
            return torch.stack(batch_maps)  # (B, h, w)

    # 2) Iterate over CIFAR-10 test set
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    testset = datasets.CIFAR10(root='/home/adi000001kmr/ECS189G/data',
                               train=False,
                               download=True,
                               transform=test_transform)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc=f"Attr Alignment ({explainer_name.upper()})"):
            x, y = x.to(device), y.to(device)

            # Teacher & student attribution maps
            A_T = get_teacher_maps(x, y)  # (B, h_t, w_t)
            A_S = get_student_maps(x, y)  # (B, h_s, w_s)

            # Upsample A_T if its size differs
            if A_T.shape[1:] != A_S.shape[1:]:
                A_T = F.interpolate(
                    A_T.unsqueeze(1), size=A_S.shape[1:],
                    mode='bilinear', align_corners=False
                ).squeeze(1)

            # Normalize each map to [0,1] per sample
            B, H, W = A_T.shape
            A_T_flat = A_T.view(B, -1)
            A_S_flat = A_S.view(B, -1)
            min_T = A_T_flat.min(dim=1)[0].view(B,1)
            max_T = A_T_flat.max(dim=1)[0].view(B,1)
            min_S = A_S_flat.min(dim=1)[0].view(B,1)
            max_S = A_S_flat.max(dim=1)[0].view(B,1)

            A_T_norm = (A_T_flat - min_T) / (max_T - min_T + 1e-8)
            A_S_norm = (A_S_flat - min_S) / (max_S - min_S + 1e-8)

            # Per-sample MSE
            mse_per_sample = ((A_T_norm - A_S_norm) ** 2).mean(dim=1)  # (B,)
            total_mse += mse_per_sample.sum().item()
            total_samples += B

    avg_mse = total_mse / total_samples
    alignment_score = 1.0 - avg_mse  # larger is better
    return avg_mse, alignment_score


# --------------------------------------------------
# 4) Evaluate a list of student checkpoints
# --------------------------------------------------
def evaluate_students(student_ckpts: dict,
                      teacher_ckpt: str,
                      explainer_names: list,
                      data_dir: str = '/home/adi000001kmr/ECS189G/data',
                      device: torch.device = None):
    """
    Evaluates multiple student checkpoints (one per explainer) on:
      - CIFAR-10 clean accuracy
      - CIFAR-10-C average robust accuracy
      - Attribution alignment w.r.t. teacher
    student_ckpts: dict mapping explainer_name -> student_checkpoint_path
    explainer_names: list of explainer names (e.g., ['ig','cam','campp'])
    Returns a dict:
      { explainer: {'clean_acc': float,
                     'robust_acc': float,
                     'avg_mse': float,
                     'alignment': float } }
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load teacher once
    teacher = resnet56(num_classes=10).to(device)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher.eval()

    results = {}
    for name in explainer_names:
        # 1) Load student
        ckpt_path = student_ckpts[name]
        student = resnet20(num_classes=10).to(device)
        student.load_state_dict(torch.load(ckpt_path, map_location=device))
        student.eval()

        # 2) Clean accuracy
        _, clean_acc = evaluate_cifar10(student,
                                        data_dir=data_dir,
                                        batch_size=128,
                                        num_workers=2,
                                        device=device)

        # 3) Robust accuracy (average over all 15 corruptions)
        robust_acc = evaluate_cifar10c_full_avg(student,
                                                data_dir=os.path.join(data_dir, 'CIFAR-10-C'),
                                                batch_size=256,
                                                num_workers=2,
                                                device=device)

        # 4) Attribution alignment
        avg_mse, alignment = compute_alignment(student,
                                               teacher,
                                               explainer_name=name,
                                               device=device,
                                               batch_size=128,
                                               num_workers=2,
                                               m_steps=50)

        results[name] = {
            'clean_acc':      clean_acc,
            'robust_acc':     robust_acc,
            'avg_mse':        avg_mse,
            'alignment':      alignment
        }

    return results


# --------------------------------------------------
# If invoked as a script, demonstrate usage on three students
# --------------------------------------------------
if __name__ == '__main__':
    # Example usage (paths are placeholders; replace with your own):
    # python evaluation.py

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_ckpt = 'Checkpoints/resnet56_teacher_best.pth'

    # Map explainer names to student checkpoint paths:
    students = {
        'ig':   'Checkpoints/student_ig.pth',
        'cam':  'Checkpoints/student_cam.pth',
        'campp':'Checkpoints/student_campp.pth'
    }
    explainer_list = ['ig', 'cam', 'campp']

    metrics = evaluate_students(students,
                                teacher_ckpt,
                                explainer_list,
                                data_dir='/home/adi000001kmr/ECS189G/data',
                                device=device)

    # Print results
    for name, vals in metrics.items():
        print(f"=== {name.upper()} ===")
        print(f"Clean Acc       : {vals['clean_acc']:.2f}%")
        print(f"Robust Acc (C10-C Avg): {vals['robust_acc']:.2f}%")
        print(f"Avg Attribution MSE   : {vals['avg_mse']:.4f}")
        print(f"Alignment (1 - MSE)   : {vals['alignment']:.4f}")
        print()
