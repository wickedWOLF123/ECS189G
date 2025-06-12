#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from datetime import datetime
import yaml
from tqdm import tqdm
import pickle
import sys
import torch.nn.functional as F
from collections import defaultdict

# Import our models
sys.path.append('.')
from teachers.densenet import DenseNet121
from teachers.vgg import VGG19  
from teachers.resnet110 import ResNet110
from student.resnet import ResNet20
from core.knowledge_distillation import SimilarityWeightedKnowledgeDistillation
from explainers.gradcam import GradCAM

# AGGRESSIVE 92%+ CONFIGURATION
config = {
    'total_epochs': 200,  # Extended training
    'warmup_epochs': 20,  # Warmup period
    'initial_lr': 0.1,
    'min_lr': 1e-6,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'alpha': 0.4,  # Higher KD weight
    'beta': 0.0,   # No attribution loss (proven optimal)
    'initial_temperature': 5.0,  # Higher initial temperature
    'final_temperature': 1.5,    # Lower final temperature
    'batch_size': 128,
    'mixup_alpha': 0.2,  # Mixup augmentation
    'cutmix_alpha': 1.0, # CutMix augmentation
    'label_smoothing': 0.1,  # Label smoothing
}

# Advanced augmentation pipeline
def get_aggressive_transforms():
    """Advanced augmentation pipeline for 92%+ performance"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

# Mixup implementation
def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix implementation
def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# Progressive scheduling functions
def get_current_temperature(epoch, total_epochs, initial_temp, final_temp):
    """Progressive temperature scheduling"""
    if epoch < 50:  # Warm start with high temperature
        return initial_temp
    else:
        progress = (epoch - 50) / (total_epochs - 50)
        return initial_temp - progress * (initial_temp - final_temp)

def get_cosine_lr(epoch, total_epochs, initial_lr, min_lr, warmup_epochs):
    """Cosine annealing with warmup"""
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * epoch / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

# Enhanced knowledge distillation with adaptive weighting
class AggressiveKnowledgeDistillation(nn.Module):
    def __init__(self, alpha, temperature, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_output, teacher_outputs, true_labels, epoch=0):
        # Adaptive similarity weighting with temperature annealing
        similarities = []
        for teacher_output in teacher_outputs:
            sim = F.cosine_similarity(student_output, teacher_output, dim=1)
            similarities.append(sim)
        
        # Convert to weights with adaptive temperature
        adaptive_temp = max(0.05, 0.3 * (1 - epoch / 200))  # Decay over time
        weights = F.softmax(torch.stack(similarities) / adaptive_temp, dim=0)
        
        # Weighted ensemble
        weighted_teacher = sum(w.unsqueeze(1) * teacher_out 
                             for w, teacher_out in zip(weights, teacher_outputs))
        
        # KD loss with temperature
        kd_loss = self.kl_loss(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(weighted_teacher / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Classification loss
        ce_loss = self.ce_loss(student_output, true_labels)
        
        # Total loss
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return {
            'total_loss': total_loss,
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
            'teacher_weights': weights.mean(dim=1).cpu().tolist()
        }

def main():
    # Setup device and directories
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/aggressive_92plus_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory: {results_dir}")

    # Load teacher models
    print("Loading teacher models...")
    teachers = {}

    teacher_configs = {
        'densenet': (DenseNet121, 'teachers/densenet121_cifar10.pth'),
        'vgg19': (VGG19, 'teachers/vgg19_cifar10.pth'),
        'resnet': (ResNet110, 'teachers/resnet110_cifar10.pth')
    }

    teacher_models = []
    teacher_names = []

    for name, (model_class, path) in teacher_configs.items():
        model = model_class(num_classes=10).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        teachers[name] = model
        teacher_models.append(model)
        teacher_names.append(name)
        print(f"  Loaded {name}")

    # Create student model
    student = ResNet20(num_classes=10).to(device)
    print(f"Student model: {sum(p.numel() for p in student.parameters())/1e6:.2f}M parameters")

    # Setup data loaders with aggressive augmentation
    print("Setting up data loaders...")
    
    train_transform = get_aggressive_transforms()
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=4, pin_memory=True)

    # Setup optimizer (no scheduler, we'll manually update LR)
    optimizer = optim.SGD(student.parameters(), 
                         lr=config['initial_lr'], 
                         momentum=config['momentum'], 
                         weight_decay=config['weight_decay'])

    # Knowledge distillation with label smoothing
    kd_criterion = AggressiveKnowledgeDistillation(
        alpha=config['alpha'], 
        temperature=config['initial_temperature'],
        label_smoothing=config['label_smoothing']
    )

    # Training tracking
    best_accuracy = 0.0
    train_losses = []
    test_accuracies = []
    
    # Enhanced logging
    training_log = {
        'config': config,
        'epoch_logs': []
    }

    print(f"Starting aggressive training: {config['total_epochs']} epochs")
    print(f"Target: 92%+ accuracy")

    def train_epoch(epoch):
        student.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Update learning rate
        current_lr = get_cosine_lr(epoch, config['total_epochs'], 
                                  config['initial_lr'], config['min_lr'], 
                                  config['warmup_epochs'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Update temperature
        current_temp = get_current_temperature(epoch, config['total_epochs'], 
                                             config['initial_temperature'], 
                                             config['final_temperature'])
        kd_criterion.temperature = current_temp

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['total_epochs']}")
        
        kd_losses = []
        ce_losses = []
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Random augmentation strategy selection
            r = np.random.rand()
            if r < 0.3 and config['mixup_alpha'] > 0:
                # Mixup
                images, labels_a, labels_b, lam = mixup_data(images, labels, config['mixup_alpha'])
                use_mixup = True
            elif r < 0.6 and config['cutmix_alpha'] > 0:
                # CutMix  
                images, labels_a, labels_b, lam = cutmix_data(images, labels, config['cutmix_alpha'])
                use_mixup = True
            else:
                # Standard training
                use_mixup = False

            # Teacher forward passes
            teacher_outputs = []
            with torch.no_grad():
                for teacher in teacher_models:
                    teacher_out = teacher(images)
                    teacher_outputs.append(teacher_out)

            # Student forward pass
            student_output = student(images)

            # Compute loss
            if use_mixup:
                # For mixup/cutmix, we need to handle the mixed labels
                loss_dict_a = kd_criterion(student_output, teacher_outputs, labels_a, epoch)
                loss_dict_b = kd_criterion(student_output, teacher_outputs, labels_b, epoch)
                
                loss = lam * loss_dict_a['total_loss'] + (1 - lam) * loss_dict_b['total_loss']
                kd_loss = lam * loss_dict_a['kd_loss'] + (1 - lam) * loss_dict_b['kd_loss']
                ce_loss = lam * loss_dict_a['ce_loss'] + (1 - lam) * loss_dict_b['ce_loss']
            else:
                loss_dict = kd_criterion(student_output, teacher_outputs, labels, epoch)
                loss = loss_dict['total_loss']
                kd_loss = loss_dict['kd_loss']
                ce_loss = loss_dict['ce_loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            kd_losses.append(kd_loss)
            ce_losses.append(ce_loss)
            
            if not use_mixup:
                _, predicted = torch.max(student_output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Update progress bar
            if not use_mixup:
                accuracy = 100. * correct / total if total > 0 else 0
                pbar.set_postfix({
                    'Loss': f"{loss.item():.3f}",
                    'Acc': f"{accuracy:.2f}%",
                    'LR': f"{current_lr:.5f}",
                    'T': f"{current_temp:.2f}"
                })
            else:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.3f}",
                    'KD': f"{kd_loss:.3f}",
                    'CE': f"{ce_loss:.3f}",
                    'LR': f"{current_lr:.5f}",
                    'T': f"{current_temp:.2f}"
                })

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100. * correct / total if total > 0 else 0
        
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_accuracy': train_accuracy,
            'avg_kd_loss': np.mean(kd_losses),
            'avg_ce_loss': np.mean(ce_losses),
            'learning_rate': current_lr,
            'temperature': current_temp
        }
        
        return avg_loss, epoch_log

    def evaluate():
        student.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                
                test_loss += F.cross_entropy(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        return accuracy, avg_test_loss

    def save_checkpoint(epoch, accuracy):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'config': config,
            'training_log': training_log
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(results_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if accuracy > best_accuracy:
            torch.save(checkpoint, os.path.join(results_dir, f'best_model_{accuracy:.2f}pct.pth'))
            torch.save(student.state_dict(), os.path.join(results_dir, 'best_student_weights.pth'))

    # Main training loop
    print("Starting training...")
    
    for epoch in range(config['total_epochs']):
        # Train
        train_loss, epoch_log = train_epoch(epoch)
        
        # Evaluate
        test_accuracy, test_loss = evaluate()
        
        # Update best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"New best accuracy: {best_accuracy:.2f}%")
        
        # Update logs
        epoch_log.update({
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'best_accuracy': best_accuracy
        })
        
        training_log['epoch_logs'].append(epoch_log)
        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)
        
        # Save checkpoint
        save_checkpoint(epoch, test_accuracy)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{config['total_epochs']}: "
              f"Loss: {train_loss:.4f}, "
              f"Test Acc: {test_accuracy:.2f}%, "
              f"Best: {best_accuracy:.2f}%")
        
        # Save training log
        with open(os.path.join(results_dir, 'training_log.json'), 'w') as f:
            import json
            json.dump(training_log, f, indent=2)

    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%")
    
    # Final evaluation and analysis
    if best_accuracy >= 92.0:
        print("SUCCESS: Achieved 92%+ target!")
    else:
        gap = 92.0 - best_accuracy
        print(f"Target missed by {gap:.2f}%")

if __name__ == "__main__":
    main() 