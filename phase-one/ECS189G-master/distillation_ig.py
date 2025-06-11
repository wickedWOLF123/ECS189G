#!/usr/bin/env python3
"""
distillation_ig.py

Standalone script for explainable knowledge distillation using Integrated Gradients (IG).
Teacher: ResNet-56
Student: ResNet-20

Loss = α·CE(student_logits, labels)
     + (1−α)·T²·KL(softmax(student_logits/T) ∥ softmax(teacher_logits/T))
     + γ·MSE(IG_student, IG_teacher)

Usage example:
    python distillation_ig.py \
      --teacher_ckpt /path/to/resnet56_teacher.pth \
      --output /path/to/student_ig.pth \
      --data_dir ./data \
      --epochs 30 \
      --lr 0.1 \
      --batch_size 1 \
      --m_steps 20 \
      --alpha 0.5 \
      --temperature 4.0 \
      --gamma_attr 1.0 \
      --num_workers 2 \
      --mixed_precision \
      --compile_model
"""

import argparse
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from data_utils import get_cifar10_loaders
from model_utils import resnet56, resnet20
from explainers.integratedGradients import integrated_gradients

# CIFAR-10 normalization
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)


def get_attr_maps_ig(model, x, y, m_steps, device):
    """
    Compute Integrated Gradients attribution map for input x (batch size = 1).
    x: Tensor of shape [1, 3, 32, 32], normalized
    y: integer label (0..9)
    Returns: Tensor of shape [1, H, W] (H=W=32)
    """
    atts = integrated_gradients(
        model, x, target_label_idx=y, baseline=None, m_steps=m_steps
    )  # shape: (1, C, H, W)
    # Sum absolute attributions across channels
    maps = atts.abs().sum(dim=1)  # (1, H, W)
    return maps.to(device)


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Mixed precision scaler (if enabled)
    scaler = GradScaler() if (args.mixed_precision and device.type == 'cuda') else None

    # Data loaders
    # For IG we recommend batch_size=1 to simplify attribution calls
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        pin_memory=(device.type == 'cuda')
    )

    # Teacher model
    teacher = resnet56(num_classes=10).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    if args.compile_model and hasattr(torch, 'compile'):
        teacher = torch.compile(teacher)

    # Student model
    student = resnet20(num_classes=10).to(device)
    student.train()
    if args.compile_model and hasattr(torch, 'compile'):
        student = torch.compile(student)

    # Loss functions and optimizer
    ce_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss(reduction='batchmean')
    attr_loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma_lr
    )

    # Hyperparameters
    T = args.temperature
    alpha = args.alpha
    beta = 1.0 - alpha
    gamma = args.gamma_attr
    m_steps = args.m_steps

    best_acc = 0.0
    epoch_times = []

    print("Starting IG-based distillation")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  LR              : {args.lr}")
    print(f"  m_steps (IG)    : {m_steps}")
    print(f"  α (CE weight)   : {alpha}")
    print(f"  T (temperature) : {T}")
    print(f"  γ (attr weight) : {gamma}")
    print(f"  Mixed Precision : {args.mixed_precision}")
    print(f"  Compile Model   : {args.compile_model}")
    print("============================================")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        student.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # y is a tensor of shape [1]; convert to int
            y_int = int(y.item())

            with autocast(enabled=(scaler is not None)):
                # Teacher forward (no grad)
                with torch.no_grad():
                    t_logits = teacher(x)

                # Student forward
                s_logits = student(x)

                # Compute CE and KD losses
                loss_ce = ce_loss(s_logits, y)
                loss_kd = (T * T) * kd_loss(
                    F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1)
                )

                # Compute IG attributions for teacher and student
                # (batch_size=1 assumed)
                A_T = get_attr_maps_ig(teacher, x, y_int, m_steps, device)  # (1, H, W)
                A_S = get_attr_maps_ig(student, x, y_int, m_steps, device)  # (1, H, W)

                # Compute attribution alignment loss
                loss_attr = attr_loss_fn(A_S, A_T)

                # Total loss
                loss = alpha * loss_ce + beta * loss_kd + gamma * loss_attr

            # Backpropagation
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        # Step LR scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Evaluate on CIFAR-10 test set
        student.eval()
        correct = total = 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                with autocast(enabled=(scaler is not None)):
                    preds = student(x_val).argmax(dim=1)
                correct += preds.eq(y_val).sum().item()
                total += y_val.size(0)
        acc = 100.0 * correct / total
        avg_loss = running_loss / num_batches

        print(f"Epoch {epoch:02d} | Acc: {acc:.2f}% | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

        # Save best student
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), args.output)

    print("============================================")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="IG Distillation: ResNet-56 → ResNet-20")
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                        help="Path to ResNet-56 teacher checkpoint")
    parser.add_argument('--output', type=str, default='student_ig.pth',
                        help="Where to save best ResNet-20 student")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help="Root directory for CIFAR data")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size (IG requires batch=1 for correct attribution)")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="DataLoader workers for CIFAR-10")
    parser.add_argument('--epochs', type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="Initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="Weight decay (L2)")
    parser.add_argument('--milestones', nargs='+', type=int, default=[15, 25],
                        help="LR decay milestones (epochs)")
    parser.add_argument('--gamma_lr', type=float, default=0.1,
                        help="LR decay factor at milestones")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Weight for CE loss (1-alpha for KD loss)")
    parser.add_argument('--temperature', type=float, default=4.0,
                        help="Temperature for KD")
    parser.add_argument('--gamma_attr', type=float, default=1.0,
                        help="Weight for attribution alignment loss")
    parser.add_argument('--m_steps', type=int, default=20,
                        help="Number of IG steps (reduced for speed)")
    parser.add_argument('--mixed_precision', action='store_true',
                        help="Enable mixed precision (FP16) training")
    parser.add_argument('--compile_model', action='store_true',
                        help="Enable torch.compile optimization (PyTorch 2.0+)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args)
