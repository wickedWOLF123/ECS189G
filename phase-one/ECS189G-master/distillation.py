import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast, GradScaler
from data_utils import get_cifar10_loaders
from model_utils import resnet56, resnet20
from explainers.integratedGradients import integrated_gradients
from explainers.gradcam import GradCAM
from explainers.gradcampp import GradCAMPlusPlus
from tqdm import tqdm
import time
import random


def get_attr_maps_ig(model, x, y, m_steps, device):
    """
    Compute batch attribution maps with Integrated Gradients.
    Returns tensor of shape (B, H, W).
    """
    try:
        atts = integrated_gradients(
            model, x, target_label_idx=y, baseline=None, m_steps=m_steps
        )  # shape: (B, C, H, W)
        # Sum absolute attributions over channels
        maps = atts.abs().sum(dim=1)  # (B, H, W)
        
        # Apply normalization to prevent extreme values
        B, H, W = maps.shape
        maps_flat = maps.view(B, -1)
        
        # Min-max normalization per sample
        min_vals = maps_flat.min(dim=1, keepdim=True)[0]
        max_vals = maps_flat.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        ranges = max_vals - min_vals
        ranges = torch.where(ranges < 1e-8, torch.ones_like(ranges), ranges)
        
        normalized = (maps_flat - min_vals) / ranges
        normalized_maps = normalized.view(B, H, W)
        
        return normalized_maps.to(device)
    except Exception as e:
        print(f"Warning: IG computation failed: {e}")
        # Return dummy maps as fallback
        return torch.ones(x.size(0), 32, 32, device=device) * 0.5


def get_attr_maps_cam_batch(explainer, x, y, device, max_samples=None):
    """
    Optimized batch attribution maps with CAM-based explainer.
    Processes samples more efficiently, optionally limiting number of samples.
    Returns tensor of shape (B, H, W).
    """
    # Limit number of samples for speed if specified
    if max_samples is not None and x.size(0) > max_samples:
        indices = torch.randperm(x.size(0))[:max_samples]
        x = x[indices]
        y = y[indices]
    
    # Process in smaller chunks to avoid memory issues
    chunk_size = min(8, x.size(0))  # Process 8 samples at a time
    maps = []
    
    for i in range(0, x.size(0), chunk_size):
        end_idx = min(i + chunk_size, x.size(0))
        x_chunk = x[i:end_idx]
        y_chunk = y[i:end_idx]
        
        chunk_maps = []
        for j in range(x_chunk.size(0)):
            xi = x_chunk[j:j+1]
            yi = int(y_chunk[j].item())
            cam = explainer(xi, target_class=yi)  # numpy (h, w)
            chunk_maps.append(torch.tensor(cam, device=device))
        maps.extend(chunk_maps)
    
    return torch.stack(maps)  # (B, h, w)


def should_compute_attribution(epoch, batch_idx, args):
    """
    Smart attribution frequency: more frequent early, less frequent later.
    Also includes random sampling to ensure all images get attribution guidance.
    """
    if args.attr_strategy == 'fixed':
        # Original fixed frequency
        return args.attr_freq == 0 or (batch_idx % args.attr_freq == 0)
    
    elif args.attr_strategy == 'progressive':
        # Progressive: frequent early, less frequent later
        if epoch <= args.epochs // 4:  # First 25% of training
            freq = max(1, args.attr_freq // 4)
        elif epoch <= args.epochs // 2:  # First 50% of training  
            freq = max(1, args.attr_freq // 2)
        else:  # Later training
            freq = args.attr_freq
        return freq == 0 or (batch_idx % freq == 0)
    
    elif args.attr_strategy == 'random':
        # Random sampling with probability
        prob = 1.0 / max(1, args.attr_freq) if args.attr_freq > 0 else 1.0
        return random.random() < prob
    
    else:  # 'hybrid' - combines progressive + random
        # Progressive frequency
        if epoch <= args.epochs // 4:
            base_freq = max(1, args.attr_freq // 4)
        elif epoch <= args.epochs // 2:
            base_freq = max(1, args.attr_freq // 2)  
        else:
            base_freq = args.attr_freq
            
        # Add some randomness to ensure coverage
        if base_freq == 0 or (batch_idx % base_freq == 0):
            return True
        else:
            # Small chance for random sampling
            return random.random() < 0.1 / max(1, base_freq)


def train(args):
    # Device
    device = args.device
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None

    # Data loaders with optimized settings
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        pin_memory=(device.type == 'cuda')
    )

    # Teacher
    teacher = resnet56(num_classes=10).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.eval()
    
    # Compile teacher for faster inference (PyTorch 2.0+)
    if hasattr(torch, 'compile') and args.compile_model:
        teacher = torch.compile(teacher)

    # Student
    student = resnet20(num_classes=10).to(device)
    student.train()
    
    # Compile student for faster training (PyTorch 2.0+)
    if hasattr(torch, 'compile') and args.compile_model:
        student = torch.compile(student)

    # Losses & optimizer
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

    # Set up explainer functions
    if args.explainer == 'ig':
        def explainer_T(x, y, max_samples=None):
            if max_samples is not None and x.size(0) > max_samples:
                indices = torch.randperm(x.size(0))[:max_samples]
                x, y = x[indices], y[indices]
            return get_attr_maps_ig(teacher, x, y, args.m_steps, device)
        def explainer_S(x, y, max_samples=None):
            if max_samples is not None and x.size(0) > max_samples:
                indices = torch.randperm(x.size(0))[:max_samples]
                x, y = x[indices], y[indices]
            return get_attr_maps_ig(student, x, y, args.m_steps, device)
    else:
        # CAM or CAM++: need target conv layer
        target_layer_t = teacher.layer3[-1].conv2
        target_layer_s = student.layer3[-1].conv2
        if args.explainer == 'cam':
            expl_t = GradCAM(teacher, target_layer_t)
            expl_s = GradCAM(student, target_layer_s)
        else:  # 'campp'
            expl_t = GradCAMPlusPlus(teacher, target_layer_t)
            expl_s = GradCAMPlusPlus(student, target_layer_s)
        def explainer_T(x, y, max_samples=None):
            return get_attr_maps_cam_batch(expl_t, x, y, device, max_samples)
        def explainer_S(x, y, max_samples=None):
            return get_attr_maps_cam_batch(expl_s, x, y, device, max_samples)

    # Hyperparams
    T = args.temperature
    alpha = args.alpha
    beta = 1.0 - alpha
    gamma = args.gamma_attr

    best_acc = 0.0
    
    # Performance tracking
    epoch_times = []
    attr_computations = 0
    total_batches = 0

    print(f"Attribution strategy: {args.attr_strategy}")
    if args.attr_batch_size > 0:
        print(f"Attribution batch size: {args.attr_batch_size}")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        student.train()
        total_loss = 0.0
        num_batches = 0
        epoch_attr_count = 0
        
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            total_batches += 1

            with autocast(enabled=(scaler is not None)):
                # Teacher forward (no grad)
                with torch.no_grad():
                    t_logits = teacher(x)

                # Student forward
                s_logits = student(x)

                # Loss components (always compute CE and KD)
                loss_ce = ce_loss(s_logits, y)
                loss_kd = (T * T) * kd_loss(
                    F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1)
                )

                # Smart attribution computation
                if should_compute_attribution(epoch, batch_idx, args):
                    # Attribution maps (expensive computation)
                    max_attr_samples = args.attr_batch_size if args.attr_batch_size > 0 else None
                    A_T = explainer_T(x, y, max_attr_samples)
                    A_S = explainer_S(x, y, max_attr_samples)
                    
                    # Resize CAM maps to match IG channel sum shape if needed
                    if A_T.shape[1:] != A_S.shape[1:]:
                        A_T = F.interpolate(
                            A_T.unsqueeze(1),
                            size=A_S.shape[1:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                    
                    loss_attr = attr_loss_fn(A_S, A_T)
                    
                    # Check for NaN in attribution loss
                    if torch.isnan(loss_attr) or torch.isinf(loss_attr):
                        print(f"Warning: NaN/Inf in attribution loss at epoch {epoch}, batch {batch_idx}")
                        loss_attr = torch.tensor(0.0, device=device)
                    
                    loss = alpha * loss_ce + beta * loss_kd + gamma * loss_attr
                    attr_computations += 1
                    epoch_attr_count += 1
                else:
                    # Skip attribution computation for speed
                    loss = alpha * loss_ce + beta * loss_kd

                # Check for NaN in total loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected in total loss at epoch {epoch}, batch {batch_idx}")
                    print(f"CE loss: {loss_ce.item()}, KD loss: {loss_kd.item()}")
                    # Skip this batch
                    continue

            # Backward pass with mixed precision
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                # Add gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Evaluation
        student.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast(enabled=(scaler is not None)):
                    preds = student(x).argmax(dim=1)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        acc = 100.0 * correct / total
        avg_loss = total_loss / num_batches
        attr_pct = 100.0 * epoch_attr_count / num_batches
        
        print(f"Epoch {epoch:03d} | Acc: {acc:.2f}% | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Attr: {attr_pct:.1f}%")
        
        # Print average epoch time every 10 epochs
        if epoch % 10 == 0:
            avg_time = sum(epoch_times[-10:]) / min(10, len(epoch_times))
            total_attr_pct = 100.0 * attr_computations / total_batches
            print(f"Avg time (last 10): {avg_time:.1f}s | Total attr coverage: {total_attr_pct:.1f}%")

        # Save best
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), args.output)

    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Attribution computed for {100.0 * attr_computations / total_batches:.1f}% of batches")
    if epoch_times:
        print(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation with Explainable Alignment")
    parser.add_argument('--explainer', choices=['ig','cam','campp'], required=True,
                        help="Which explainer to use for attribution alignment.")
    parser.add_argument('--teacher_ckpt', type=str, required=True,
                        help="Path to trained ResNet-56 teacher checkpoint.")
    parser.add_argument('--output', type=str, default='student_best.pth',
                        help="Where to save best student checkpoint.")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help="Root directory for CIFAR data.")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size (increased default for better GPU utilization)")
    parser.add_argument('--num_workers', type=int, default=8,
                        help="Number of data loading workers (increased default)")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30,40])
    parser.add_argument('--gamma_lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.7,
                        help="Weight for CE loss; (1 - alpha) for KD loss.")
    parser.add_argument('--temperature', type=float, default=4.0,
                        help="Softmax temperature for KD.")
    parser.add_argument('--gamma_attr', type=float, default=0.05,
                        help="Weight for attribution-alignment loss.")
    parser.add_argument('--m_steps', type=int, default=15,
                        help="Steps for Integrated Gradients (reduced for speed and stability)")
    parser.add_argument('--attr_freq', type=int, default=10,
                        help="Base frequency for attribution computation")
    parser.add_argument('--attr_strategy', choices=['fixed', 'progressive', 'random', 'hybrid'], 
                        default='hybrid',
                        help="Strategy for attribution computation frequency")
    parser.add_argument('--attr_batch_size', type=int, default=32,
                        help="Max samples for attribution computation (0=use full batch)")
    parser.add_argument('--mixed_precision', action='store_true',
                        help="Use mixed precision training for speed (requires CUDA)")
    parser.add_argument('--compile_model', action='store_true',
                        help="Use torch.compile for faster execution (PyTorch 2.0+)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(args)
