import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse
import os
from utils import get_cifar10_dataloaders, train_epoch, test_epoch, save_checkpoint, get_lr_scheduler

def create_vgg19_cifar10():
    """
    Create VGG-19 adapted for CIFAR-10
    """
    # Load VGG-19 without pre-trained weights
    model = models.vgg19(weights=None)
    
    # Modify the classifier for CIFAR-10
    # VGG's classifier expects 7*7*512 = 25088 features from 224x224 input
    # For CIFAR-10 (32x32), we need to calculate the correct input size
    # After all conv layers and pooling, 32x32 becomes 1x1x512
    model.classifier = nn.Sequential(
        nn.Linear(512 * 1 * 1, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 10),
    )
    
    # Modify the adaptive average pooling to work with CIFAR-10
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train VGG-19 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=args.batch_size)
    print(f'Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}')

    # Create model
    model = create_vgg19_cifar10().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, 'step', args.epochs)

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        print(f'Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, test_acc, 'vgg19_best', 'checkpoints')
        
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, test_acc, f'vgg19', 'checkpoints')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 