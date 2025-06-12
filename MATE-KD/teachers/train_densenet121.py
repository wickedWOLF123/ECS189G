import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse
import os
from utils import get_cifar10_dataloaders, train_epoch, test_epoch, save_checkpoint, get_lr_scheduler

def create_densenet121_cifar10():
    """
    Create DenseNet-121 adapted for CIFAR-10
    """
    # Load DenseNet-121 without pre-trained weights
    model = models.densenet121(weights=None)
    
    # Modify the first convolution layer for CIFAR-10 (32x32 input)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the initial max pooling layer (not needed for small images)
    model.features.pool0 = nn.Identity()
    
    # Modify the classifier for CIFAR-10 (10 classes)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train DenseNet-121 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
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
    model = create_densenet121_cifar10().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, 'cosine', args.epochs)

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
            save_checkpoint(model, optimizer, epoch, test_acc, 'densenet121_best', 'checkpoints')
        
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, test_acc, f'densenet121', 'checkpoints')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 