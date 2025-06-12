import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from utils import get_cifar10_dataloaders, train_epoch, test_epoch, save_checkpoint, get_lr_scheduler

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet110():
    # ResNet-110: 9n+2 layers, where n=18, so 18*6+2=110 layers
    # Each layer has 2*n blocks, so [18, 18, 18] blocks for each stage
    return ResNet(BasicBlock, [18, 18, 18])

def main():
    parser = argparse.ArgumentParser(description='Train ResNet-110 on CIFAR-10')
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
    model = ResNet110().to(device)
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
            save_checkpoint(model, optimizer, epoch, test_acc, 'resnet110_best', 'checkpoints')
        
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, test_acc, f'resnet110', 'checkpoints')
    
    print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 