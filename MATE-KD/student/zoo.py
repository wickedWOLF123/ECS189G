import torch
import torch.nn as nn
import torchvision.models as tvm
from .resnet_20 import ResNet20

# ResNet components for custom ResNet110
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

def create_resnet110_cifar10():
    """Create ResNet-110 for CIFAR-10"""
    # ResNet-110: 6n+2 layers, where n=18, so 18*6+2=110 layers
    # Each stage has 18 blocks: [18, 18, 18]
    return ResNet(BasicBlock, [18, 18, 18])

def create_densenet121_cifar10():
    """Create DenseNet-121 adapted for CIFAR-10"""
    # Load DenseNet-121 without pre-trained weights
    model = tvm.densenet121(weights=None)
    
    # Modify the first convolution layer for CIFAR-10 (32x32 input)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the initial max pooling layer (not needed for small images)
    model.features.pool0 = nn.Identity()
    
    # Modify the classifier for CIFAR-10 (10 classes)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    
    return model

def create_vgg_cifar10(arch_name):
    """Create VGG model adapted for CIFAR-10"""
    if arch_name == 'vgg16':
        model = tvm.vgg16(weights=None)
    elif arch_name == 'vgg19':
        model = tvm.vgg19(weights=None)
    else:
        raise ValueError(f"Unknown VGG architecture: {arch_name}")
    
    # Modify the adaptive average pooling to work with CIFAR-10
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # Modify classifier for CIFAR-10 - matches training script
    model.classifier = nn.Sequential(
        nn.Linear(512 * 1 * 1, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 10),
    )
    
    return model

def load_teacher(path, arch):
    if arch == 'densenet':
        model = create_densenet121_cifar10()
    elif arch == 'vgg16':
        model = create_vgg_cifar10('vgg16')
    elif arch == 'vgg19':
        model = create_vgg_cifar10('vgg19')
    elif arch == 'resnet':
        model = create_resnet110_cifar10()  # Use custom ResNet110
    else:
        raise ValueError(f'Unknown teacher {arch}')
    
    chk = torch.load(path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(chk, dict) and 'model_state_dict' in chk:
        # Checkpoint contains model_state_dict
        model.load_state_dict(chk['model_state_dict'])
    elif isinstance(chk, dict):
        # Checkpoint is just the state_dict
        model.load_state_dict(chk)
    else:
        # Fallback for other formats
        model.load_state_dict(chk)
    
    model.eval()
    return model

def build_student(num_classes=10):
    return ResNet20(num_classes) 