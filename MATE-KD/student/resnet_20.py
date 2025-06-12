import torch.nn as nn
from torchvision.models.resnet import BasicBlock

# ---------- shared helper ---------- #
def _make_layer(inplanes, planes, blocks, stride):
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes))
    layers = [BasicBlock(inplanes, planes, stride, downsample)]
    inplanes = planes
    layers += [BasicBlock(inplanes, planes) for _ in range(blocks - 1)]
    return nn.Sequential(*layers), inplanes

# ---------- ResNet-10 (depth 10 = 2,2,2 blocks) ---------- #
class ResNet10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1, self.relu = nn.BatchNorm2d(64), nn.ReLU(inplace=True)

        self.layer1, self.inplanes = _make_layer(self.inplanes, 64 , 2, 1)
        self.layer2, self.inplanes = _make_layer(self.inplanes, 128, 2, 2)
        self.layer3, self.inplanes = _make_layer(self.inplanes, 256, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.fc(self.avgpool(x).flatten(1))

# ---------- ResNet-20 (depth 20 = 3,3,3 blocks) ---------- #
class ResNet20(nn.Module):
    """
    Classic CIFAR-10 ResNet-20 (6n+2 with n=3).
    Stages: 3×{conv3-64}, 3×{conv3-128}, 3×{conv3-256}
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1, self.relu = nn.BatchNorm2d(16), nn.ReLU(inplace=True)

        # 3 blocks per stage
        self.layer1, self.inplanes = _make_layer(self.inplanes, 16 , 3, 1)
        self.layer2, self.inplanes = _make_layer(self.inplanes, 32 , 3, 2)
        self.layer3, self.inplanes = _make_layer(self.inplanes, 64 , 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.fc(self.avgpool(x).flatten(1))
