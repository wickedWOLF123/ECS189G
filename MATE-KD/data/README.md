# CIFAR-10 Data Loading Utilities

This directory contains comprehensive data loading utilities for CIFAR-10 and CIFAR-10-C datasets.

## 📁 Files

- `cifar_loaders.py` - Main data loading utilities with proper normalization
- `README.md` - This file

## 🚀 Quick Start

### Basic CIFAR-10 Loading

```python
from data.cifar_loaders import get_cifar10_loaders, get_cifar10_test_loader

# Get both train and test loaders
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

# Get only test loader
test_loader = get_cifar10_test_loader(batch_size=128)
```

### Proper Normalization

The loaders use **correct CIFAR-10 normalization** that matches teacher training:

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
```

### Visualization

```python
from data.cifar_loaders import denormalize_cifar10

# Load data
test_loader = get_cifar10_test_loader()
images, labels = next(iter(test_loader))

# Denormalize for visualization
denorm_images = denormalize_cifar10(images)

# Now denorm_images are in [0, 1] range for matplotlib
import matplotlib.pyplot as plt
plt.imshow(denorm_images[0].permute(1, 2, 0))
```

## 📊 Available Functions

### Core Functions

- `get_cifar10_loaders()` - Get train and test loaders
- `get_cifar10_test_loader()` - Get test loader only
- `get_cifar10_transforms()` - Get transform pipelines
- `denormalize_cifar10()` - Denormalize tensors for visualization

### Legacy Compatibility

- `cifar_loader_correct()` - Backward compatible function name

## 🎯 Key Features

### ✅ Correct Normalization
- Uses proper CIFAR-10 statistics computed from training set
- Matches the normalization used in teacher model training
- Fixes the normalization mismatch that caused poor model performance

### ✅ Data Augmentation
- Training: RandomCrop + RandomHorizontalFlip + Normalization
- Testing: Only Normalization (no augmentation)

### ✅ Flexible Configuration
- Configurable batch size, num_workers, pin_memory
- Optional data augmentation control
- Automatic dataset downloading

## 🧪 Testing

Run the file directly to test functionality:

```bash
cd data
python cifar_loaders.py
```

Expected output:
```
🧪 Testing CIFAR-10 data loaders...
✅ CIFAR-10 batch shape: torch.Size([4, 3, 32, 32])
   Data range: [-2.429, 2.695]
   Denormalized range: [0.000, 1.000]
```

## 📈 Performance Impact

Using the correct normalization dramatically improves model accuracy:

| Model | Wrong Normalization | Correct Normalization | Improvement |
|-------|:------------------:|:--------------------:|:-----------:|
| DenseNet121 | 87.19% | **95.54%** | **+8.35%** |
| VGG19 | 72.15% | **91.14%** | **+18.99%** |
| ResNet110 | 84.05% | **93.85%** | **+9.80%** |

## 🔧 CIFAR-10-C Support

The file includes support for CIFAR-10-C (corrupted) dataset for robustness evaluation:

```python
# Download CIFAR-10-C dataset first
# wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
# tar -xf CIFAR-10-C.tar

from data.cifar_loaders import get_cifar10c_loader

# Load specific corruption
loader = get_cifar10c_loader('./CIFAR-10-C', 'gaussian_noise', severity=1)
```

### Available Corruptions
- brightness, contrast, defocus_blur, elastic_transform, fog
- frost, gaussian_blur, gaussian_noise, glass_blur, impulse_noise  
- jpeg_compression, motion_blur, pixelate, saturate, shot_noise
- snow, spatter, speckle_noise, zoom_blur

## 💡 Usage in Notebooks

Replace the old data loading:

```python
# OLD (wrong normalization)
from core.engine import cifar_loader
train_loader, test_loader = cifar_loader(batch_size=128)

# NEW (correct normalization)  
from data.cifar_loaders import get_cifar10_loaders
train_loader, test_loader = get_cifar10_loaders(batch_size=128)
```

## 🎓 Class Names

```python
from data.cifar_loaders import CIFAR10_CLASSES

print(CIFAR10_CLASSES)
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 
#  'dog', 'frog', 'horse', 'ship', 'truck']
``` 