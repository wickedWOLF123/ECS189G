# Teacher Models for CIFAR-10

This directory contains training scripts for four teacher models on the CIFAR-10 dataset:

1. **ResNet-56** - Deep residual network with 56 layers
2. **ResNet-110** - Deep residual network with 110 layers  
3. **DenseNet-121** - Densely connected network with 121 layers
4. **VGG-16** - Visual Geometry Group network with 16 layers

## Directory Structure

```
teachers/
├── utils.py                 # Common utilities for training
├── train_resnet56.py       # ResNet-56 training script
├── train_resnet110.py      # ResNet-110 training script
├── train_densenet121.py    # DenseNet-121 training script
├── train_vgg16.py          # VGG-16 training script
├── train_all_teachers.py   # Master script to train all models
├── checkpoints/            # Directory for saved model checkpoints
└── README.md              # This file
```

## Model Architectures

### ResNet-56/110
- Custom implementation optimized for CIFAR-10
- Uses BasicBlock with 3x3 convolutions
- Three stages with [9,9,9] blocks (ResNet-56) or [18,18,18] blocks (ResNet-110)
- Initial 16 channels, doubling at each stage
- Global average pooling before classification

### DenseNet-121
- Based on torchvision's DenseNet-121
- Modified for CIFAR-10: 3x3 initial conv, no max pooling
- Dense connections within blocks
- Growth rate of 32, compression factor of 0.5

### VGG-16
- Based on torchvision's VGG-16
- Modified classifier for CIFAR-10 (10 classes)
- Adaptive average pooling for 32x32 input size
- Dropout regularization in classifier

## Training Configuration

### Default Hyperparameters

| Model | Learning Rate | Weight Decay | Scheduler | Epochs |
|-------|---------------|--------------|-----------|---------|
| ResNet-56 | 0.1 | 1e-4 | Cosine | 200 |
| ResNet-110 | 0.1 | 1e-4 | Cosine | 200 |
| DenseNet-121 | 0.1 | 1e-4 | Cosine | 200 |
| VGG-16 | 0.01 | 5e-4 | Step | 200 |

### Data Augmentation
- Random crop (32x32 with padding=4)
- Random horizontal flip
- Normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)

## Usage

### Train Individual Models

```bash
# Train ResNet-56
python train_resnet56.py --epochs 100 --batch_size 128

# Train ResNet-110
python train_resnet110.py --epochs 100 --batch_size 128

# Train DenseNet-121
python train_densenet121.py --epochs 100 --batch_size 128

# Train VGG-16
python train_vgg16.py --epochs 100 --batch_size 128
```

### Train All Models

```bash
# Train all models with default settings
python train_all_teachers.py

# Train specific models
python train_all_teachers.py --models resnet56 densenet121

# Custom configuration
python train_all_teachers.py --epochs 50 --batch_size 64 --device cuda
```

### Command Line Arguments

All training scripts support the following arguments:

- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (model-specific defaults)
- `--weight_decay`: Weight decay (model-specific defaults)
- `--save_freq`: Checkpoint save frequency (default: 50)
- `--device`: Device to use ('cuda' or 'cpu', default: 'cuda')

## Expected Performance

With proper training, these models should achieve the following approximate accuracies on CIFAR-10:

- **ResNet-56**: ~93-94%
- **ResNet-110**: ~94-95%
- **DenseNet-121**: ~95-96%
- **VGG-16**: ~92-93%

## Checkpoints

Model checkpoints are automatically saved in the `checkpoints/` directory:

- Best model: `{model_name}_best_epoch_{epoch}_acc_{accuracy}.pth`
- Regular checkpoints: `{model_name}_epoch_{epoch}_acc_{accuracy}.pth`

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Epoch number
- Best accuracy achieved

## Requirements

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.21.0
- tqdm >= 4.64.0
- matplotlib >= 3.5.0

## Notes

- Training times vary significantly between models (VGG-16 < ResNet-56 < DenseNet-121 < ResNet-110)
- GPU memory usage: VGG-16 (~2GB) < ResNet-56 (~3GB) < ResNet-110 (~4GB) < DenseNet-121 (~5GB)
- All models use SGD optimizer with momentum=0.9
- Learning rate scheduling helps achieve better convergence 