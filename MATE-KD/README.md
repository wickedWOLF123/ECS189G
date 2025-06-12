# MATE-KD: Multi-teacher Attribution-weighted Knowledge Distillation


## Overview

MATE-KD (Multi-teacher Attribution-weighted Knowledge Distillation) is a novel approach to knowledge distillation that leverages Grad-CAM attribution maps to dynamically weight multiple teacher models during student training. Unlike traditional knowledge distillation that relies on fixed teacher weights or simple averaging, MATE-KD adaptively determines teacher contributions based on attribution similarity between student and teacher models.

## Key Features

- **Dynamic Teacher Weighting**: Uses Grad-CAM attribution similarity to weight teacher contributions
- **Multi-teacher Architecture**: Combines knowledge from ResNet110, DenseNet121, and VGG19 teachers
- **Efficient Student Model**: ResNet20 student achieving 90.22% accuracy on CIFAR-10
- **Robust Performance**: Comprehensive evaluation on CIFAR-10-C corruptions
- **Attribution Analysis**: Detailed Grad-CAM visualization and similarity metrics

## Architecture

### Teacher Models
- **ResNet110**: Deep residual network with 110 layers
- **DenseNet121**: Dense connectivity with 121 layers  
- **VGG19**: Classic convolutional architecture with 19 layers

### Student Model
- **ResNet20**: Lightweight 20-layer residual network
- **Parameters**: ~270K (vs ~11M average for teachers)
- **Performance**: 90.22% accuracy on CIFAR-10

### Key Components
- **Attribution-weighted Loss**: Combines classification, knowledge distillation, and attribution matching
- **Dynamic Teacher Weighting**: Real-time adaptation based on Grad-CAM similarity
- **Efficient Caching**: Pre-computed teacher attribution maps for fast training

## Results

| Model | CIFAR-10 Accuracy | Parameters | Model Size |
|-------|------------------|------------|------------|
| ResNet110 | 93.5% | 1.7M | 6.8MB |
| DenseNet121 | 94.2% | 7.0M | 28MB |
| VGG19 | 93.8% | 20M | 80MB |
| **Student (MATE-KD)** | **90.22%** | **0.27M** | **1.1MB** |

### Robustness (CIFAR-10-C Average)
- **DenseNet121**: 75.89%
- **ResNet110**: 74.05% 
- **VGG19**: 73.47%
- **Student**: 68.51%

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/MATE-KD.git
cd MATE-KD

# Create environment
conda env create -f environment.yml
conda activate mate-kd

# Or use pip
pip install -r requirements.txt
```

### Download Data

```bash
# Download CIFAR-10 and CIFAR-10-C
python data/download_data.py --dataset cifar10
python data/download_data.py --dataset cifar10c
```

### Training

```bash
# Train student model with MATE-KD
python train_run_b_extended.py --config cfg.yaml
```

### Evaluation

```bash
# Evaluate trained model
python scripts/reproduce_results.py --model archive/results/best_student_90.22pct/best_model_90.22pct.pth
```

## Notebook Demo

See `MATE-KD.ipynb` for comprehensive analysis including:
- Model accuracy comparison
- Confusion matrix analysis
- CIFAR-10-C robustness evaluation
- Grad-CAM visualization and similarity analysis

## Project Structure

```
MATE-KD/
├── README.md                          # This file
├── SETUP.md                           # Detailed setup instructions
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── train_run_b_extended.py           # Main training script
├── cfg.yaml                          # Configuration file
├── MATE-KD.ipynb                     # Analysis notebook
│
├── core/                             # Core algorithms
│   ├── similarity.py                 # Attribution similarity metrics
│   ├── losses.py                     # Loss functions
│   ├── engine.py                     # Training engine
│   └── weighting.py                  # Teacher weighting
│
├── student/                          # Student model
│   ├── zoo.py                        # Model architectures
│   └── resnet_20.py                  # ResNet20 implementation
│
├── explainers/                       # Attribution methods
│   ├── gradcam.py                    # Grad-CAM implementation
│   └── gradcam_utils.py              # VGG utilities
│
├── data/                             # Data handling
│   ├── indexed_dataset.py            # Dataset classes
│   ├── fast_cache.py                 # Cache management
│   └── download_data.py              # Data download
│
├── scripts/                          # Utility scripts
│   └── reproduce_results.py          # Result reproduction
│
└── archive/                          # Preserved results
    ├── models/                       # Trained teacher models
    └── results/                      # Training results & logs
```

## Method Details

### Attribution-weighted Knowledge Distillation

MATE-KD introduces a novel loss function that combines:

1. **Classification Loss**: Standard cross-entropy on ground truth labels
2. **Knowledge Distillation Loss**: Weighted combination of teacher logits
3. **Attribution Loss**: L2 distance between student and teacher Grad-CAM maps

### Dynamic Teacher Weighting

Teacher weights are computed using attribution similarity:

```python
similarity = cosine_similarity(student_gradcam, teacher_gradcam)
weights = softmax(similarity / temperature)
```

This allows the student to focus on teachers that produce similar attention patterns for each sample.

## Citation

```bibtex
@misc{mate-kd-2024,
  title={MATE-KD: Multi-teacher Attribution-weighted Knowledge Distillation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MATE-KD}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CIFAR-10 dataset by [Krizhevsky & Hinton](https://www.cs.toronto.edu/~kriz/cifar.html)
- CIFAR-10-C corruptions by [Hendrycks & Dietterich](https://github.com/hendrycks/robustness)
- Grad-CAM implementation inspired by [original paper](https://arxiv.org/abs/1610.02391) 