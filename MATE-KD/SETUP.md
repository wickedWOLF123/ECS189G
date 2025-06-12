# MATE-KD Setup Instructions

This guide provides step-by-step instructions to set up and run the MATE-KD project.

## System Requirements

- **Python**: 3.10 or higher
- **CUDA**: 11.7+ (for GPU training)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB free space for datasets and models

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate mate-kd
```

### Option 2: Virtual Environment + Pip

```bash
# Create virtual environment
python -m venv mate-kd-env
source mate-kd-env/bin/activate  # Linux/Mac
# mate-kd-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Manual Installation

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn
pip install scikit-learn opencv-python pillow
pip install tqdm pyyaml jupyter
pip install uv  # For faster package management
```

## Data Setup

### Automatic Download (Recommended)

```bash
# Download CIFAR-10
python data/download_data.py --dataset cifar10 --root ./data

# Download CIFAR-10-C (corruption benchmark)
python data/download_data.py --dataset cifar10c --root ./data
```

### Manual Download

1. **CIFAR-10**:
   ```bash
   wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   tar -xzf cifar-10-python.tar.gz
   mv cifar-10-batches-py ./data/
   ```

2. **CIFAR-10-C**:
   ```bash
   mkdir -p data/CIFAR-10-C
   # Download from: https://zenodo.org/record/2535967
   # Extract all .npy files to data/CIFAR-10-C/
   ```

## Pre-trained Models

### Teacher Models

The repository includes pre-trained teacher models in `archive/models/`:
- `resnet110_best.pth`
- `densenet121_best.pth` 
- `vgg19_best.pth`

If missing, train them with:
```bash
python teachers/train_resnet110.py
python teachers/train_densenet121.py
python teachers/train_vgg19.py
```

### Student Model

The best student model is available at:
`archive/results/best_student_90.22pct/best_model_90.22pct.pth`

## Training Setup

### Generate Teacher Attribution Cache (Optional)

For faster training, pre-compute teacher Grad-CAM maps:

```bash
# Generate cache for all teachers
python data/precompute_gradcam_cache.py --models resnet densenet vgg19
```

This creates attribution maps in `data/gradcam_cache/` (excluded from git).

### Configuration

Edit `cfg.yaml` to customize training:

```yaml
# Training parameters
batch_size: 256
epochs: 100
learning_rate: 0.0005

# Loss weights
alpha_ce: 0.2      # Classification loss
beta_map: 0.15     # Attribution loss

# Teacher models
teachers:
  - resnet
  - densenet  
  - vgg19
```

## Running the Code

### 1. Training

```bash
# Full training run
python train_run_b_extended.py --config cfg.yaml

# Quick test (few epochs)
python train_run_b_extended.py --config cfg.yaml --epochs 5
```

### 2. Evaluation

```bash
# Reproduce main results
python scripts/reproduce_results.py

# Evaluate specific model
python scripts/reproduce_results.py --model path/to/model.pth
```

### 3. Analysis Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open MATE-KD.ipynb
# Run all cells for complete analysis
```

## Verification

### Quick Test

```bash
# Test data loading
python -c "
from data.indexed_dataset import CIFAR10WithIndex
dataset = CIFAR10WithIndex(root='./data/cifar-10-batches-py', train=False)
print(f'Dataset loaded: {len(dataset)} samples')
"

# Test model loading
python -c "
from student.zoo import build_student
model = build_student(num_classes=10)
print(f'Student model: {sum(p.numel() for p in model.parameters())} parameters')
"
```

### Expected Outputs

1. **Training**: Should reach ~88-90% validation accuracy
2. **Evaluation**: Produces accuracy tables and visualizations
3. **Notebook**: Generates comprehensive analysis plots

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size in cfg.yaml
   batch_size: 128  # or smaller
   ```

2. **Missing Data Files**:
   ```bash
   # Re-run data download
   python data/download_data.py --dataset cifar10 --force
   ```

3. **Import Errors**:
   ```bash
   # Ensure current directory is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

4. **Slow Training**:
   ```bash
   # Use pre-computed teacher cache
   python data/precompute_gradcam_cache.py --models resnet densenet vgg19
   ```

### GPU Setup

For optimal performance:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Monitor GPU usage during training
nvidia-smi -l 1
```

### Memory Optimization

If running out of memory:

1. Reduce batch size: `batch_size: 64`
2. Use gradient checkpointing: `gradient_checkpointing: true`
3. Reduce cache size: `cache_size: 5000`

## Development Setup

For development and experimentation:

```bash
# Install in development mode
pip install -e .

# Pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Jupyter extensions
pip install jupyterlab-widgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Performance Expectations

### Training Time
- **With cache**: ~4-6 minutes/epoch (GTX 1080 Ti)
- **Without cache**: ~2-3 hours/epoch
- **Full training**: 6-10 hours with cache

### Memory Usage
- **Training**: ~4-6GB GPU memory
- **Evaluation**: ~2-3GB GPU memory
- **Cache generation**: ~8-10GB GPU memory

## Next Steps

After successful setup:

1. Run the complete training pipeline
2. Explore the analysis notebook
3. Experiment with different configurations
4. Try custom teacher models or datasets

For questions or issues, please check the troubleshooting section or open an issue on GitHub. 