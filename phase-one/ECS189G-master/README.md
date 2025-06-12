# Phase One: Single-Teacher Attribution-Enhanced Knowledge Distillation

## Overview

Phase One establishes the foundation for attribution-enhanced knowledge distillation by evaluating different explanation methods in a single-teacher setting. This phase focuses on understanding how well student models can learn to mimic not just the predictions, but also the reasoning patterns of their teacher models.

## Objectives

1. **Attribution Method Comparison**: Evaluate three different attribution techniques:
   - **Grad-CAM**: Class Activation Mapping using gradients
   - **Grad-CAM++**: Enhanced version with better localization
   - **Integrated Gradients**: Path-based attribution method

2. **Alignment Measurement**: Develop metrics to quantify how well student attribution maps align with teacher explanations

3. **Baseline Establishment**: Create performance benchmarks for explanation-guided knowledge distillation

4. **Method Selection**: Identify the most effective attribution method for multi-teacher extension

## Technical Approach

### Architecture Setup
- **Teacher Model**: ResNet-56 (larger capacity)
- **Student Model**: ResNet-20 (compressed version)
- **Dataset**: CIFAR-10 for training and evaluation
- **Robustness Testing**: CIFAR-10-C for out-of-distribution evaluation

### Attribution Alignment Process

1. **Forward Pass**: Both teacher and student process the same input
2. **Attribution Generation**: Compute explanation maps using selected method
3. **Alignment Calculation**: Measure similarity between teacher and student attributions
4. **Loss Integration**: Combine standard KD loss with attribution alignment loss

### Alignment Metric

```python
def compute_alignment(student_model, teacher_model, explainer_name):
    # Generate attribution maps
    teacher_maps = get_attribution(teacher_model, x, y)
    student_maps = get_attribution(student_model, x, y)
    
    # Normalize to [0,1] range
    teacher_norm = normalize_attribution(teacher_maps)
    student_norm = normalize_attribution(student_maps)
    
    # Compute MSE and convert to alignment score
    mse = ((teacher_norm - student_norm) ** 2).mean()
    alignment_score = 1.0 - mse
    
    return alignment_score
```



### Key Functions

#### `evaluate_student(student_path, teacher_path, explainer_name)`
Comprehensive evaluation of a student model including:
- Clean CIFAR-10 accuracy
- Attribution alignment with teacher
- MSE between attribution maps

#### `compute_alignment(student_model, teacher_model, explainer_name)`
Measures how well student attributions match teacher explanations:
- Supports 'ig', 'cam', 'campp' explainers
- Returns MSE and normalized alignment score
- Uses 1000 test samples for efficiency

#### `evaluate_cifar10c_subset()` / `evaluate_cifar10c_full_avg()`
Robustness evaluation on corrupted images:
- Tests on CIFAR-10-C perturbations
- Measures out-of-distribution performance
- Supports both subset and full corruption evaluation

## Usage Examples

### 🚀 Easiest Way: Jupyter Notebook
1. Open `ECS189G.ipynb`
2. Run all cells
3. View results automatically printed

### Programmatic Usage

#### Basic Student Evaluation
```python
from evaluation import evaluate_student

# Evaluate the best model
results = evaluate_student(
    student_path='Checkpoints/student_cam_opt_b_200ep.pth',
    teacher_path='Checkpoints/resnet56_teacher_best.pth',
    explainer_name='cam'
)

print(f"Clean Accuracy: {results['clean_accuracy']:.2f}%")
print(f"Alignment Score: {results['alignment_score']:.4f}")
```

#### Comprehensive Comparison
```python
# Compare all models (as done in notebook)
student_paths = {
    'cam': 'Checkpoints/student_cam.pth', 
    'campp': 'Checkpoints/student_campp.pth',
    'best_path': 'Checkpoints/student_cam_opt_b_200ep.pth'
}

results = {}
for explainer_name, student_path in student_paths.items():
    results[explainer_name] = evaluate_student(
        student_path=student_path,
        teacher_path='Checkpoints/resnet56_teacher_best.pth',
        explainer_name=explainer_name
    )

# Print results
for explainer_name, model_results in results.items():
    print(f"{explainer_name.upper()}:")
    print(f"  Clean Accuracy: {model_results['clean_accuracy']:.2f}%")
    print(f"  Alignment Score: {model_results['alignment_score']:.4f}")
```

### Robustness Evaluation
```python
from evaluation import evaluate_model_comprehensive

# Full evaluation including CIFAR-10-C
results = evaluate_model_comprehensive(
    model_path='Checkpoints/student_cam_opt_b_200ep.pth',
    model_type='resnet20',
    model_name='Best CAM Student'
)
```

## Model Details

### Teacher Model
- **Architecture**: ResNet-56
- **Parameters**: ~0.85M
- **Clean Accuracy**: ~92% on CIFAR-10
- **Location**: `Checkpoints/resnet56_teacher_best.pth`

### Best Student Model
- **Architecture**: ResNet-20
- **Parameters**: ~0.27M (3x compression)
- **Training**: 200 epochs with Grad-CAM attribution alignment
- **Clean Accuracy**: ~88-90%
- **Location**: `Checkpoints/student_cam_opt_b_200ep.pth`

## Dependencies

```bash
pip install torch torchvision numpy tqdm matplotlib
```

Required data:
- CIFAR-10 dataset (auto-downloaded)
- CIFAR-10-C dataset (for robustness testing)
- Pre-trained model weights (included in Checkpoints/)

## Key Insights

1. **Attribution Alignment Matters**: Students with better attribution alignment show improved robustness
2. **Grad-CAM Effectiveness**: Outperforms other methods in both accuracy and alignment
3. **Robustness Transfer**: Explanation-guided distillation improves out-of-distribution performance
4. **Computational Trade-offs**: Grad-CAM offers best efficiency vs. performance balance

## Limitations

- Single teacher model (addressed in Phase Two)
- Limited to vision tasks (CIFAR-10)
- Attribution methods have different computational costs
- Alignment metric assumes spatial correspondence

## Next Steps → Phase Two

Phase One findings inform Phase Two development:
- **Selected Method**: Grad-CAM for multi-teacher framework
- **Baseline Performance**: Established single-teacher benchmarks
- **Alignment Metric**: Validated attribution similarity measurement
- **Architecture**: Proven ResNet teacher-student setup

The multi-teacher extension in Phase Two builds upon these foundations to achieve:
- Dynamic teacher weighting
- Instance-wise teacher selection
- Improved robustness and interpretability

---

**🎯 Quick Start: Open `ECS189G.ipynb` and run all cells to see the complete evaluation results!**

**Phase One establishes that attribution-enhanced knowledge distillation is both feasible and beneficial, setting the stage for the multi-teacher innovations in Phase Two.**
