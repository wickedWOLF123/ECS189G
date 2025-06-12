# MATE-KD: Multi-Teacher Attribution-Enhanced Knowledge Distillation

**ECS189G - Trustworthiness in AI Course Project**

## Overview

MATE-KD (Multi-Teacher Attribution-Enhanced Knowledge Distillation) is a novel framework that enhances knowledge distillation by incorporating explanation consistency between teacher and student models. Unlike conventional knowledge distillation approaches that only focus on output distributions, our method ensures that student networks learn both predictive accuracy and interpretable reasoning patterns from their teachers.

## Key Innovation

Traditional knowledge distillation ignores the reasoning patterns underlying teacher predictions. MATE-KD addresses this by:

- **Attribution-Aware Distillation**: Aligning explanation maps between teacher and student models
- **Dynamic Multi-Teacher Weighting**: Instance-wise teacher selection based on attribution similarity
- **Improved Interpretability**: Ensuring student models learn not just what to predict, but how to reason

## Project Structure

This project is organized into two main phases:

### Phase 1: Single-Teacher Attribution Evaluation
- Evaluates different attribution methods (Grad-CAM, Grad-CAM++, Integrated Gradients)
- Establishes baseline performance for explanation-guided knowledge distillation
- Identifies Grad-CAM as the most effective attribution method
- See `phase-one/README.md` for detailed implementation

### Phase 2: Multi-Teacher Attribution-Enhanced KD ⭐ (Main Contribution)
- Extends to multi-teacher distillation framework
- Implements dynamic teacher weighting using cosine similarity between Grad-CAM attribution maps
- Enables selective learning from the most relevant teacher for each input instance
- See `phase-two/README.md` for detailed implementation

## Results Summary

### CIFAR-10 Performance
- **Accuracy**: 90.22% (competitive with state-of-the-art)
- **Robustness**: Improved out-of-distribution performance on CIFAR-10-C
  - MATE-KD: 67.46%
  - Baseline: 65.47%
- **Interpretability**: Enhanced alignment between teacher and student attribution maps

### Key Findings
1. **Grad-CAM** outperforms other attribution methods for explanation alignment
2. **Multi-teacher approach** 
3. **Attribution consistency** transfers interpretable reasoning patterns alongside predictive knowledge


## Course Context

This project was developed for **ECS189G - Trustworthiness in AI** at UC Davis, exploring how knowledge distillation can be enhanced to produce not only accurate but also interpretable and robust compressed models. The work addresses key trustworthiness concerns including transparency, robustness, and explainability in deep neural networks.

## Future Work

- Extension to other domains (NLP, medical imaging)
- Integration with other explanation methods (LIME, SHAP)
- Theoretical analysis of attribution alignment guarantees
- Scaling to larger teacher-student architecture gaps
