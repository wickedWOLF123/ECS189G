"""
Hyperparameter Grid Search for Knowledge Distillation
Quick 10-epoch validation passes to find optimal α, β, T
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from itertools import product

# Import training components
from data.indexed_dataset import get_indexed_cifar10_loaders
from data.fast_cache import FastTeacherCache
from core.student_gradcam import EfficientStudentGradCAM
from student.zoo import load_teacher, ResNet20

class HyperparameterSearcher:
    """Grid search for optimal KD hyperparameters"""
    
    def __init__(self, cache_dir='data/gradcam_cache', device='cuda'):
        self.cache_dir = cache_dir
        self.device = device
        
        # Save directory for search results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(f'results/hyperparam_search_{timestamp}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print("HYPERPARAMETER GRID SEARCH")
        print("=" * 50)
        print(f"Results Directory: {self.save_dir}")
        
        self._setup_data()
        self._setup_teachers()
        self._setup_cache()
        
        # Results storage
        self.search_results = []
        
        print("\nSearch system ready!")
    
    def _setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        self.train_loader, self.test_loader = get_indexed_cifar10_loaders(
            batch_size=128,
            num_workers=4,
            root='./data',
            augment_train=True,
            shuffle=True
        )
        
        print(f"  Train: {len(self.train_loader)} batches")
        print(f"  Test: {len(self.test_loader)} batches")
    
    def _setup_teachers(self):
        """Setup teacher models"""
        print("Loading teachers...")
        
        teacher_configs = {
            'densenet': 'teachers/best_models/densenet121_best.pth',
            'vgg19': 'teachers/best_models/vgg19_best.pth', 
            'resnet': 'teachers/best_models/resnet110_best.pth'
        }
        
        self.teachers = {}
        for name, path in teacher_configs.items():
            self.teachers[name] = load_teacher(path, name)
        
        for name, teacher in self.teachers.items():
            teacher.to(self.device)
            teacher.eval()
        
        print(f"  Teachers loaded: {list(self.teachers.keys())}")
    
    def _setup_cache(self):
        """Setup fast memory-resident cache"""
        print("Setting up cache...")
        
        self.cache = FastTeacherCache(self.cache_dir, device=self.device)
        cache_stats = self.cache.get_cache_stats()
        
        print(f"  Cache loaded: {cache_stats['memory_usage_mb']:.1f} MB")
    
    def _compute_similarities(self, student_maps, teacher_maps):
        """Compute mean-free cosine similarities"""
        batch_size = student_maps.shape[0]
        similarities = {}
        
        student_flat = student_maps.view(batch_size, -1)
        student_centered = student_flat - student_flat.mean(dim=1, keepdim=True)
        student_norm = F.normalize(student_centered, p=2, dim=1)
        
        for teacher_name, teacher_heatmaps in teacher_maps.items():
            teacher_flat = teacher_heatmaps.view(batch_size, -1)
            teacher_centered = teacher_flat - teacher_flat.mean(dim=1, keepdim=True)
            teacher_norm = F.normalize(teacher_centered, p=2, dim=1)
            
            similarity = F.relu(torch.sum(student_norm * teacher_norm, dim=1))
            similarities[teacher_name] = similarity
        
        return similarities
    
    def _compute_teacher_weights(self, similarities, tau=0.3):
        """Compute teacher weights using temperature-scaled softmax"""
        teacher_names = list(similarities.keys())
        batch_size = similarities[teacher_names[0]].shape[0]
        
        sim_stack = torch.stack([similarities[name] for name in teacher_names], dim=1)
        sim_max = sim_stack.max(dim=1, keepdim=True)[0]
        sim_scaled = sim_stack / (sim_max + 1e-6)
        
        weights = F.softmax(sim_scaled / tau, dim=1)
        
        teacher_weights = {}
        for i, name in enumerate(teacher_names):
            teacher_weights[name] = weights[:, i]
        
        return teacher_weights, sim_scaled.mean().item()
    
    def evaluate_hyperparameters(self, alpha_ce, beta_map, T_kd, num_epochs=10):
        """Evaluate a single hyperparameter combination"""
        
        print(f"\nTesting: α={alpha_ce:.2f}, β={beta_map:.2f}, T={T_kd:.1f}")
        
        # Create fresh student model
        student = ResNet20(num_classes=10).to(self.device)
        student_gradcam = EfficientStudentGradCAM(student, target_layer_name='layer3')
        
        # Optimizer
        optimizer = optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_accuracy = 0.0
        epoch_accuracies = []
        final_loss_components = {}
        
        for epoch in range(num_epochs):
            student.train()
            
            epoch_loss_ce = 0.0
            epoch_loss_kd = 0.0
            epoch_loss_map = 0.0
            num_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
            for batch_idx, (images, labels, indices) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                indices = indices.to(self.device)
                
                # Student forward + Grad-CAM
                student_logits, student_maps = student_gradcam.forward_with_gradcam(images)
                
                # Teacher forward
                teacher_logits = {}
                for teacher_name, teacher in self.teachers.items():
                    with torch.no_grad():
                        teacher_logits[teacher_name] = teacher(images)
                
                # Get cached heatmaps
                teacher_maps = self.cache.get_all_teacher_heatmaps_batch(indices)
                
                # Compute similarities and weights
                similarities = self._compute_similarities(student_maps, teacher_maps)
                tau = max(0.05, 0.3 * (1 - epoch / num_epochs))
                teacher_weights, mean_sim = self._compute_teacher_weights(similarities, tau)
                
                # Loss computation
                loss_ce = F.cross_entropy(student_logits, labels)
                
                # Weighted KD Loss
                loss_kd = 0.0
                T = T_kd
                
                for teacher_name, teacher_logit in teacher_logits.items():
                    weights = teacher_weights[teacher_name].unsqueeze(1)
                    
                    student_soft = F.log_softmax(student_logits / T, dim=1)
                    teacher_soft = F.softmax(teacher_logit / T, dim=1)
                    
                    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='none').sum(dim=1)
                    weighted_kd = (weights.squeeze() * kd_loss).mean()
                    loss_kd += weighted_kd
                
                loss_kd *= (T * T)
                
                # Attribution MSE Loss
                loss_map = 0.0
                for teacher_name in teacher_maps.keys():
                    teacher_attribution = teacher_maps[teacher_name]
                    weight = teacher_weights[teacher_name].unsqueeze(-1).unsqueeze(-1)
                    
                    # Normalize attributions for better MSE comparison
                    s_norm = F.normalize(student_maps.view(student_maps.size(0), -1), p=2, dim=1)
                    t_norm = F.normalize(teacher_attribution.view(teacher_attribution.size(0), -1), p=2, dim=1)
                    
                    mse = F.mse_loss(s_norm, t_norm, reduction='none').sum(dim=1)
                    weighted_mse = (weight.squeeze() * mse).mean()
                    loss_map += weighted_mse
                
                # Total loss
                total_loss = alpha_ce * loss_ce + (1 - alpha_ce) * loss_kd + beta_map * loss_map
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Track loss components
                epoch_loss_ce += loss_ce.item()
                epoch_loss_kd += loss_kd.item()
                epoch_loss_map += loss_map.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'CE': f"{loss_ce.item():.3f}",
                    'KD': f"{loss_kd.item():.3f}",
                    'Map': f"{loss_map.item():.3f}",
                    'Sim': f"{mean_sim:.3f}"
                })
                
                # Early break for quick search
                if batch_idx >= 100:  # Only train on ~100 batches for speed
                    break
            
            scheduler.step()
            
            # Quick evaluation
            accuracy = self._quick_eval(student)
            epoch_accuracies.append(accuracy)
            best_accuracy = max(best_accuracy, accuracy)
            
            print(f"  Epoch {epoch+1}: Acc={accuracy:.2f}%, Best={best_accuracy:.2f}%")
        
        # Store final loss components for analysis
        final_loss_components = {
            'ce': epoch_loss_ce / num_batches,
            'kd': epoch_loss_kd / num_batches, 
            'map': epoch_loss_map / num_batches
        }
        
        return best_accuracy, epoch_accuracies, final_loss_components
    
    def _quick_eval(self, student, max_batches=20):
        """Quick evaluation on subset of test set"""
        student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(self.test_loader):
                if batch_idx >= max_batches:
                    break
                    
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = student(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total
    
    def run_grid_search(self):
        """Run complete hyperparameter grid search"""
        print("\nStarting grid search...")
        
        # Define search space
        alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # CE weight
        beta_values = [0.0, 0.1, 0.5, 1.0]        # Attribution weight  
        T_values = [3.0, 4.0, 5.0, 6.0]           # Temperature
        
        total_combinations = len(alpha_values) * len(beta_values) * len(T_values)
        print(f"  Total combinations: {total_combinations}")
        
        combination_idx = 0
        
        for alpha, beta, T in product(alpha_values, beta_values, T_values):
            combination_idx += 1
            
            start_time = time.time()
            best_acc, epoch_accs, loss_components = self.evaluate_hyperparameters(
                alpha, beta, T, num_epochs=10
            )
            elapsed = time.time() - start_time
            
            # Store results
            result = {
                'combination': combination_idx,
                'alpha_ce': alpha,
                'beta_map': beta,
                'T_kd': T,
                'best_accuracy': best_acc,
                'final_accuracy': epoch_accs[-1] if epoch_accs else 0.0,
                'epoch_accuracies': epoch_accs,
                'loss_components': loss_components,
                'elapsed_time': elapsed
            }
            
            self.search_results.append(result)
            
            print(f"  [{combination_idx}/{total_combinations}] Best: {best_acc:.2f}% ({elapsed:.1f}s)")
            
            # Save intermediate results
            if combination_idx % 5 == 0:
                self.save_results()
        
        print("\nGrid search complete!")
        self.analyze_results()
    
    def save_results(self):
        """Save search results to JSON"""
        results_file = self.save_dir / 'search_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.search_results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    def analyze_results(self):
        """Analyze and report best hyperparameters"""
        if not self.search_results:
            print("No results to analyze!")
            return
        
        # Sort by best accuracy
        sorted_results = sorted(self.search_results, key=lambda x: x['best_accuracy'], reverse=True)
        
        print("\nTOP 5 HYPERPARAMETER COMBINATIONS:")
        print("-" * 60)
        print(f"{'Rank':<4} {'α':<5} {'β':<5} {'T':<5} {'Best Acc':<10} {'Final Acc':<10}")
        print("-" * 60)
        
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1:<4} {result['alpha_ce']:<5.1f} {result['beta_map']:<5.1f} "
                  f"{result['T_kd']:<5.1f} {result['best_accuracy']:<10.2f} "
                  f"{result['final_accuracy']:<10.2f}")
        
        # Best combination analysis
        best = sorted_results[0]
        print(f"\nBEST COMBINATION:")
        print(f"  α (CE weight): {best['alpha_ce']}")
        print(f"  β (Map weight): {best['beta_map']}")
        print(f"  T (Temperature): {best['T_kd']}")
        print(f"  Best accuracy: {best['best_accuracy']:.2f}%")
        print(f"  Final accuracy: {best['final_accuracy']:.2f}%")
        
        # Save best configuration
        best_config = {
            'alpha': best['alpha_ce'],
            'beta': best['beta_map'], 
            'temperature': best['T_kd'],
            'best_accuracy': best['best_accuracy']
        }
        
        config_file = self.save_dir / 'best_config.json'
        with open(config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\nBest config saved to: {config_file}")
        
        # Effect analysis
        print(f"\nHYPERPARAMETER EFFECTS:")
        self._analyze_parameter_effects()
    
    def _analyze_parameter_effects(self):
        """Analyze individual parameter effects"""
        import pandas as pd
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.search_results)
        
        # Alpha effect
        alpha_effects = df.groupby('alpha_ce')['best_accuracy'].agg(['mean', 'std', 'max'])
        print(f"\nAlpha (CE weight) effects:")
        print(alpha_effects.round(2))
        
        # Beta effect  
        beta_effects = df.groupby('beta_map')['best_accuracy'].agg(['mean', 'std', 'max'])
        print(f"\nBeta (Map weight) effects:")
        print(beta_effects.round(2))
        
        # Temperature effect
        temp_effects = df.groupby('T_kd')['best_accuracy'].agg(['mean', 'std', 'max'])
        print(f"\nTemperature effects:")
        print(temp_effects.round(2))

def main():
    """Run hyperparameter search"""
    searcher = HyperparameterSearcher()
    searcher.run_grid_search()
    searcher.save_results()

if __name__ == "__main__":
    main() 