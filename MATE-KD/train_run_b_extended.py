#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from datetime import datetime
import yaml
from tqdm import tqdm
import pickle
import sys

# Import our models and core components
sys.path.append('.')
from teachers.densenet import DenseNet121
from teachers.vgg import VGG19
from teachers.resnet110 import ResNet110
from student.resnet import ResNet20
from core.knowledge_distillation import SimilarityWeightedKnowledgeDistillation
from core.augmentation import create_augmented_data_loaders

class ExtendedTrainer:
    def __init__(self, config_path="cfg_100_epochs.yaml", resume_checkpoint=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Extended training parameters
        self.total_epochs = 150  # Extended from 100
        self.start_epoch = 0
        self.resume_checkpoint = resume_checkpoint
        
        # Temperature scheduling parameters
        self.initial_temperature = 4.0
        self.final_temperature = 2.0
        
        # Setup results directory
        self.setup_results_dir()
        
        # Initialize models and data
        self.setup_models()
        self.setup_data()
        self.setup_training()
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
    
    def setup_results_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"results/run_b_extended_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results will be saved to: {self.results_dir}")
    
    def setup_models(self):
        """Load teacher models and create student"""
        # Teacher models
        self.teacher_densenet = DenseNet121(num_classes=10).to(self.device)
        self.teacher_vgg = VGG19(num_classes=10).to(self.device)
        self.teacher_resnet = ResNet110(num_classes=10).to(self.device)
        
        # Load teacher weights
        self.teacher_densenet.load_state_dict(torch.load('teachers/densenet121_cifar10.pth', map_location=self.device))
        self.teacher_vgg.load_state_dict(torch.load('teachers/vgg19_cifar10.pth', map_location=self.device))
        self.teacher_resnet.load_state_dict(torch.load('teachers/resnet110_cifar10.pth', map_location=self.device))
        
        # Set teachers to eval mode
        for teacher in [self.teacher_densenet, self.teacher_vgg, self.teacher_resnet]:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
        
        # Student model
        self.student = ResNet20(num_classes=10).to(self.device)
        
        print("Models loaded successfully")
    
    def setup_data(self):
        """Setup data loaders with augmentation"""
        # Load cached CAM data
        with open('data/gradcam_cache_293MB.pkl', 'rb') as f:
            self.gradcam_cache = pickle.load(f)
        print(f"Loaded GradCAM cache: {len(self.gradcam_cache)} samples")
        
        # Create augmented data loaders
        self.train_loader, self.test_loader = create_augmented_data_loaders(
            batch_size=128,
            augment_train=True,
            use_cached_cams=True,
            gradcam_cache=self.gradcam_cache
        )
        print("Data loaders created with augmentation")
    
    def setup_training(self):
        """Setup optimizer, scheduler, and KD loss"""
        # Optimizer and scheduler
        self.optimizer = optim.SGD(
            self.student.parameters(),
            lr=self.config['lr'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_epochs
        )
        
        # Knowledge distillation
        self.kd_loss = SimilarityWeightedKnowledgeDistillation(
            alpha=self.config['alpha'],
            beta=0.0,  # No attribution loss - proven to hurt performance
            temperature=self.initial_temperature
        )
        
        # Tracking variables
        self.best_accuracy = 0.0
        self.train_losses = []
        self.test_accuracies = []
        
        print("Training setup complete")
    
    def get_current_temperature(self, epoch):
        """Temperature scheduling: linearly decrease from initial to final"""
        progress = epoch / self.total_epochs
        temperature = self.initial_temperature - progress * (self.initial_temperature - self.final_temperature)
        return max(temperature, self.final_temperature)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.train_losses = checkpoint.get('train_losses', [])
        self.test_accuracies = checkpoint.get('test_accuracies', [])
        
        print(f"Resumed from epoch {self.start_epoch}, best accuracy: {self.best_accuracy:.2f}%")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.student.train()
        
        # Update temperature for this epoch
        current_temp = self.get_current_temperature(epoch)
        self.kd_loss.temperature = current_temp
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.total_epochs}")
        
        for batch_idx, batch_data in enumerate(pbar):
            if len(batch_data) == 3:  # With cached CAMs
                images, labels, cached_cams = batch_data
                cached_cams = {k: v.to(self.device) for k, v in cached_cams.items()}
            else:  # Without cached CAMs
                images, labels = batch_data
                cached_cams = None
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass through teachers
            with torch.no_grad():
                teacher_outputs = [
                    self.teacher_densenet(images),
                    self.teacher_vgg(images),
                    self.teacher_resnet(images)
                ]
            
            # Student forward pass
            student_output = self.student(images)
            
            # Compute loss
            loss_dict = self.kd_loss(
                student_output=student_output,
                teacher_outputs=teacher_outputs,
                true_labels=labels,
                cached_teacher_cams=cached_cams
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_samples += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'T': f"{current_temp:.2f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.5f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, epoch):
        """Evaluate model on test set"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        self.test_accuracies.append(accuracy)
        
        print(f"Epoch {epoch}: Test Accuracy = {accuracy:.2f}%")
        
        return accuracy
    
    def save_checkpoint(self, epoch, accuracy):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,  # Next epoch to resume from
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.results_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if accuracy > self.best_accuracy:
            torch.save(checkpoint, os.path.join(self.results_dir, f'best_model_{accuracy:.2f}pct.pth'))
            print(f"New best model saved: {accuracy:.2f}%")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.total_epochs} epochs")
        print(f"Temperature schedule: {self.initial_temperature} -> {self.final_temperature}")
        
        for epoch in range(self.start_epoch, self.total_epochs):
            # Train one epoch
            avg_loss = self.train_epoch(epoch)
            
            # Evaluate
            accuracy = self.evaluate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save checkpoint
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
            self.save_checkpoint(epoch, accuracy)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%, Best = {self.best_accuracy:.2f}%")
            print("-" * 80)
        
        print(f"Training complete! Best accuracy: {self.best_accuracy:.2f}%")

if __name__ == "__main__":
    # Training configuration
    config_path = "cfg.yaml"
    
    # Check for resume checkpoint
    resume_checkpoint = None
    if len(sys.argv) > 1:
        resume_checkpoint = sys.argv[1]
        print(f"Resuming from checkpoint: {resume_checkpoint}")
    
    # Create trainer and start training
    trainer = ExtendedTrainer(config_path=config_path, resume_checkpoint=resume_checkpoint)
    trainer.train() 