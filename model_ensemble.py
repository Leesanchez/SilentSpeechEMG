import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import wandb
from tqdm import tqdm
import os

from spatiotemporal_model import SilentSpeechTransformer
from model_utils import compute_wer_cer, decode_predictions, decode_targets, train_model
from visualization_utils import plot_confusion_matrix, plot_training_metrics

class ModelEnsemble:
    def __init__(self, num_models: int, model_params: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_models = num_models
        self.model_params = model_params
        self.device = device
        self.models = []
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize multiple model instances with different random seeds."""
        for i in range(self.num_models):
            torch.manual_seed(i)  # Different seed for each model
            model = SilentSpeechTransformer(**self.model_params).to(self.device)
            self.models.append(model)
    
    def train_with_cross_validation(self, dataset, num_folds: int, batch_size: int,
                                  num_epochs: int, collate_fn=None):
        """Train ensemble using k-fold cross validation."""
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        # Store metrics for each fold
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\nTraining Fold {fold + 1}/{num_folds}")
            
            # Create data loaders for this fold
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                dataset, batch_size=batch_size,
                sampler=train_sampler, collate_fn=collate_fn,
                num_workers=4, pin_memory=True
            )
            
            val_loader = DataLoader(
                dataset, batch_size=batch_size,
                sampler=val_sampler, collate_fn=collate_fn,
                num_workers=4, pin_memory=True
            )
            
            # Train each model in the ensemble
            fold_model_metrics = []
            for model_idx, model in enumerate(self.models):
                print(f"\nTraining Model {model_idx + 1}/{self.num_models}")
                
                # Initialize new wandb run for this model
                run_name = f"fold{fold+1}_model{model_idx+1}"
                wandb.init(
                    project="silent-speech-interface",
                    name=run_name,
                    group=f"fold_{fold+1}",
                    reinit=True
                )
                
                # Train the model using train_model from model_utils
                metrics = train_model(
                    model, train_loader, val_loader,
                    num_epochs=num_epochs, device=self.device
                )
                
                fold_model_metrics.append(metrics)
                wandb.finish()
            
            fold_metrics.append(fold_model_metrics)
            
            # Save models for this fold
            self.save_models(f"fold_{fold+1}")
        
        # Analyze cross-validation results
        self.analyze_cv_results(fold_metrics)
    
    def predict(self, emg_data: torch.Tensor) -> Tuple[str, float]:
        """Make prediction using ensemble of models."""
        all_outputs = []
        
        # Get predictions from each model
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(emg_data.to(self.device))
                all_outputs.append(outputs)
        
        # Average predictions
        ensemble_output = torch.mean(torch.stack(all_outputs), dim=0)
        
        # Get confidence scores
        probs = torch.softmax(ensemble_output, dim=-1)
        confidence = torch.max(probs, dim=-1)[0].mean().item()
        
        return ensemble_output, confidence
    
    def save_models(self, prefix: str):
        """Save all models in the ensemble."""
        save_dir = f"models/{prefix}"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{save_dir}/model_{i+1}.pth")
    
    def load_models(self, prefix: str):
        """Load all models in the ensemble."""
        load_dir = f"models/{prefix}"
        
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(f"{load_dir}/model_{i+1}.pth"))
            model.eval()
    
    def analyze_cv_results(self, fold_metrics: List[List[Dict[str, float]]]):
        """Analyze and visualize cross-validation results."""
        # Calculate mean and std of metrics across folds
        all_metrics = {}
        for metric in ['val_loss', 'val_acc', 'val_wer', 'val_cer']:
            values = []
            for fold in fold_metrics:
                for model_metrics in fold:
                    values.append(model_metrics[metric])
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            all_metrics[metric] = {'mean': mean_val, 'std': std_val}
        
        # Log to wandb
        wandb.init(project="silent-speech-interface", name="cv_analysis", reinit=True)
        for metric, stats in all_metrics.items():
            wandb.log({
                f"{metric}/mean": stats['mean'],
                f"{metric}/std": stats['std']
            })
        
        # Create visualization of metrics distribution
        plot_training_metrics(
            {k: [m[k] for fold in fold_metrics for m in fold] 
             for k in ['val_loss', 'val_acc', 'val_wer', 'val_cer']},
            save_path="cv_metrics.png"
        )
        wandb.log({"cv_metrics": wandb.Image("cv_metrics.png")})
        wandb.finish()

def create_ensemble(num_models: int, vocab_size: int) -> ModelEnsemble:
    """Create an ensemble of models with specified parameters."""
    model_params = {
        'num_channels': 8,
        'input_size': 128,
        'num_classes': vocab_size
    }
    
    return ModelEnsemble(num_models, model_params) 