"""
Script to run electrode reduction study and analyze optimal electrode configurations.
"""

import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.preprocessing import PreprocessingConfig, EMGPreprocessor
from src.visualization.electrode_viz import ElectrodeVisualizer
from src.models.model import SilentSpeechModel  # Assuming this is your model class

class ElectrodeStudy:
    """Class for conducting electrode reduction studies."""
    
    def __init__(self,
                 data_dir: str,
                 output_dir: str,
                 config: PreprocessingConfig):
        """Initialize the study."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.preprocessor = EMGPreprocessor(config)
        self.visualizer = ElectrodeVisualizer()
        
        # Initialize results dictionary
        self.results = {
            'importance_scores': {},
            'accuracy_vs_electrodes': {},
            'optimal_configs': {},
            'confusion_matrix': None
        }
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the EMG data."""
        # TODO: Implement data loading based on your data format
        # This is a placeholder that should be adapted to your data structure
        data = np.load(self.data_dir / 'emg_data.npy')
        labels = np.load(self.data_dir / 'labels.npy')
        
        return data, labels
    
    def compute_electrode_importance(self,
                                   data: np.ndarray,
                                   labels: np.ndarray) -> Dict[str, float]:
        """Compute importance scores for each electrode using mutual information."""
        from sklearn.feature_selection import mutual_info_classif
        
        importance_scores = {}
        n_channels = data.shape[2]  # Assuming shape is (n_samples, time_steps, n_channels)
        
        # Compute feature importance for each channel
        for i in range(n_channels):
            channel_data = data[:, :, i].reshape(len(data), -1)
            importance = mutual_info_classif(channel_data, labels, random_state=42)
            importance_scores[f'Electrode {i+1}'] = float(np.mean(importance))
            
        return importance_scores
    
    def evaluate_electrode_subset(self,
                                data: np.ndarray,
                                labels: np.ndarray,
                                electrode_indices: List[int]) -> float:
        """Evaluate model performance with a subset of electrodes."""
        # Select data for specified electrodes
        subset_data = data[:, :, electrode_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            subset_data, labels, test_size=0.2, random_state=42
        )
        
        # Preprocess data
        X_train_processed = np.stack([
            self.preprocessor.process_trial(x)[0] for x in X_train
        ])
        X_test_processed = np.stack([
            self.preprocessor.process_trial(x)[0] for x in X_test
        ])
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_processed)
        X_test_tensor = torch.FloatTensor(X_test_processed)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Create data loaders
        train_loader = DataLoader(
            list(zip(X_train_tensor, y_train_tensor)),
            batch_size=32,
            shuffle=True
        )
        test_loader = DataLoader(
            list(zip(X_test_tensor, y_test_tensor)),
            batch_size=32
        )
        
        # Initialize and train model
        model = SilentSpeechModel(
            input_channels=len(electrode_indices),
            # Add other model parameters as needed
        )
        
        # Train model (simplified version)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(10):  # Simplified training loop
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate model
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        if len(electrode_indices) == data.shape[2]:  # If using all electrodes
            self.results['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
            
        return accuracy
    
    def find_optimal_subsets(self,
                           data: np.ndarray,
                           labels: np.ndarray,
                           min_electrodes: int = 3) -> Dict[int, List[int]]:
        """Find optimal electrode subsets of different sizes."""
        n_channels = data.shape[2]
        importance_scores = self.compute_electrode_importance(data, labels)
        self.results['importance_scores'] = importance_scores
        
        # Sort electrodes by importance
        sorted_electrodes = sorted(
            range(n_channels),
            key=lambda i: importance_scores[f'Electrode {i+1}'],
            reverse=True
        )
        
        optimal_subsets = {}
        accuracies = {}
        
        # Evaluate performance with different numbers of electrodes
        for n_electrodes in range(min_electrodes, n_channels + 1):
            subset = sorted_electrodes[:n_electrodes]
            accuracy = self.evaluate_electrode_subset(data, labels, subset)
            
            optimal_subsets[n_electrodes] = subset
            accuracies[n_electrodes] = accuracy
            
            print(f'Accuracy with {n_electrodes} electrodes: {accuracy:.4f}')
            
        self.results['accuracy_vs_electrodes'] = accuracies
        self.results['optimal_configs'] = {
            k: [int(i) for i in v] for k, v in optimal_subsets.items()
        }
        
        return optimal_subsets
    
    def run_study(self) -> Dict:
        """Run the complete electrode reduction study."""
        print("Loading data...")
        data, labels = self.load_data()
        
        print("Finding optimal electrode configurations...")
        self.find_optimal_subsets(data, labels)
        
        print("Generating visualizations...")
        self.visualizer.plot_feature_importance(self.results['importance_scores'])
        self.visualizer.save_figure(f'{self.output_dir}/feature_importance.png')
        
        self.visualizer.plot_electrode_reduction_impact(self.results['accuracy_vs_electrodes'])
        self.visualizer.save_figure(f'{self.output_dir}/accuracy_vs_electrodes.png')
        
        if self.results['confusion_matrix'] is not None:
            self.visualizer.plot_confusion_matrix(self.results['confusion_matrix'])
            self.visualizer.save_figure(f'{self.output_dir}/confusion_matrix.png')
        
        # Save results
        with open(self.output_dir / 'study_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
            
        return self.results

def main():
    parser = argparse.ArgumentParser(description='Run electrode reduction study')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing EMG data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--min_electrodes', type=int, default=3,
                       help='Minimum number of electrodes to consider')
    
    args = parser.parse_args()
    
    config = PreprocessingConfig()
    study = ElectrodeStudy(args.data_dir, args.output_dir, config)
    results = study.run_study()
    
    print("\nStudy completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 