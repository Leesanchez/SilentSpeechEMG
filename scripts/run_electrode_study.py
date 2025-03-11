#!/usr/bin/env python
"""
Script to run the electrode reduction study using synthetic data and trained models.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
import wandb

from src.utils.electrode_analysis import ElectrodeAnalyzer
from src.data.synthetic_data import generate_synthetic_data
from src.models.model import SilentSpeechModel

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_wandb(config):
    """Initialize Weights & Biases logging."""
    return wandb.init(
        project=config['logging']['wandb']['project'],
        entity=config['logging']['wandb']['entity'],
        config=config,
        name='electrode_reduction_study'
    )

def generate_study_data(config, n_samples=1000):
    """Generate synthetic data for the study."""
    print("Generating synthetic data...")
    signals = []
    texts = []
    
    for _ in tqdm(range(n_samples)):
        signal, text = generate_synthetic_data(
            duration=2.0,
            sampling_rate=config['data']['sampling_rate'],
            n_channels=config['data']['n_channels']
        )
        signals.append(signal)
        texts.append(text)
    
    return np.array(signals), texts

def load_trained_model(config, checkpoint_path):
    """Load trained model from checkpoint."""
    model = SilentSpeechModel(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def run_electrode_study(config, model, signals, texts):
    """Run the electrode reduction study."""
    print("Running electrode reduction study...")
    
    # Initialize analyzer
    analyzer = ElectrodeAnalyzer(n_channels=config['data']['n_channels'])
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        predictions = []
        for signal in tqdm(signals):
            signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
            pred = model.forward(signal_tensor)
            predictions.append(pred.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    
    # Run analysis for different channel counts
    target_channels = list(range(3, config['data']['n_channels'] + 1))
    report = analyzer.generate_reduction_report(signals, predictions, target_channels)
    
    # Log results to wandb
    for n_channels, results in report['subset_evaluations'].items():
        wandb.log({
            f'performance/mutual_information_{n_channels}': results['performance']['mutual_information'],
            f'performance/avg_correlation_{n_channels}': results['performance']['avg_correlation']
        })
    
    return report

def save_results(report, output_dir):
    """Save study results to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full report
    with open(output_dir / 'electrode_study_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save summary
    summary = {
        'channel_rankings': report['full_rankings']['ranked_indices'].tolist(),
        'importance_scores': report['full_rankings']['importance_scores'].tolist(),
        'optimal_subsets': {
            str(k): v['selected_channels'] 
            for k, v in report['subset_evaluations'].items()
        }
    }
    
    with open(output_dir / 'electrode_study_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Run electrode reduction study')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/electrode_study',
                      help='Directory to save results')
    parser.add_argument('--n_samples', type=int, default=1000,
                      help='Number of synthetic samples to generate')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize wandb
    run = setup_wandb(config)
    
    try:
        # Generate synthetic data
        signals, texts = generate_study_data(config, args.n_samples)
        
        # Load trained model
        model = load_trained_model(config, args.checkpoint)
        
        # Run study
        report = run_electrode_study(config, model, signals, texts)
        
        # Save results
        save_results(report, args.output_dir)
        
        print(f"Study completed successfully. Results saved to {args.output_dir}")
        
    finally:
        # Ensure wandb run is finished
        wandb.finish()

if __name__ == '__main__':
    main() 