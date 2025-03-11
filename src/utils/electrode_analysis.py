"""
Utility functions for analyzing EMG electrode importance and reduction strategies.
"""

import numpy as np
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

class ElectrodeAnalyzer:
    """Class for analyzing EMG electrode importance and reduction strategies."""
    
    def __init__(self, n_channels=8):
        """Initialize the analyzer with the number of channels."""
        self.n_channels = n_channels
        self.scaler = StandardScaler()
        
    def compute_mutual_information(self, signals, predictions):
        """
        Compute mutual information between EMG signals and predictions.
        
        Args:
            signals (np.ndarray): EMG signals of shape (n_samples, n_channels, time_steps)
            predictions (np.ndarray): Predicted text embeddings
            
        Returns:
            np.ndarray: Mutual information scores for each channel
        """
        # Reshape signals to (n_samples, n_channels * time_steps)
        n_samples, n_channels, time_steps = signals.shape
        signals_flat = signals.reshape(n_samples, -1)
        
        # Standardize the signals
        signals_scaled = self.scaler.fit_transform(signals_flat)
        
        # Compute mutual information for each channel
        mi_scores = []
        for ch in range(n_channels):
            ch_data = signals_scaled[:, ch*time_steps:(ch+1)*time_steps]
            mi = mutual_info_regression(ch_data, predictions.ravel())
            mi_scores.append(np.mean(mi))
            
        return np.array(mi_scores)
    
    def compute_channel_correlation(self, signals):
        """
        Compute correlation between EMG channels.
        
        Args:
            signals (np.ndarray): EMG signals of shape (n_samples, n_channels, time_steps)
            
        Returns:
            np.ndarray: Correlation matrix of shape (n_channels, n_channels)
        """
        n_samples, n_channels, time_steps = signals.shape
        signals_flat = signals.reshape(n_samples * time_steps, n_channels)
        return np.corrcoef(signals_flat.T)
    
    def rank_electrodes(self, signals, predictions):
        """
        Rank electrodes by importance using multiple metrics.
        
        Args:
            signals (np.ndarray): EMG signals
            predictions (np.ndarray): Predicted text embeddings
            
        Returns:
            dict: Dictionary containing electrode rankings and scores
        """
        # Compute mutual information
        mi_scores = self.compute_mutual_information(signals, predictions)
        
        # Compute channel correlations
        corr_matrix = self.compute_channel_correlation(signals)
        
        # Compute redundancy scores (average correlation with other channels)
        redundancy_scores = np.mean(np.abs(corr_matrix), axis=1)
        
        # Compute final importance scores (MI score / redundancy)
        importance_scores = mi_scores / (redundancy_scores + 1e-6)
        
        # Sort electrodes by importance
        ranked_indices = np.argsort(importance_scores)[::-1]
        
        return {
            'ranked_indices': ranked_indices,
            'importance_scores': importance_scores,
            'mi_scores': mi_scores,
            'redundancy_scores': redundancy_scores,
            'correlation_matrix': corr_matrix
        }
    
    def select_optimal_subset(self, signals, predictions, target_channels):
        """
        Select optimal subset of electrodes.
        
        Args:
            signals (np.ndarray): EMG signals
            predictions (np.ndarray): Predicted text embeddings
            target_channels (int): Target number of channels to select
            
        Returns:
            list: Indices of selected channels
        """
        rankings = self.rank_electrodes(signals, predictions)
        return rankings['ranked_indices'][:target_channels].tolist()
    
    def evaluate_subset_performance(self, signals, predictions, selected_channels):
        """
        Evaluate performance of a subset of channels.
        
        Args:
            signals (np.ndarray): EMG signals
            predictions (np.ndarray): Predicted text embeddings
            selected_channels (list): List of channel indices to evaluate
            
        Returns:
            dict: Performance metrics for the selected subset
        """
        # Extract selected channels
        subset_signals = signals[:, selected_channels, :]
        
        # Compute mutual information for subset
        subset_mi = self.compute_mutual_information(subset_signals, predictions)
        
        # Compute correlation within subset
        subset_corr = self.compute_channel_correlation(subset_signals)
        
        return {
            'mutual_information': np.mean(subset_mi),
            'avg_correlation': np.mean(np.abs(subset_corr - np.eye(len(selected_channels)))),
            'n_channels': len(selected_channels)
        }
    
    def generate_reduction_report(self, signals, predictions, target_channels_list):
        """
        Generate comprehensive report for electrode reduction analysis.
        
        Args:
            signals (np.ndarray): EMG signals
            predictions (np.ndarray): Predicted text embeddings
            target_channels_list (list): List of target channel counts to evaluate
            
        Returns:
            dict: Comprehensive analysis report
        """
        report = {
            'full_rankings': self.rank_electrodes(signals, predictions),
            'subset_evaluations': {}
        }
        
        for n_channels in target_channels_list:
            selected = self.select_optimal_subset(signals, predictions, n_channels)
            performance = self.evaluate_subset_performance(signals, predictions, selected)
            report['subset_evaluations'][n_channels] = {
                'selected_channels': selected,
                'performance': performance
            }
            
        return report 