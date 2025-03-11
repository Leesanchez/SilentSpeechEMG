"""
Visualization module for EMG electrode analysis and performance visualization.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class ElectrodeVisualizer:
    """Class for visualizing EMG electrode data and analysis results."""
    
    def __init__(self, style: str = 'seaborn'):
        """Initialize visualizer with specified style."""
        plt.style.use(style)
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def plot_raw_signals(self, 
                        signals: np.ndarray,
                        sampling_rate: int,
                        channels: Optional[List[str]] = None,
                        title: str = 'Raw EMG Signals',
                        figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot raw EMG signals from multiple channels."""
        n_channels = signals.shape[1]
        time = np.arange(len(signals)) / sampling_rate
        
        if channels is None:
            channels = [f'Channel {i+1}' for i in range(n_channels)]
        
        fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
        fig.suptitle(title)
        
        for i, (ax, channel) in enumerate(zip(axes, channels)):
            ax.plot(time, signals[:, i], color=self.colors[i % 10])
            ax.set_ylabel(channel)
            ax.grid(True)
        
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
    def plot_power_spectrum(self,
                          signals: np.ndarray,
                          sampling_rate: int,
                          channels: Optional[List[str]] = None,
                          title: str = 'Power Spectrum',
                          figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot power spectrum for each channel."""
        n_channels = signals.shape[1]
        
        if channels is None:
            channels = [f'Channel {i+1}' for i in range(n_channels)]
            
        fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
        fig.suptitle(title)
        
        for i, (ax, channel) in enumerate(zip(axes, channels)):
            freqs, psd = signal.welch(signals[:, i], 
                                    fs=sampling_rate,
                                    nperseg=min(2048, len(signals)))
            ax.semilogy(freqs, psd, color=self.colors[i % 10])
            ax.set_ylabel(f'{channel}\nPower/Freq')
            ax.grid(True)
        
        axes[-1].set_xlabel('Frequency (Hz)')
        plt.tight_layout()
        
    def plot_feature_importance(self,
                              importance_scores: Dict[str, float],
                              title: str = 'Electrode Feature Importance',
                              figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot feature importance scores for electrodes."""
        plt.figure(figsize=figsize)
        
        # Sort importance scores
        sorted_items = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_items)
        
        # Create bar plot
        plt.bar(features, scores, color=self.colors)
        plt.title(title)
        plt.xlabel('Electrodes')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        
    def plot_electrode_correlation(self,
                                signals: np.ndarray,
                                channels: Optional[List[str]] = None,
                                title: str = 'Electrode Correlation Matrix',
                                figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot correlation matrix between electrodes."""
        n_channels = signals.shape[1]
        
        if channels is None:
            channels = [f'Channel {i+1}' for i in range(n_channels)]
            
        # Compute correlation matrix
        corr_matrix = np.corrcoef(signals.T)
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix,
                    xticklabels=channels,
                    yticklabels=channels,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    vmin=-1,
                    vmax=1)
        plt.title(title)
        plt.tight_layout()
        
    def plot_electrode_reduction_impact(self,
                                      accuracies: Dict[int, float],
                                      title: str = 'Impact of Electrode Reduction',
                                      figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot impact of reducing electrode count on accuracy."""
        electrodes = list(accuracies.keys())
        scores = list(accuracies.values())
        
        plt.figure(figsize=figsize)
        plt.plot(electrodes, scores, 'o-', color=self.colors[0])
        plt.fill_between(electrodes,
                        [s - 0.05 for s in scores],
                        [s + 0.05 for s in scores],
                        alpha=0.2,
                        color=self.colors[0])
        
        plt.title(title)
        plt.xlabel('Number of Electrodes')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.tight_layout()
        
    def plot_signal_quality(self,
                          quality_metrics: Dict[str, List[float]],
                          channels: Optional[List[str]] = None,
                          title: str = 'Signal Quality Metrics',
                          figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot signal quality metrics for each electrode."""
        metrics_df = pd.DataFrame(quality_metrics)
        
        if channels is not None:
            metrics_df.index = channels
            
        plt.figure(figsize=figsize)
        metrics_df.plot(kind='bar', width=0.8)
        plt.title(title)
        plt.xlabel('Electrodes')
        plt.ylabel('Quality Score')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y')
        plt.tight_layout()
        
    def plot_dimensionality_reduction(self,
                                    signals: np.ndarray,
                                    method: str = 'pca',
                                    labels: Optional[np.ndarray] = None,
                                    title: str = 'Dimensionality Reduction',
                                    figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot dimensionality reduction of electrode signals."""
        # Reshape signals if needed
        if len(signals.shape) > 2:
            signals_2d = signals.reshape(signals.shape[0], -1)
        else:
            signals_2d = signals
            
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42)
            
        reduced = reducer.fit_transform(signals_2d)
        
        plt.figure(figsize=figsize)
        if labels is not None:
            scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10')
            plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], color=self.colors[0])
            
        plt.title(f'{title} ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.tight_layout()
        
    def plot_confusion_matrix(self,
                            conf_matrix: np.ndarray,
                            classes: Optional[List[str]] = None,
                            title: str = 'Confusion Matrix',
                            figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot confusion matrix for classification results."""
        plt.figure(figsize=figsize)
        
        if classes is None:
            classes = [f'Class {i+1}' for i in range(len(conf_matrix))]
            
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=classes,
                    yticklabels=classes)
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
    def save_figure(self, filename: str, dpi: int = 300) -> None:
        """Save the current figure to a file."""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close()

def create_report(visualizer: ElectrodeVisualizer,
                 signals: np.ndarray,
                 sampling_rate: int,
                 results: Dict,
                 output_dir: str) -> None:
    """Create a comprehensive visualization report."""
    # Plot raw signals
    visualizer.plot_raw_signals(signals, sampling_rate)
    visualizer.save_figure(f'{output_dir}/raw_signals.png')
    
    # Plot power spectrum
    visualizer.plot_power_spectrum(signals, sampling_rate)
    visualizer.save_figure(f'{output_dir}/power_spectrum.png')
    
    # Plot electrode correlation
    visualizer.plot_electrode_correlation(signals)
    visualizer.save_figure(f'{output_dir}/electrode_correlation.png')
    
    # Plot feature importance if available
    if 'importance_scores' in results:
        visualizer.plot_feature_importance(results['importance_scores'])
        visualizer.save_figure(f'{output_dir}/feature_importance.png')
    
    # Plot electrode reduction impact if available
    if 'accuracy_vs_electrodes' in results:
        visualizer.plot_electrode_reduction_impact(results['accuracy_vs_electrodes'])
        visualizer.save_figure(f'{output_dir}/electrode_reduction_impact.png')
    
    # Plot confusion matrix if available
    if 'confusion_matrix' in results:
        visualizer.plot_confusion_matrix(results['confusion_matrix'])
        visualizer.save_figure(f'{output_dir}/confusion_matrix.png')

class ElectrodeViz:
    """Class for visualizing electrode reduction study results."""
    
    def __init__(self, results_dir):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)
        self.load_results()
        
    def load_results(self):
        """Load study results from files."""
        # Load full report
        with open(self.results_dir / 'electrode_study_report.json', 'r') as f:
            self.report = json.load(f)
            
        # Load summary
        with open(self.results_dir / 'electrode_study_summary.json', 'r') as f:
            self.summary = json.load(f)
    
    def plot_importance_scores(self, save_path=None):
        """Plot electrode importance scores."""
        plt.figure(figsize=(10, 6))
        scores = np.array(self.summary['importance_scores'])
        rankings = np.array(self.summary['channel_rankings'])
        
        plt.bar(range(len(scores)), scores[rankings])
        plt.xlabel('Electrode Rank')
        plt.ylabel('Importance Score')
        plt.title('Electrode Importance Scores (Ranked)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_correlation_matrix(self, save_path=None):
        """Plot electrode correlation matrix."""
        corr_matrix = np.array(self.report['full_rankings']['correlation_matrix'])
        
        plt.figure(figsize=(8, 8))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   vmin=-1, 
                   vmax=1)
        plt.title('Electrode Correlation Matrix')
        plt.xlabel('Electrode Index')
        plt.ylabel('Electrode Index')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_performance_metrics(self, save_path=None):
        """Plot performance metrics across different channel counts."""
        n_channels = []
        mi_scores = []
        corr_scores = []
        
        for n, results in self.report['subset_evaluations'].items():
            n_channels.append(int(n))
            mi_scores.append(results['performance']['mutual_information'])
            corr_scores.append(results['performance']['avg_correlation'])
            
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Mutual Information vs Channel Count',
                                         'Average Correlation vs Channel Count'))
        
        # Mutual Information plot
        fig.add_trace(
            go.Scatter(x=n_channels, y=mi_scores,
                      mode='lines+markers',
                      name='Mutual Information'),
            row=1, col=1
        )
        
        # Correlation plot
        fig.add_trace(
            go.Scatter(x=n_channels, y=corr_scores,
                      mode='lines+markers',
                      name='Avg Correlation'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text='Performance Metrics Across Channel Counts',
            showlegend=True,
            height=500,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def plot_optimal_subsets(self, save_path=None):
        """Plot optimal electrode subsets for different channel counts."""
        fig = go.Figure()
        
        for n_channels, subset in self.summary['optimal_subsets'].items():
            mask = np.zeros(8)  # Assuming 8 total channels
            mask[subset] = 1
            
            fig.add_trace(go.Heatmap(
                z=[mask],
                text=[[str(i) if m == 1 else '' for i, m in enumerate(mask)]],
                texttemplate="%{text}",
                textfont={"size": 20},
                name=f'{n_channels} channels',
                showscale=False
            ))
        
        fig.update_layout(
            title='Optimal Electrode Subsets',
            yaxis=dict(
                ticktext=[f'{n} channels' for n in self.summary['optimal_subsets'].keys()],
                tickvals=list(range(len(self.summary['optimal_subsets']))),
                title='Configuration'
            ),
            xaxis=dict(
                title='Electrode Index',
                tickvals=list(range(8))
            ),
            height=400,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def generate_report(self, output_dir):
        """Generate comprehensive visualization report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        self.plot_importance_scores(output_dir / 'importance_scores.png')
        self.plot_correlation_matrix(output_dir / 'correlation_matrix.png')
        self.plot_performance_metrics(output_dir / 'performance_metrics.html')
        self.plot_optimal_subsets(output_dir / 'optimal_subsets.html')
        
        # Create summary HTML
        html_content = f"""
        <html>
        <head>
            <title>Electrode Reduction Study Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .plot {{ margin: 20px 0; }}
                .plot img {{ max-width: 100%; }}
                .interactive {{ width: 100%; height: 600px; border: none; }}
            </style>
        </head>
        <body>
            <h1>Electrode Reduction Study Results</h1>
            
            <h2>Importance Scores</h2>
            <div class="plot">
                <img src="importance_scores.png" alt="Importance Scores">
            </div>
            
            <h2>Correlation Matrix</h2>
            <div class="plot">
                <img src="correlation_matrix.png" alt="Correlation Matrix">
            </div>
            
            <h2>Performance Metrics</h2>
            <div class="plot">
                <iframe src="performance_metrics.html" class="interactive"></iframe>
            </div>
            
            <h2>Optimal Subsets</h2>
            <div class="plot">
                <iframe src="optimal_subsets.html" class="interactive"></iframe>
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / 'report.html', 'w') as f:
            f.write(html_content)
        
        print(f"Report generated at {output_dir}/report.html") 