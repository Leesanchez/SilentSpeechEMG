import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import wandb
import torch
from scipy import signal

def plot_emg_channels(emg_data: np.ndarray, title: str = "EMG Signals", save_path: str = None, return_fig: bool = False):
    """Plot multiple EMG channels."""
    num_channels = emg_data.shape[0]
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 2*num_channels))
    fig.suptitle(title)
    
    for i, channel in enumerate(emg_data):
        if num_channels == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(channel)
        ax.set_ylabel(f'Channel {i+1}')
        ax.grid(True)
    
    plt.tight_layout()
    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_augmentation_effects(original: np.ndarray, augmented: np.ndarray, 
                            augmentation_name: str, save_path: str = None):
    """Plot original vs augmented signals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot original
    for i, channel in enumerate(original):
        ax1.plot(channel, label=f'Channel {i+1}')
    ax1.set_title('Original Signal')
    ax1.grid(True)
    ax1.legend()
    
    # Plot augmented
    for i, channel in enumerate(augmented):
        ax2.plot(channel, label=f'Channel {i+1}')
    ax2.set_title(f'After {augmentation_name}')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_spectrogram(emg_data: np.ndarray, fs: int = 1000, 
                    title: str = "EMG Spectrogram", save_path: str = None,
                    return_fig: bool = False):
    """Plot spectrogram of EMG signals."""
    num_channels = emg_data.shape[0]
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 3*num_channels))
    fig.suptitle(title)
    
    for i, channel in enumerate(emg_data):
        if num_channels == 1:
            ax = axes
        else:
            ax = axes[i]
            
        f, t, Sxx = signal.spectrogram(channel, fs=fs, nperseg=256, noverlap=128)
        pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'Channel {i+1}')
        fig.colorbar(pcm, ax=ax)
    
    plt.tight_layout()
    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def log_augmentation_examples(original: np.ndarray, augmented_dict: Dict[str, np.ndarray]):
    """Log augmentation examples to WandB."""
    # Create individual plots for each augmentation
    for aug_name, aug_data in augmented_dict.items():
        fig_path = f"augmentation_{aug_name}.png"
        plot_augmentation_effects(original, aug_data, aug_name, save_path=fig_path)
        wandb.log({f"augmentation/{aug_name}": wandb.Image(fig_path)})
        
        # Log spectrograms
        spec_path = f"spectrogram_{aug_name}.png"
        plot_spectrogram(aug_data, title=f"Spectrogram after {aug_name}", save_path=spec_path)
        wandb.log({f"spectrogram/{aug_name}": wandb.Image(spec_path)})

def visualize_predictions(emg_data: np.ndarray, true_text: str, pred_text: str, 
                        confidence: float, save_path: str = None, return_fig: bool = False):
    """Visualize EMG signals with corresponding predictions."""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot EMG signals
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    for i, channel in enumerate(emg_data):
        ax1.plot(channel, label=f'Channel {i+1}')
    ax1.set_title('EMG Signals')
    ax1.grid(True)
    ax1.legend()
    
    # Plot text predictions
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.axis('off')
    ax2.text(0.1, 0.6, f'True: {true_text}', fontsize=12)
    ax2.text(0.1, 0.2, f'Pred: {pred_text} (conf: {confidence:.2f})', fontsize=12)
    
    plt.tight_layout()
    if return_fig:
        return fig
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_metrics(metrics_dict: Dict[str, List[float]], save_path: str = None):
    """Plot training metrics over time."""
    num_metrics = len(metrics_dict)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(15, 4*num_metrics))
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if num_metrics == 1:
            ax = axes
        else:
            ax = axes[i]
        
        ax.plot(values)
        ax.set_title(metric_name)
        ax.grid(True)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(true_chars: List[str], pred_chars: List[str], 
                        char_to_idx: Dict[str, int], save_path: str = None):
    """Plot character-level confusion matrix."""
    num_classes = len(char_to_idx)
    conf_matrix = np.zeros((num_classes, num_classes))
    
    for t, p in zip(true_chars, pred_chars):
        if t in char_to_idx and p in char_to_idx:
            conf_matrix[char_to_idx[t], char_to_idx[p]] += 1
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='.0f', 
                xticklabels=list(char_to_idx.keys()),
                yticklabels=list(char_to_idx.keys()))
    plt.title('Character Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 