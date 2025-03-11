import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import os
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from typing import List, Tuple

from spatiotemporal_model import SilentSpeechTransformer, PhonemeLanguageModel
from data_utils import preprocess_openbci_emg, extract_emg_features, TextTransform
from model_utils import (
    create_ensemble,
    compute_wer_cer,
    decode_predictions,
    decode_targets,
    train_model,
    LabelSmoothing,
    EarlyStopping
)
from visualization_utils import (
    plot_confusion_matrix, 
    plot_training_metrics,
    visualize_predictions,
    log_augmentation_examples
)

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Separate EMG and labels
    emg, labels = zip(*batch)
    
    # Stack EMG data (already padded)
    emg = torch.stack(emg)
    
    # Find max label length
    max_len = max(len(l) for l in labels)
    
    # Pad labels with -1 (will be ignored in loss)
    padded_labels = []
    for label in labels:
        padding = torch.full((max_len - len(label),), -1, dtype=torch.long)
        padded_labels.append(torch.cat([label, padding]))
    
    labels = torch.stack(padded_labels)
    
    return emg, labels

class EMGDataset(Dataset):
    def __init__(self, emg_data, labels, transform=None):
        self.emg_data = emg_data
        self.labels = labels
        self.transform = transform
        self.text_transform = TextTransform()
        
    def __len__(self):
        return len(self.emg_data)
        
    def __getitem__(self, idx):
        emg = self.emg_data[idx]  # Shape: (channels, time)
        label = self.labels[idx]   # Variable-length sequence of integers
        
        # Extract features from each channel
        features = []
        for channel_data in emg:
            channel_features = extract_emg_features(channel_data)
            features.append(channel_features)
        features = np.array(features)  # Shape: (channels, features)
        
        if self.transform:
            features = self.transform(features)
            
        return torch.FloatTensor(features), torch.LongTensor(label)

class TimeWarp:
    def __init__(self, sigma=0.2):
        self.sigma = sigma
        
    def __call__(self, signal):
        # Time warping using linear interpolation
        length = signal.shape[1]
        warp = np.random.normal(1.0, self.sigma, size=length)
        warp = np.cumsum(warp)
        warp = length * warp / warp[-1]
        warped = np.zeros_like(signal)
        
        for i in range(signal.shape[0]):
            warped[i] = np.interp(np.arange(length), warp, signal[i])
            
        return warped

class SignalJitter:
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        
    def __call__(self, signal):
        return signal + np.random.normal(0, self.sigma, signal.shape)

class RandomScaling:
    def __init__(self, scale_range=(0.8, 1.2)):
        self.scale_range = scale_range
        
    def __call__(self, signal):
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return signal * scale

class ChannelDropout:
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, signal):
        mask = np.random.binomial(1, 1-self.p, size=signal.shape[0])
        return signal * mask[:, np.newaxis]

class GaussianNoise:
    def __init__(self, std_range=(0.01, 0.05)):
        self.std_range = std_range
        
    def __call__(self, signal):
        std = np.random.uniform(self.std_range[0], self.std_range[1])
        noise = np.random.normal(0, std, signal.shape)
        return signal + noise

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal

class BaselineWander:
    """Simulates baseline drift in EMG signals."""
    def __init__(self, freq_range=(0.1, 1.0), amplitude_range=(0.1, 0.3)):
        self.freq_range = freq_range
        self.amplitude_range = amplitude_range
        
    def __call__(self, signal):
        time_points = signal.shape[1]
        t = np.linspace(0, time_points/100, time_points)  # Assume 100Hz sampling rate
        
        # Generate random frequency and amplitude
        freq = np.random.uniform(*self.freq_range)
        amplitude = np.random.uniform(*self.amplitude_range)
        
        # Create baseline wander
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add to all channels
        return signal + wander

class PowerlineNoise:
    """Simulates power line interference (50/60 Hz)."""
    def __init__(self, frequencies=[50, 60], amplitude_range=(0.01, 0.05)):
        self.frequencies = frequencies
        self.amplitude_range = amplitude_range
        
    def __call__(self, signal):
        time_points = signal.shape[1]
        t = np.linspace(0, time_points/100, time_points)  # Assume 100Hz sampling rate
        
        noise = np.zeros_like(t)
        for freq in self.frequencies:
            amplitude = np.random.uniform(*self.amplitude_range)
            noise += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add to all channels
        return signal + noise[:, np.newaxis].T

class MuscleFatigue:
    """Simulates muscle fatigue by modifying signal characteristics."""
    def __init__(self, freq_decay_range=(0.8, 0.95), amplitude_increase_range=(1.05, 1.2)):
        self.freq_decay_range = freq_decay_range
        self.amplitude_increase_range = amplitude_increase_range
        
    def __call__(self, signal):
        # Simulate frequency content decay
        freq_decay = np.random.uniform(*self.freq_decay_range)
        amplitude_increase = np.random.uniform(*self.amplitude_increase_range)
        
        # Apply frequency decay using FFT
        transformed = []
        for channel in signal:
            ft = np.fft.fft(channel)
            freqs = np.fft.fftfreq(len(channel))
            ft_modified = ft * np.exp(-np.abs(freqs) / freq_decay)
            channel_modified = np.real(np.fft.ifft(ft_modified))
            transformed.append(channel_modified * amplitude_increase)
        
        return np.array(transformed)

class ElectrodeShift:
    """Simulates small electrode position shifts."""
    def __init__(self, shift_range=(-0.1, 0.1), rotation_range=(-5, 5)):
        self.shift_range = shift_range
        self.rotation_range = rotation_range
        
    def __call__(self, signal):
        num_channels = signal.shape[0]
        
        # Random channel mixing matrix
        theta = np.random.uniform(*self.rotation_range) * np.pi / 180
        rotation_matrix = np.eye(num_channels) * np.cos(theta) + \
                         np.roll(np.eye(num_channels), 1, axis=1) * np.sin(theta)
        
        # Apply mixing
        shifted_signal = np.dot(rotation_matrix, signal)
        
        # Add random shifts
        shifts = np.random.uniform(*self.shift_range, size=num_channels)
        shifted_signal += shifts[:, np.newaxis]
        
        return shifted_signal

class BandpassFilter:
    """Applies random bandpass filtering to simulate different frequency responses."""
    def __init__(self, low_freq_range=(20, 30), high_freq_range=(450, 500)):
        self.low_freq_range = low_freq_range
        self.high_freq_range = high_freq_range
        
    def __call__(self, signal):
        from scipy.signal import butter, filtfilt
        
        # Random cutoff frequencies
        low_freq = np.random.uniform(*self.low_freq_range)
        high_freq = np.random.uniform(*self.high_freq_range)
        
        # Design filter
        nyquist = 500  # Assume 1000Hz sampling rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered = np.array([filtfilt(b, a, channel) for channel in signal])
        
        return filtered

def main():
    # Load data
    emg_data = np.load('collected_data/emg_samples.npy')
    labels = np.load('collected_data/labels.npy', allow_pickle=True)
    
    # Create dataset with augmentations
    transform = Compose([
        # Basic augmentations
        TimeWarp(sigma=0.2),
        SignalJitter(sigma=0.05),
        RandomScaling(scale_range=(0.8, 1.2)),
        
        # EMG-specific augmentations
        BaselineWander(freq_range=(0.1, 1.0), amplitude_range=(0.1, 0.3)),
        PowerlineNoise(frequencies=[50, 60], amplitude_range=(0.01, 0.05)),
        MuscleFatigue(freq_decay_range=(0.8, 0.95), amplitude_increase_range=(1.05, 1.2)),
        ElectrodeShift(shift_range=(-0.1, 0.1), rotation_range=(-5, 5)),
        BandpassFilter(low_freq_range=(20, 30), high_freq_range=(450, 500)),
        
        # Final noise and dropout
        ChannelDropout(p=0.1),
        GaussianNoise(std_range=(0.01, 0.05))
    ])
    
    # Create full dataset
    dataset = EMGDataset(emg_data, labels, transform=transform)
    
    # Get vocabulary size
    text_transform = TextTransform()
    vocab_size = len(text_transform.chars)
    
    # Create model ensemble with cross-validation
    ensemble = create_ensemble(num_models=3, vocab_size=vocab_size)
    
    # Train with cross-validation
    ensemble.train_with_cross_validation(
        dataset=dataset,
        num_folds=5,
        batch_size=64,
        num_epochs=100,
        collate_fn=collate_fn
    )

if __name__ == '__main__':
    main() 