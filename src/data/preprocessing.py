"""
Comprehensive data preprocessing pipeline for EMG signals.
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
from typing import List, Tuple, Optional, Dict
import torch
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class PreprocessingConfig:
    """Configuration for EMG signal preprocessing."""
    sampling_rate: int = 1000
    notch_freq: float = 50.0  # Hz (for power line interference)
    bandpass_low: float = 20.0  # Hz
    bandpass_high: float = 450.0  # Hz
    window_size: int = 256
    hop_length: int = 128
    n_mels: int = 80
    normalize: bool = True

class EMGPreprocessor:
    """Class for preprocessing EMG signals."""
    
    def __init__(self, config: PreprocessingConfig):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.scaler = StandardScaler()
        self._setup_filters()
        
    def _setup_filters(self):
        """Setup digital filters for signal processing."""
        nyq = self.config.sampling_rate / 2
        
        # Notch filter for power line interference
        notch_b, notch_a = signal.iirnotch(
            self.config.notch_freq,
            Q=30.0,
            fs=self.config.sampling_rate
        )
        
        # Bandpass filter
        bandpass_b, bandpass_a = signal.butter(
            4,
            [self.config.bandpass_low / nyq, self.config.bandpass_high / nyq],
            btype='band'
        )
        
        self.notch_b = notch_b
        self.notch_a = notch_a
        self.bandpass_b = bandpass_b
        self.bandpass_a = bandpass_a
        
    def remove_baseline_drift(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove low-frequency baseline drift using high-pass filter."""
        return signal.detrend(signal_data, axis=0)
    
    def apply_filters(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply notch and bandpass filters to remove noise."""
        # Apply notch filter
        signal_notched = signal.filtfilt(self.notch_b, self.notch_a, signal_data, axis=0)
        
        # Apply bandpass filter
        return signal.filtfilt(self.bandpass_b, self.bandpass_a, signal_notched, axis=0)
    
    def compute_envelope(self, signal_data: np.ndarray) -> np.ndarray:
        """Compute signal envelope using Hilbert transform."""
        analytic_signal = signal.hilbert(signal_data, axis=0)
        return np.abs(analytic_signal)
    
    def extract_features(self, signal_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time and frequency domain features."""
        features = {}
        
        # Time domain features
        features['rms'] = np.sqrt(np.mean(signal_data**2, axis=0))
        features['mav'] = np.mean(np.abs(signal_data), axis=0)
        features['var'] = np.var(signal_data, axis=0)
        features['ssc'] = np.sum(np.diff(np.sign(np.diff(signal_data, axis=0))), axis=0)
        
        # Frequency domain features
        freqs, psd = signal.welch(signal_data, 
                                fs=self.config.sampling_rate,
                                nperseg=self.config.window_size,
                                noverlap=self.config.window_size//2)
        
        features['mean_freq'] = np.sum(freqs[:, None] * psd, axis=0) / np.sum(psd, axis=0)
        features['median_freq'] = np.median(freqs)
        features['peak_freq'] = freqs[np.argmax(psd, axis=0)]
        
        return features
    
    def segment_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Segment signal into overlapping windows."""
        n_samples = signal_data.shape[0]
        n_channels = signal_data.shape[1]
        
        # Calculate number of segments
        n_segments = (n_samples - self.config.window_size) // self.config.hop_length + 1
        
        segments = np.zeros((n_segments, self.config.window_size, n_channels))
        for i in range(n_segments):
            start = i * self.config.hop_length
            end = start + self.config.window_size
            segments[i] = signal_data[start:end]
            
        return segments
    
    def normalize_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Normalize signal using z-score normalization."""
        if not self.config.normalize:
            return signal_data
            
        if len(signal_data.shape) == 2:  # Single trial
            return self.scaler.fit_transform(signal_data)
        else:  # Multiple segments
            original_shape = signal_data.shape
            reshaped = signal_data.reshape(-1, original_shape[-1])
            normalized = self.scaler.fit_transform(reshaped)
            return normalized.reshape(original_shape)
    
    def process_trial(self, signal_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Process a single trial of EMG data."""
        # Remove baseline drift
        signal_clean = self.remove_baseline_drift(signal_data)
        
        # Apply filters
        signal_filtered = self.apply_filters(signal_clean)
        
        # Compute envelope
        envelope = self.compute_envelope(signal_filtered)
        
        # Extract features
        features = self.extract_features(signal_filtered)
        
        # Segment signal
        segments = self.segment_signal(signal_filtered)
        
        # Normalize segments
        segments_normalized = self.normalize_signal(segments)
        
        return segments_normalized, features
    
    def process_batch(self, batch_data: List[np.ndarray]) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Process a batch of EMG trials."""
        return [self.process_trial(trial) for trial in batch_data]
    
    def prepare_for_training(self, signal_data: np.ndarray) -> torch.Tensor:
        """Prepare processed signal for model training."""
        # Process the signal
        segments, _ = self.process_trial(signal_data)
        
        # Convert to tensor
        return torch.FloatTensor(segments)

class EMGAugmenter:
    """Class for EMG signal augmentation."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def add_gaussian_noise(self, signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
        """Add Gaussian noise with specified SNR."""
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise
    
    def time_stretch(self, signal: np.ndarray, rate: float = 1.2) -> np.ndarray:
        """Time stretch the signal."""
        return signal.repeat(int(rate), axis=0)
    
    def random_crop(self, signal: np.ndarray, crop_size: int) -> np.ndarray:
        """Randomly crop the signal."""
        if len(signal) <= crop_size:
            return signal
        start = np.random.randint(0, len(signal) - crop_size)
        return signal[start:start + crop_size]
    
    def channel_dropout(self, signal: np.ndarray, p: float = 0.1) -> np.ndarray:
        """Randomly drop out channels."""
        mask = np.random.binomial(1, 1-p, size=signal.shape[1])
        return signal * mask[None, :]
    
    def augment(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        # Randomly choose augmentations
        if np.random.random() < 0.5:
            signal = self.add_gaussian_noise(signal)
        if np.random.random() < 0.3:
            signal = self.time_stretch(signal)
        if np.random.random() < 0.3:
            signal = self.channel_dropout(signal)
        return signal 