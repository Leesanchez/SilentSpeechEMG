"""
Tests for EMG signal preprocessing module.
"""

import numpy as np
import pytest
from src.data.preprocessing import PreprocessingConfig, EMGPreprocessor, EMGAugmenter

@pytest.fixture
def config():
    """Create a preprocessing configuration for testing."""
    return PreprocessingConfig(
        sampling_rate=1000,
        notch_freq=50.0,
        bandpass_low=20.0,
        bandpass_high=450.0,
        window_size=256,
        hop_length=128,
        n_mels=80,
        normalize=True
    )

@pytest.fixture
def preprocessor(config):
    """Create an EMG preprocessor instance for testing."""
    return EMGPreprocessor(config)

@pytest.fixture
def augmenter(config):
    """Create an EMG augmenter instance for testing."""
    return EMGAugmenter(config)

@pytest.fixture
def sample_signal():
    """Create a sample EMG signal for testing."""
    # Create a 1-second signal with 8 channels
    t = np.linspace(0, 1, 1000)
    signal = np.zeros((1000, 8))
    
    # Add some synthetic EMG-like components
    for i in range(8):
        # Main EMG frequency components
        signal[:, i] = (
            np.sin(2 * np.pi * 50 * t) +  # Power line noise
            np.sin(2 * np.pi * 100 * t) +  # EMG component
            np.random.normal(0, 0.1, len(t))  # Random noise
        )
    
    return signal

def test_preprocessing_config():
    """Test preprocessing configuration initialization."""
    config = PreprocessingConfig()
    assert config.sampling_rate == 1000
    assert config.notch_freq == 50.0
    assert config.bandpass_low == 20.0
    assert config.bandpass_high == 450.0

def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization."""
    assert preprocessor.config.sampling_rate == 1000
    assert hasattr(preprocessor, 'notch_b')
    assert hasattr(preprocessor, 'notch_a')
    assert hasattr(preprocessor, 'bandpass_b')
    assert hasattr(preprocessor, 'bandpass_a')

def test_remove_baseline_drift(preprocessor, sample_signal):
    """Test baseline drift removal."""
    processed = preprocessor.remove_baseline_drift(sample_signal)
    assert processed.shape == sample_signal.shape
    assert np.mean(processed) < np.mean(sample_signal)  # Should reduce DC component

def test_apply_filters(preprocessor, sample_signal):
    """Test filter application."""
    filtered = preprocessor.apply_filters(sample_signal)
    assert filtered.shape == sample_signal.shape
    
    # Check if power line noise is reduced
    freqs, psd_original = np.fft.fftfreq(len(sample_signal)), np.abs(np.fft.fft(sample_signal[:, 0]))
    _, psd_filtered = np.fft.fftfreq(len(filtered)), np.abs(np.fft.fft(filtered[:, 0]))
    
    # Find power at 50 Hz
    idx_50hz = np.argmin(np.abs(freqs - 50))
    assert psd_filtered[idx_50hz] < psd_original[idx_50hz]

def test_compute_envelope(preprocessor, sample_signal):
    """Test envelope computation."""
    envelope = preprocessor.compute_envelope(sample_signal)
    assert envelope.shape == sample_signal.shape
    assert np.all(envelope >= 0)  # Envelope should be non-negative

def test_extract_features(preprocessor, sample_signal):
    """Test feature extraction."""
    features = preprocessor.extract_features(sample_signal)
    
    # Check if all expected features are present
    expected_features = ['rms', 'mav', 'var', 'ssc', 'mean_freq', 'median_freq', 'peak_freq']
    assert all(key in features for key in expected_features)
    
    # Check feature dimensions
    assert features['rms'].shape == (8,)  # Should have one value per channel
    assert features['mav'].shape == (8,)
    assert features['var'].shape == (8,)

def test_segment_signal(preprocessor, sample_signal):
    """Test signal segmentation."""
    segments = preprocessor.segment_signal(sample_signal)
    
    # Check segment dimensions
    n_segments = (len(sample_signal) - preprocessor.config.window_size) // preprocessor.config.hop_length + 1
    assert segments.shape == (n_segments, preprocessor.config.window_size, sample_signal.shape[1])

def test_normalize_signal(preprocessor, sample_signal):
    """Test signal normalization."""
    normalized = preprocessor.normalize_signal(sample_signal)
    assert normalized.shape == sample_signal.shape
    assert np.abs(np.mean(normalized)) < 1e-10  # Should be approximately zero-mean
    assert np.abs(np.std(normalized) - 1.0) < 1e-10  # Should have unit variance

def test_process_trial(preprocessor, sample_signal):
    """Test complete trial processing."""
    segments, features = preprocessor.process_trial(sample_signal)
    
    # Check outputs
    assert isinstance(segments, np.ndarray)
    assert isinstance(features, dict)
    assert len(features) > 0
    
    # Check if segments are normalized
    assert np.abs(np.mean(segments)) < 1e-10

def test_augmenter_gaussian_noise(augmenter, sample_signal):
    """Test Gaussian noise augmentation."""
    augmented = augmenter.add_gaussian_noise(sample_signal)
    assert augmented.shape == sample_signal.shape
    assert not np.array_equal(augmented, sample_signal)

def test_augmenter_time_stretch(augmenter, sample_signal):
    """Test time stretching augmentation."""
    augmented = augmenter.time_stretch(sample_signal)
    assert augmented.shape[1] == sample_signal.shape[1]  # Same number of channels
    assert augmented.shape[0] > sample_signal.shape[0]  # Longer in time

def test_augmenter_channel_dropout(augmenter, sample_signal):
    """Test channel dropout augmentation."""
    augmented = augmenter.channel_dropout(sample_signal)
    assert augmented.shape == sample_signal.shape
    
    # Check if some channels are zeroed out
    channel_sums = np.sum(np.abs(augmented), axis=0)
    assert np.any(channel_sums == 0)

def test_augmenter_random_crop(augmenter, sample_signal):
    """Test random crop augmentation."""
    crop_size = 500
    cropped = augmenter.random_crop(sample_signal, crop_size)
    assert cropped.shape == (crop_size, sample_signal.shape[1])

def test_augmenter_full_pipeline(augmenter, sample_signal):
    """Test complete augmentation pipeline."""
    augmented = augmenter.augment(sample_signal)
    assert augmented.shape[1] == sample_signal.shape[1]  # Same number of channels
    assert not np.array_equal(augmented, sample_signal)  # Should be different from original 