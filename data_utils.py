import string
import numpy as np
import scipy.signal
import torch
import matplotlib.pyplot as plt

def preprocess_openbci_emg(signal, fs=1000):
    """Preprocess OpenBCI EMG signals according to silent speech specifications."""
    # Bandpass filter 20-450 Hz
    b, a = scipy.signal.butter(4, [20/(fs/2), 450/(fs/2)], btype='band')
    filtered = scipy.signal.filtfilt(b, a, signal)
    
    # Subject-wise normalization
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)
    
    return normalized

def extract_emg_features(signal, window_size=128, overlap=64):
    """Extract time and frequency domain features from EMG signal."""
    features = []
    
    # Time domain features
    rms = np.sqrt(np.mean(signal**2))
    zcr = np.sum(np.diff(np.signbit(signal).astype(int))) / len(signal)
    
    # Frequency domain features
    freqs, psd = scipy.signal.welch(signal, fs=1000, nperseg=window_size)
    
    # Combine features
    features.extend([rms, zcr])
    features.extend(psd)
    
    return np.array(features)

class FeatureNormalizer(object):
    def __init__(self, feature_samples, share_scale=False):
        """ features_samples should be list of 2d matrices with dimension (time, feature) """
        feature_samples = np.concatenate(feature_samples, axis=0)
        self.feature_means = feature_samples.mean(axis=0, keepdims=True)
        if share_scale:
            self.feature_stddevs = feature_samples.std()
        else:
            self.feature_stddevs = feature_samples.std(axis=0, keepdims=True)

    def normalize(self, sample):
        sample -= self.feature_means
        sample /= self.feature_stddevs
        return sample

    def inverse(self, sample):
        sample = sample * self.feature_stddevs
        sample = sample + self.feature_means
        return sample

class TextTransform(object):
    def __init__(self):
        self.chars = string.ascii_lowercase + string.digits + ' '
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def clean_text(self, text):
        text = text.lower()
        return ''.join(c for c in text if c in self.chars)

    def text_to_int(self, text):
        text = self.clean_text(text)
        return [self.char_to_idx[c] for c in text]

    def int_to_text(self, ints):
        return ''.join(self.idx_to_char[i] for i in ints)
