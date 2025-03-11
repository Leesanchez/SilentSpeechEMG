import numpy as np
import os
from data_utils import TextTransform
from scipy import signal

def generate_synthetic_emg(duration_ms: int, num_channels: int = 8, sampling_rate: int = 1000,
                         text: str = "", noise_std: float = 0.1, base_frequency: float = 50):
    """
    Generate synthetic EMG data with text-based modulation.
    
    Args:
        duration_ms: Duration of the signal in milliseconds
        num_channels: Number of EMG channels to generate
        sampling_rate: Sampling rate in Hz
        text: Text to encode in the signal
        noise_std: Standard deviation of the noise
        base_frequency: Base frequency of the EMG signal in Hz
    
    Returns:
        numpy array of shape (num_channels, num_samples)
    """
    num_samples = int(duration_ms * sampling_rate / 1000)
    t = np.linspace(0, duration_ms/1000, num_samples)
    
    # Generate base signals for each channel
    emg_data = np.zeros((num_channels, num_samples))
    
    for i in range(num_channels):
        # Generate carrier wave
        carrier = np.sin(2 * np.pi * base_frequency * t)
        
        # Add character-specific modulation
        if text:
            char_duration = num_samples // len(text)
            for j, char in enumerate(text):
                start_idx = j * char_duration
                end_idx = (j + 1) * char_duration if j < len(text) - 1 else num_samples
                
                # Use character ASCII value to modulate frequency
                char_freq = ord(char) / 2
                modulation = np.sin(2 * np.pi * char_freq * t[start_idx:end_idx])
                carrier[start_idx:end_idx] *= (1 + modulation)
        
        # Add random amplitude modulation
        amplitude = np.random.uniform(0.8, 1.2)
        carrier *= amplitude
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, num_samples)
        
        # Combine signals
        emg_data[i] = carrier + noise
        
        # Apply bandpass filter to simulate EMG characteristics
        b, a = signal.butter(4, [20/(sampling_rate/2), 450/(sampling_rate/2)], btype='band')
        emg_data[i] = signal.filtfilt(b, a, emg_data[i])
    
    return emg_data

def pad_sequence(sequence, target_length):
    """Pad sequence to target length."""
    current_length = sequence.shape[1]
    if current_length >= target_length:
        return sequence[:, :target_length]
    else:
        padding = np.zeros((sequence.shape[0], target_length - current_length))
        return np.concatenate([sequence, padding], axis=1)

def generate_dataset(num_samples=100, vocab=None, max_duration_ms=1000):
    """Generate a synthetic dataset of EMG samples and labels.
    
    Args:
        num_samples: Number of samples to generate
        vocab: List of words to use as labels
        max_duration_ms: Maximum duration of EMG signals in milliseconds
    """
    if vocab is None:
        vocab = ['hello', 'world', 'yes', 'no', 'stop', 'go', 
                'left', 'right', 'up', 'down']
    
    # Create output directory
    os.makedirs('collected_data', exist_ok=True)
    
    # Generate samples
    emg_samples = []
    labels = []
    text_transform = TextTransform()
    
    # Calculate maximum sequence length
    max_samples = int(max_duration_ms * 1000 / 1000)  # at 1000Hz sampling rate
    
    for i in range(num_samples):
        # Generate random label
        label = np.random.choice(vocab)
        label_encoded = text_transform.text_to_int(label)
        
        # Generate EMG data with duration proportional to label length
        duration = min(len(label) * 200, max_duration_ms)  # 200ms per character
        emg = generate_synthetic_emg(duration_ms=duration)
        
        # Pad sequence to max length
        emg_padded = pad_sequence(emg, max_samples)
        
        emg_samples.append(emg_padded)
        labels.append(label_encoded)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    # Convert to numpy arrays
    emg_samples = np.array(emg_samples)
    labels = np.array(labels, dtype=object)  # Use object dtype for variable-length sequences
    
    # Save to files
    np.save('collected_data/emg_samples.npy', emg_samples)
    np.save('collected_data/labels.npy', labels)
    
    print(f"\nGenerated {num_samples} synthetic samples")
    print(f"EMG data shape: {emg_samples.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample EMG range: [{emg_samples.min():.3f}, {emg_samples.max():.3f}]")
    
if __name__ == '__main__':
    # Generate synthetic dataset
    generate_dataset(num_samples=100) 