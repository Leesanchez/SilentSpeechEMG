import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import git
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import seaborn as sns
import random
from transformers import Adafactor, get_cosine_schedule_with_warmup
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import StepLR
import math
import torch.nn.functional as F

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class EMGDataset(Dataset):
    def __init__(self, data_paths, max_length=8000, max_channels=8, tokenizer=None, chunk_size=1000):
        self.max_length = max_length
        self.max_channels = max_channels
        self.tokenizer = tokenizer if tokenizer else T5Tokenizer.from_pretrained('t5-small')
        self.chunk_size = chunk_size
        self.file_paths = []
        self.cache = {}
        self.cache_size = 100  # Maximum number of samples to keep in memory
        
        # Handle single path or list of paths
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        # Store file paths instead of loading all data into memory
        for data_path in data_paths:
            if os.path.exists(data_path):
                for file in os.listdir(data_path):
                    if file.endswith('_info.json'):
                        json_path = os.path.join(data_path, file)
                        emg_path = json_path.replace('_info.json', '_emg.npy')
                        
                        if os.path.exists(emg_path):
                            try:
                                # Just check if the file is valid without loading it
                                with open(json_path, 'r') as f:
                                    info = json.load(f)
                                    if info['text'].strip():  # Only add if text is not empty
                                        self.file_paths.append((emg_path, json_path))
                            except Exception as e:
                                print(f"Error checking file {emg_path}: {str(e)}")
            else:
                print(f"Warning: Path not found: {data_path}")
                            
        print(f"Found {len(self.file_paths)} valid samples in {len(data_paths)} directories")
    
    def __len__(self):
        return len(self.file_paths)
    
    def _load_and_process_data(self, emg_path, json_path):
        # Load EMG data in chunks to save memory
        emg_data = np.load(emg_path, mmap_mode='r')  # Memory-mapped file
        
        # Process in chunks if the data is large
        if emg_data.shape[1] > self.chunk_size:
            chunks = []
            for i in range(0, emg_data.shape[1], self.chunk_size):
                end = min(i + self.chunk_size, emg_data.shape[1])
                chunk = emg_data[:, i:end].copy()  # Copy only the chunk we need
                chunks.append(chunk)
            emg_data = np.concatenate(chunks, axis=1)
        else:
            emg_data = emg_data.copy()  # Make a copy of the entire array
            
        # Handle channel dimension
        if emg_data.shape[0] > self.max_channels:
            emg_data = emg_data[:self.max_channels]
        elif emg_data.shape[0] < self.max_channels:
            pad_channels = np.zeros((self.max_channels - emg_data.shape[0], emg_data.shape[1]))
            emg_data = np.vstack([emg_data, pad_channels])
        
        # Handle sequence length
        if emg_data.shape[1] > self.max_length:
            emg_data = emg_data[:, :self.max_length]
        elif emg_data.shape[1] < self.max_length:
            pad_width = ((0, 0), (0, self.max_length - emg_data.shape[1]))
            emg_data = np.pad(emg_data, pad_width, mode='constant', constant_values=0)
        
        # Load text
        with open(json_path, 'r') as f:
            info = json.load(f)
            text = info['text']
        
        # Convert to torch tensor
        emg_tensor = torch.FloatTensor(emg_data)
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'emg': emg_tensor,
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'text': text
        }
    
    def __getitem__(self, idx):
        emg_path, json_path = self.file_paths[idx]
        
        # Check if data is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load and process data
        data = self._load_and_process_data(emg_path, json_path)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item if cache is full
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[idx] = data
        return data

def preprocess_signal(emg_data, augment=False):
    """Preprocess EMG signals with advanced augmentation techniques"""
    # Normalize
    emg_data = (emg_data - np.mean(emg_data, axis=1, keepdims=True)) / (np.std(emg_data, axis=1, keepdims=True) + 1e-8)
    
    if augment:
        # 1. Time warping
        if np.random.random() < 0.5:
            stretch_factor = np.random.uniform(0.9, 1.1)
            orig_len = emg_data.shape[1]
            warped_len = int(orig_len * stretch_factor)
            warped_data = np.zeros_like(emg_data)
            for i in range(emg_data.shape[0]):
                warped_data[i] = signal.resample(emg_data[i], warped_len)
            if warped_len > orig_len:
                emg_data = warped_data[:, :orig_len]
            else:
                emg_data = np.pad(warped_data, ((0,0), (0, orig_len - warped_len)), mode='constant')
        
        # 2. Noise injection (multiple types)
        if np.random.random() < 0.5:
            noise_type = np.random.choice(['gaussian', 'pink', 'artifacts'])
            if noise_type == 'gaussian':
                noise = np.random.normal(0, 0.01, emg_data.shape)
            elif noise_type == 'pink':
                noise = np.array([signal.welch(np.random.normal(0, 1, emg_data.shape[1]))[1] for _ in range(emg_data.shape[0])])
                noise = noise / np.max(np.abs(noise)) * 0.01
            else:  # artifacts
                noise = np.zeros_like(emg_data)
                num_artifacts = np.random.randint(1, 4)
                for _ in range(num_artifacts):
                    pos = np.random.randint(0, emg_data.shape[1])
                    width = np.random.randint(10, 50)
                    noise[:, pos:pos+width] = np.random.uniform(-0.02, 0.02)
            emg_data = emg_data + noise
        
        # 3. Channel masking
        if np.random.random() < 0.3:
            num_masks = np.random.randint(1, 3)
            for _ in range(num_masks):
                channel = np.random.randint(0, emg_data.shape[0])
                start = np.random.randint(0, emg_data.shape[1] - 100)
                length = np.random.randint(50, 100)
                emg_data[channel, start:start+length] = 0
        
        # 4. Random scaling per channel
        if np.random.random() < 0.5:
            scales = np.random.uniform(0.9, 1.1, size=(emg_data.shape[0], 1))
            emg_data = emg_data * scales
        
        # 5. Random phase shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-20, 20)
            emg_data = np.roll(emg_data, shift, axis=1)
    
    return emg_data

def extract_features(emg_data, sample_rate=1000):
    """Extract MFCC features with delta coefficients"""
    # Parameters adjusted for 1000-point signal
    frame_length = 128
    hop_length = 64
    
    batch_size = emg_data.shape[0]
    features_list = []
    
    for batch_idx in range(batch_size):
        batch_features = []
        for channel in range(emg_data.shape[1]):
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=emg_data[batch_idx, channel],
                sr=sample_rate,
                n_mfcc=13,
                n_fft=frame_length,
                hop_length=hop_length
            )
            
            # Compute deltas
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Combine features
            channel_features = np.concatenate([mfcc, delta, delta2], axis=0)
            batch_features.append(channel_features)
        
        # Stack all channels
        batch_features = np.concatenate(batch_features, axis=0)
        features_list.append(batch_features.flatten())
    
    # Stack all batch items
    features = np.stack(features_list, axis=0)
    return features

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Multi-head attention with residual connection and layer norm
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed forward with residual connection and layer norm
        fed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(fed_forward))
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class SilentSpeechModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=32128, dropout_rate=0.3):
        super(SilentSpeechModel, self).__init__()
        
        # Initial feature extraction with residual blocks
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(3)
        ])
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads=8, dropout_rate=dropout_rate)
            for _ in range(2)
        ])
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout_rate)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout_rate)
        
        # Classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        batch_size, channels, seq_len = x.size()
        
        # Reshape for feature extraction
        x = x.permute(0, 2, 1).reshape(-1, channels)
        
        # Extract features
        x = self.feature_extractor(x)
        
        # Reshape back to [batch_size, sequence_length, hidden_size]
        x = x.view(batch_size, seq_len, -1)
        
        # Apply residual blocks
        x_res = x.permute(0, 2, 1)  # [batch_size, hidden_size, sequence_length]
        for res_block in self.residual_blocks:
            x_res = res_block(x_res)
        x = x_res.permute(0, 2, 1)  # [batch_size, sequence_length, hidden_size]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(lstm_out)
        
        # Apply multi-head attention
        x_attended, _ = self.attention(x, x, x)
        
        # Global average pooling
        x = torch.mean(x_attended, dim=1)
        
        # Classification
        x = self.classifier(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=8000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_signal(self, emg_data):
        """Preprocess EMG signals"""
        # Normalize
        emg_data = (emg_data - np.mean(emg_data, axis=1, keepdims=True)) / (np.std(emg_data, axis=1, keepdims=True) + 1e-8)
        
        # Apply bandpass filter
        nyquist = 500  # Half of sampling rate (1000 Hz)
        low = 20 / nyquist
        high = 450 / nyquist
        b, a = butter(4, [low, high], btype='band')
        emg_data = filtfilt(b, a, emg_data, axis=1)
        
        # Remove power line interference
        b, a = iirnotch(60.0, 30.0, 1000)
        emg_data = filtfilt(b, a, emg_data, axis=1)
        
        return emg_data

class FeatureExtractor:
    def __init__(self, sample_rate=1000):
        self.sample_rate = sample_rate
        self.frame_length = 128
        self.hop_length = 64
        self.n_mfcc = 13
        self.wavelet_level = 4
        self.pca = PCA(n_components=8)
        self.scaler = StandardScaler()
        
    def extract_wavelet_features(self, signal):
        """Extract wavelet transform features"""
        import pywt
        coeffs = pywt.wavedec(signal, 'db4', level=self.wavelet_level)
        return np.concatenate([np.abs(c) for c in coeffs])
    
    def extract_spectral_features(self, signal):
        """Extract spectral features"""
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=signal, sr=self.sample_rate)[0]
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=signal, sr=self.sample_rate)[0]
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=self.sample_rate)[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=signal)[0]
        
        return np.concatenate([centroid, rolloff, bandwidth, rms])
    
    def extract_temporal_features(self, signal):
        """Extract temporal features"""
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(signal)[0]
        
        # Envelope
        envelope = np.abs(signal)
        
        # Statistical features
        stats = np.array([
            np.mean(signal),
            np.std(signal),
            stats.skew(signal),
            stats.kurtosis(signal),
            np.max(signal),
            np.min(signal),
            np.median(signal)
        ])
        
        return np.concatenate([zcr, envelope, stats])
    
    def extract_features(self, emg_data):
        """Extract comprehensive features from EMG data"""
        batch_size = emg_data.shape[0]
        features_list = []
        
        for batch_idx in range(batch_size):
            batch_features = []
            for channel in range(emg_data.shape[1]):
                channel_data = emg_data[batch_idx, channel]
                
                # 1. MFCC features
                mfcc = librosa.feature.mfcc(
                    y=channel_data,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.frame_length,
                    hop_length=self.hop_length
                )
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                
                # 2. Wavelet features
                wavelet_features = self.extract_wavelet_features(channel_data)
                
                # 3. Spectral features
                spectral_features = self.extract_spectral_features(channel_data)
                
                # 4. Temporal features
                temporal_features = self.extract_temporal_features(channel_data)
                
                # Combine all features
                channel_features = np.concatenate([
                    mfcc.flatten(),
                    delta.flatten(),
                    delta2.flatten(),
                    wavelet_features,
                    spectral_features,
                    temporal_features
                ])
                
                batch_features.append(channel_features)
            
            # Combine features from all channels
            combined_features = np.concatenate(batch_features)
            
            # Apply PCA for dimensionality reduction
            if batch_idx == 0:
                # Fit PCA on first batch
                combined_features = self.pca.fit_transform(combined_features.reshape(1, -1))
            else:
                combined_features = self.pca.transform(combined_features.reshape(1, -1))
            
            features_list.append(combined_features.flatten())
        
        # Stack all batch items
        features = np.stack(features_list, axis=0)
        
        # Normalize features
        if batch_size > 1:
            features = self.scaler.fit_transform(features)
        
        return features

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    input_size = 8  # Number of EMG channels
    hidden_size = 256
    num_classes = 39  # Number of phonemes
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-5
    dropout_rate = 0.3
    label_smoothing = 0.1

    # Create dataset and dataloaders
    train_paths = ["emg_data/silent_parallel_data/emg_data/silent_parallel_data/5-4_silent", 
                  "emg_data/silent_parallel_data/emg_data/silent_parallel_data/5-5_silent", 
                  "emg_data/silent_parallel_data/emg_data/silent_parallel_data/5-6_silent",
                  "emg_data/silent_parallel_data/emg_data/silent_parallel_data/5-10_silent",
                  "emg_data/silent_parallel_data/emg_data/silent_parallel_data/5-11_silent"]
    test_paths = ["emg_data/silent_parallel_data/emg_data/silent_parallel_data/5-8_silent", 
                 "emg_data/silent_parallel_data/emg_data/silent_parallel_data/5-9_silent"]
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    vocab_size = tokenizer.vocab_size
    
    train_dataset = EMGDataset(train_paths, tokenizer=tokenizer)
    test_dataset = EMGDataset(test_paths, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model with correct vocab size
    model = SilentSpeechModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_classes=vocab_size,  # Use tokenizer's vocabulary size
        dropout_rate=dropout_rate
    ).to(device)

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    print("Starting training...")
    model, best_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        patience=10
    )
    print(f"\nTraining completed! Best accuracy: {best_accuracy:.2f}%")

    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_accuracy,
        'hyperparameters': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_classes': num_classes,
            'dropout_rate': dropout_rate
        }
    }, 'final_model.pth')
    print("Model saved to 'final_model.pth'")

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device, patience=10):
    # Enable automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    best_accuracy = 0.0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Curriculum learning - start with easier samples
    def get_curriculum_weight(epoch, max_epochs):
        return min(1.0, (epoch + 1) / (max_epochs * 0.3))
    
    # Contrastive learning temperature parameter
    temperature = 0.07
    
    def contrastive_loss(features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create labels matrix
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_matrix = labels_matrix.float()
        
        # Mask out self-similarity
        mask = torch.eye(labels_matrix.shape[0], dtype=torch.bool, device=device)
        labels_matrix = labels_matrix.masked_fill(mask, 0)
        
        # Compute positive and negative pairs
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        positives = (similarity_matrix * labels_matrix).sum(dim=1)
        negatives = torch.logsumexp(similarity_matrix, dim=1)
        
        loss = -positives + negatives
        return loss.mean()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        # Get curriculum weight for this epoch
        curr_weight = get_curriculum_weight(epoch, num_epochs)
        
        for batch_idx, batch in enumerate(train_loader):
            # Get EMG data and labels from batch
            inputs = batch['emg'].to(device)
            labels = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Take only the first token as target
            target_labels = labels[:, 0]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                # Get model outputs and intermediate features
                outputs = model(inputs)
                features = model.feature_extractor(inputs.permute(0, 2, 1).reshape(-1, inputs.size(1)))
                features = features.view(inputs.size(0), -1)
                
                # Compute main loss
                main_loss = criterion(outputs, target_labels)
                
                # Compute contrastive loss
                contr_loss = contrastive_loss(features, target_labels)
                
                # Combine losses with curriculum weight
                loss = curr_weight * main_loss + (1 - curr_weight) * contr_loss
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with mixed precision
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            # Statistics
            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += target_labels.size(0)
            correct_train += predicted.eq(target_labels).sum().item()
            
            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Main Loss: {main_loss.item():.4f}, '
                      f'Contrastive Loss: {contr_loss.item():.4f}, '
                      f'Accuracy: {100.*correct_train/total_train:.2f}%')
        
        # Calculate training metrics
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_acc = 100. * correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['emg'].to(device)
                labels = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Take only the first token as target
                target_labels = labels[:, 0]
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, target_labels)
                
                total_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += target_labels.size(0)
                correct_val += predicted.eq(target_labels).sum().item()
        
        # Calculate validation metrics
        epoch_val_loss = total_val_loss / len(test_loader)
        epoch_val_acc = 100. * correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Print epoch results
        print(f'\nEpoch [{epoch+1}/{num_epochs}]:')
        print(f'Training Loss: {epoch_train_loss:.4f}, Training Accuracy: {epoch_train_acc:.2f}%')
        print(f'Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.2f}%')
        print(f'Curriculum Weight: {curr_weight:.2f}')
        
        # Save best model and check for early stopping
        if epoch_val_acc > best_accuracy:
            best_accuracy = epoch_val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f'New best model saved! Accuracy: {best_accuracy:.2f}%')
        else:
            patience_counter += 1
            print(f'Patience counter: {patience_counter}/{patience}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
    return model, best_accuracy

if __name__ == "__main__":
    main() 