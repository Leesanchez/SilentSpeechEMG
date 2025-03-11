import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SpatiotemporalCNN(nn.Module):
    def __init__(self, num_channels=8, input_size=128):
        super().__init__()
        
        # Spatial convolution across channels
        self.spatial_conv = nn.Conv1d(num_channels, 64, kernel_size=1)
        
        # Temporal convolutions with residual connections and instance normalization
        self.temporal_block1 = nn.Sequential(
            nn.InstanceNorm1d(64),  # Use InstanceNorm instead of LayerNorm
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.temporal_block2 = nn.Sequential(
            nn.InstanceNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.temporal_block3 = nn.Sequential(
            nn.InstanceNorm1d(256),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Residual projection layers
        self.res_proj1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2),
            nn.InstanceNorm1d(128)
        )
        
        self.res_proj2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=1, stride=2),
            nn.InstanceNorm1d(256)
        )
        
        self.res_proj3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, stride=2),
            nn.InstanceNorm1d(512)
        )
        
        # Final processing
        self.final_proj = nn.Sequential(
            nn.InstanceNorm1d(512),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Adaptive pooling to get fixed sequence length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(5)  # Match label sequence length
        
    def forward(self, x):
        # Initial spatial convolution
        x = self.spatial_conv(x)
        
        # Residual blocks
        res1 = self.res_proj1(x)
        x = self.temporal_block1(x)
        x = x + res1
        
        res2 = self.res_proj2(x)
        x = self.temporal_block2(x)
        x = x + res2
        
        res3 = self.res_proj3(x)
        x = self.temporal_block3(x)
        x = x + res3
        
        # Final processing
        x = self.final_proj(x)
        x = self.adaptive_pool(x)
        return x

class SilentSpeechTransformer(nn.Module):
    def __init__(self, num_channels=8, input_size=128, num_classes=40):
        super().__init__()
        
        # Spatiotemporal CNN for feature extraction
        self.cnn = SpatiotemporalCNN(num_channels, input_size)
        
        # Transformer parameters
        self.d_model = 512
        self.nhead = 8
        self.num_layers = 8
        self.dropout = 0.2
        
        # Linear projection to transformer dimension
        self.input_proj = nn.Sequential(
            nn.Linear(256, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer encoder with pre-norm architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=2048,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=encoder_norm
        )
        
        # Output layers with skip connection and layer norm
        self.fc1 = nn.Linear(self.d_model, self.d_model * 2)
        self.fc2 = nn.Linear(self.d_model * 2, num_classes)
        self.layer_norm1 = nn.LayerNorm(self.d_model * 2)
        self.layer_norm2 = nn.LayerNorm(num_classes)
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)  # (batch, 256, time)
        
        # Prepare for transformer
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x = self.input_proj(x)  # Project to d_model dimension
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Output layers with residual connections and layer norm
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.layer_norm1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layer_norm2(x)
        
        return x

class PhonemeLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, memory, tgt_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        return output

def create_phoneme_mask(size):
    """Create mask for transformer to prevent attending to future tokens."""
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask 