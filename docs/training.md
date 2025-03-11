# Training Pipeline Documentation

## Overview
The training pipeline is designed to train the Silent Speech Recognition model using EMG data. It includes features such as:
- Configurable model architecture and training parameters
- Automatic checkpointing and model resumption
- Early stopping to prevent overfitting
- Learning rate scheduling
- Integration with Weights & Biases for experiment tracking
- Comprehensive visualization and metrics logging

## Prerequisites
Before starting training, ensure you have:
1. Installed all required packages:
   ```bash
   pip install torch tqdm wandb pyyaml
   ```
2. Set up a Weights & Biases account (https://wandb.ai)
3. Organized your data in the correct directory structure:
   ```
   data/
   ├── train/      # Training data
   ├── val/        # Validation data
   └── test/       # Test data
   ```

## Configuration
The training pipeline is configured using a YAML file (`configs/training_config.yaml`). Key configuration sections include:

### Model Configuration
```yaml
model:
  num_channels: 8        # Number of EMG channels
  input_size: 128       # Input feature size
  hidden_size: 256      # Hidden layer size
  num_heads: 8          # Number of attention heads
  num_layers: 6         # Number of transformer layers
  dropout: 0.1         # Dropout rate
```

### Data Configuration
```yaml
data:
  train_dir: "data/train"
  val_dir: "data/val"
  test_dir: "data/test"
  sampling_rate: 1000    # EMG sampling rate in Hz
  window_size: 256      # Window size for feature extraction
  hop_length: 128       # Hop length for feature extraction
```

### Training Configuration
```yaml
training:
  num_epochs: 100       # Maximum number of epochs
  batch_size: 32        # Batch size
  learning_rate: 0.001  # Initial learning rate
  weight_decay: 0.0001  # Weight decay for regularization
  label_smoothing: 0.1  # Label smoothing factor
  grad_clip: 5.0        # Gradient clipping threshold
  num_workers: 4        # Number of data loading workers
  log_interval: 10      # Steps between logging
  save_interval: 5      # Epochs between checkpoints
  viz_interval: 1       # Epochs between visualizations
  patience: 10          # Early stopping patience
  checkpoint_dir: "checkpoints"
```

## Running Training

### Starting a New Training Run
```bash
python src/train.py --config configs/training_config.yaml
```

### Resuming from a Checkpoint
```bash
python src/train.py --config configs/training_config.yaml --resume checkpoints/best_model.pth
```

## Monitoring Training

### Weights & Biases Dashboard
The training pipeline logs the following metrics to W&B:
- Training loss
- Validation loss
- Learning rate
- Confusion matrices
- Prediction visualizations

To view training progress:
1. Log in to your W&B account
2. Navigate to your project dashboard
3. Select your run to view detailed metrics and visualizations

### Local Checkpoints
Checkpoints are saved in the specified `checkpoint_dir`:
- `best_model.pth`: Best model based on validation loss
- `checkpoint_epoch_N.pth`: Regular checkpoints every `save_interval` epochs

## Training Features

### Early Stopping
Training automatically stops if validation loss doesn't improve for `patience` epochs.

### Learning Rate Scheduling
The learning rate is reduced by a factor of 0.5 when validation loss plateaus for 5 epochs.

### Label Smoothing
Label smoothing is applied to prevent overconfident predictions and improve generalization.

### Gradient Clipping
Gradients are clipped to prevent exploding gradients during training.

## Customization

### Adding New Metrics
To add new metrics, modify the `validate()` method in `src/train.py`.

### Custom Visualizations
Add new visualization functions in `src/visualization/visualization_utils.py`.

### Model Architecture Changes
Modify the model architecture in `src/models/spatiotemporal_model.py`.

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or model size in config
2. **Slow Training**: Adjust number of workers or batch size
3. **Overfitting**: Increase dropout or weight decay
4. **Poor Convergence**: Adjust learning rate or model architecture

### Memory Usage
Approximate memory requirements:
- Model: ~100MB
- Batch (32 samples): ~500MB
- Total GPU memory needed: ~2GB

## Best Practices

1. **Start Small**: Begin with a smaller model and dataset to verify setup
2. **Monitor Early**: Check W&B dashboard frequently during initial epochs
3. **Regular Backups**: Keep copies of best checkpoints
4. **Experiment Tracking**: Use meaningful names for W&B runs
5. **Resource Management**: Adjust batch size and workers based on available hardware 