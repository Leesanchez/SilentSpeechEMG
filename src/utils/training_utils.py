import torch
import wandb
from pathlib import Path

def setup_wandb(config):
    """Initialize Weights & Biases logging."""
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=config['wandb']['name'],
        config=config,
        tags=config['wandb']['tags']
    )

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    
    print(f'Loaded checkpoint from epoch {start_epoch} with validation loss {val_loss:.4f}')
    return start_epoch, val_loss 