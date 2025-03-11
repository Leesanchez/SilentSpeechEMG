import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from pathlib import Path
import yaml
from tqdm import tqdm

from models.spatiotemporal_model import SilentSpeechTransformer
from data.data_utils import TextTransform, EMGDataset
from models.model_utils import LabelSmoothing
from utils.training_utils import setup_wandb, save_checkpoint, load_checkpoint
from visualization.visualization_utils import plot_confusion_matrix, plot_predictions

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_components()
        
    def setup_components(self):
        """Initialize all components needed for training."""
        # Text transform for converting between text and integers
        self.text_transform = TextTransform()
        
        # Model initialization
        self.model = SilentSpeechTransformer(
            num_channels=self.config['model']['num_channels'],
            input_size=self.config['model']['input_size'],
            num_classes=len(self.text_transform.chars)
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = LabelSmoothing(
            size=len(self.text_transform.chars),
            padding_idx=self.text_transform.char_to_int['<pad>'],
            smoothing=self.config['training']['label_smoothing']
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Datasets and dataloaders
        train_dataset = EMGDataset(
            data_dir=self.config['data']['train_dir'],
            transform=self.text_transform
        )
        val_dataset = EMGDataset(
            data_dir=self.config['data']['val_dir'],
            transform=self.text_transform
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers']
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers']
        )
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (emg, text) in enumerate(progress_bar):
            emg = emg.to(self.device)
            text = text.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(emg)
            loss = self.criterion(output, text)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
            if batch_idx % self.config['training']['log_interval'] == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'step': batch_idx
                })
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for emg, text in tqdm(self.val_loader, desc='Validating'):
                emg = emg.to(self.device)
                text = text.to(self.device)
                
                output = self.model(emg)
                loss = self.criterion(output, text)
                total_loss += loss.item()
                
                predictions = output.argmax(dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(text.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = total_loss / len(self.val_loader)
        
        # Generate and log visualizations
        if epoch % self.config['training']['viz_interval'] == 0:
            confusion_matrix = plot_confusion_matrix(
                all_targets,
                all_predictions,
                self.text_transform.chars
            )
            predictions_plot = plot_predictions(
                all_predictions[:3],
                all_targets[:3],
                self.text_transform
            )
            
            wandb.log({
                'val_loss': val_loss,
                'confusion_matrix': wandb.Image(confusion_matrix),
                'predictions': wandb.Image(predictions_plot),
                'epoch': epoch
            })
        
        return val_loss
    
    def train(self):
        """Main training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    Path(self.config['training']['checkpoint_dir']) / 'best_model.pth'
                )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['training']['patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
            
            # Regular checkpoint saving
            if epoch % self.config['training']['save_interval'] == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    Path(self.config['training']['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pth'
                )

def main():
    parser = argparse.ArgumentParser(description='Train Silent Speech Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup wandb
    setup_wandb(config)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        load_checkpoint(trainer.model, trainer.optimizer, args.resume)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main() 