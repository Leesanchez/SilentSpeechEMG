import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from silent_speech_recognition import (
    EMGDataset,
    SilentSpeechModel,
    Preprocessor,
    FeatureExtractor,
    train_model,
    preprocess_signal,
    extract_features
)
import matplotlib.pyplot as plt
from transformers import T5Tokenizer
import gc
from torch.utils.tensorboard import SummaryWriter
import datetime

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (emg_data, target_ids) in enumerate(test_loader):
            emg_data = emg_data.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            output = model(emg_data)
            
            # Ensure target_ids matches output batch size
            target = target_ids[:, 0]  # Take first token as target
            
            # Calculate loss
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            
            # Update metrics
            total_loss += loss.item() * emg_data.size(0)
            total_samples += emg_data.size(0)
    
    return total_loss / total_samples

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize components
    preprocessor = Preprocessor()
    feature_extractor = FeatureExtractor()
    
    # Initialize tokenizer with memory-efficient settings
    tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=128)

    # Load and preprocess dataset
    print("Loading dataset...")
    try:
        dataset = EMGDataset('data/emg_data/silent_parallel_data', tokenizer=tokenizer)
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Split dataset into train, validation, and test
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders with smaller batch sizes
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,  # Reduced batch size
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=True
    )

    # Model configuration
    config = {
        'input_size': 8000,  # Max sequence length
        'hidden_size': 256,  # Reduced hidden size
        'num_classes': tokenizer.vocab_size,
        'dropout_rate': 0.3,
        'num_transformer_layers': 4,
        'num_attention_heads': 8
    }

    try:
        # Initialize model
        model = SilentSpeechModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        ).to(device)

        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        # Initialize optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler with warmup
        num_epochs = 100
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=2e-4,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Loss function with label smoothing
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=tokenizer.pad_token_id)

        # Train model
        print("Starting training...")
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            patience=15  # Early stopping patience
        )

        # Save model and training history
        print("Saving checkpoint...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, 'model_checkpoint.pth')

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()

        # Clear some memory before evaluation
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Evaluate on test set
        print("\nEvaluating on test set...")
        try:
            test_loss = evaluate(model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f}")
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")

        # Add TensorBoard logging
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        writer = SummaryWriter(f'runs/silent_speech_{current_time}')

        # Add in training loop
        for epoch, loss in enumerate(train_losses):
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # Add in validation loop
        for epoch, val_loss in enumerate(val_losses):
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Metrics/WER', wer, epoch)
            writer.add_scalar('Metrics/CER', cer, epoch)

        # Close writer at the end
        writer.close()

    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

    print("Training complete!")

if __name__ == '__main__':
    main() 