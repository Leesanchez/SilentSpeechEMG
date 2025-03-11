import torch
import torch.nn as nn
import torch.optim as optim
import editdistance
from typing import List, Tuple
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from data_utils import TextTransform
from visualization_utils import plot_confusion_matrix, visualize_predictions, log_augmentation_examples
from transformers import get_cosine_schedule_with_warmup

def compute_wer_cer(predictions: List[str], targets: List[str]) -> Tuple[float, float]:
    """Compute Word Error Rate and Character Error Rate."""
    total_words = 0
    total_chars = 0
    word_errors = 0
    char_errors = 0
    
    for pred, target in zip(predictions, targets):
        # Word-level metrics
        pred_words = pred.split()
        target_words = target.split()
        word_errors += editdistance.eval(pred_words, target_words)
        total_words += len(target_words)
        
        # Character-level metrics
        char_errors += editdistance.eval(pred, target)
        total_chars += len(target)
    
    wer = word_errors / total_words if total_words > 0 else 1.0
    cer = char_errors / total_chars if total_chars > 0 else 1.0
    
    return wer, cer

def decode_predictions(outputs, text_transform):
    """Convert model outputs to text predictions."""
    _, predicted = outputs.max(-1)
    predictions = []
    for seq in predicted:
        text = ''
        for idx in seq:
            if idx == -1:  # Skip padding tokens
                continue
            text += text_transform.idx_to_char[idx.item()]
        predictions.append(text)
    return predictions

def decode_targets(targets, text_transform):
    """Convert target indices to text."""
    texts = []
    for seq in targets:
        text = ''
        for idx in seq:
            if idx == -1:  # Skip padding tokens
                continue
            text += text_transform.idx_to_char[idx.item()]
        texts.append(text)
    return texts

def create_ensemble(num_models: int, vocab_size: int):
    """Create an ensemble of models with specified parameters."""
    from spatiotemporal_model import SilentSpeechTransformer
    
    model_params = {
        'num_channels': 8,
        'input_size': 128,
        'num_classes': vocab_size
    }
    
    from model_ensemble import ModelEnsemble
    return ModelEnsemble(num_models, model_params)

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.criterion = nn.KLDivLoss(reduction='sum')
        
    def forward(self, pred, target):
        # Get shapes
        batch_size, seq_len, n_classes = pred.shape
        
        # Create mask for non-padding tokens
        mask = (target != -1).view(-1)  # Flatten mask to match reshaped tensors
        
        # Reshape predictions and targets
        pred = pred.view(-1, n_classes)  # [batch_size * seq_len, n_classes]
        target = target.view(-1)         # [batch_size * seq_len]
        
        # Apply mask
        pred = pred[mask]    # [valid_tokens, n_classes]
        target = target[mask]  # [valid_tokens]
        
        if len(target) == 0:  # Handle empty case
            return torch.tensor(0.0, device=pred.device)
        
        # Convert targets to one-hot
        target_one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        
        # Apply label smoothing
        target_smooth = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Calculate loss
        pred_log = F.log_softmax(pred, dim=-1)
        loss = self.criterion(pred_log, target_smooth)
        
        # Normalize by number of valid tokens
        return loss / len(target)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    criterion = LabelSmoothing(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Cosine learning rate schedule with warmup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=1e-4)
    
    # Initialize metrics storage
    metrics_history = {
        'train_loss': [], 'train_acc': [], 'train_wer': [], 'train_cer': [],
        'val_loss': [], 'val_acc': [], 'val_wer': [], 'val_cer': []
    }
    
    best_val_loss = float('inf')
    text_transform = TextTransform()
    
    # Log augmentation examples at the start
    if hasattr(train_loader.dataset, 'transform'):
        original_sample = train_loader.dataset.emg_data[0]
        augmented_samples = {}
        
        # Apply each augmentation individually
        for transform in train_loader.dataset.transform.transforms:
            aug_name = transform.__class__.__name__
            augmented_samples[aug_name] = transform(original_sample.copy())
        
        # Log to wandb
        log_augmentation_examples(original_sample, augmented_samples)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_targets = []
        
        for batch_idx, (batch_emg, batch_labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            batch_emg = batch_emg.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_emg)
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy and collect predictions
            mask = (batch_labels != -1)
            _, predicted = outputs.max(-1)
            train_total += mask.sum().item()
            train_correct += ((predicted == batch_labels) & mask).sum().item()
            
            # Collect predictions and targets for WER/CER
            if batch_idx % 10 == 0:  # Compute metrics every 10 batches to save time
                preds = decode_predictions(outputs.detach().cpu(), text_transform)
                targets = decode_targets(batch_labels.cpu(), text_transform)
                all_train_preds.extend(preds)
                all_train_targets.extend(targets)
                
                # Visualize some predictions
                if batch_idx == 0:
                    for i in range(min(3, len(batch_emg))):
                        visualize_predictions(
                            batch_emg[i].cpu().numpy(),
                            targets[i],
                            preds[i],
                            torch.softmax(outputs[i], dim=-1).max().item(),
                            save_path=f"pred_epoch{epoch}_sample{i}.png"
                        )
                        wandb.log({f"predictions/sample_{i}": wandb.Image(f"pred_epoch{epoch}_sample{i}.png")})
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Calculate training WER/CER
        train_wer, train_cer = compute_wer_cer(all_train_preds, all_train_targets)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_emg, batch_labels in val_loader:
                batch_emg = batch_emg.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_emg)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                mask = (batch_labels != -1)
                _, predicted = outputs.max(-1)
                val_total += mask.sum().item()
                val_correct += ((predicted == batch_labels) & mask).sum().item()
                
                # Collect predictions and targets
                preds = decode_predictions(outputs.cpu(), text_transform)
                targets = decode_targets(batch_labels.cpu(), text_transform)
                all_val_preds.extend(preds)
                all_val_targets.extend(targets)
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Calculate validation WER/CER
        val_wer, val_cer = compute_wer_cer(all_val_preds, all_val_targets)
        
        # Update metrics history
        for metric_name, value in [
            ('train_loss', train_loss), ('train_acc', train_acc),
            ('train_wer', train_wer), ('train_cer', train_cer),
            ('val_loss', val_loss), ('val_acc', val_acc),
            ('val_wer', val_wer), ('val_cer', val_cer)
        ]:
            metrics_history[metric_name].append(value)
        
        # Plot and log confusion matrix
        if epoch % 5 == 0:
            plot_confusion_matrix(
                [c for text in all_val_targets for c in text],
                [c for text in all_val_preds for c in text],
                text_transform.char_to_idx,
                save_path=f"confusion_matrix_epoch{epoch}.png"
            )
            wandb.log({
                "confusion_matrix": wandb.Image(f"confusion_matrix_epoch{epoch}.png")
            })
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_wer': train_wer,
            'train_cer': train_cer,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_wer': val_wer,
            'val_cer': val_cer,
            'learning_rate': scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'val_wer': val_wer,
                'val_cer': val_cer,
            }, 'best_model.pth')
            
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Train WER: {train_wer:.4f}, Train CER: {train_cer:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val WER: {val_wer:.4f}, Val CER: {val_cer:.4f}')
        
        # Early stopping check
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
    
    return metrics_history 