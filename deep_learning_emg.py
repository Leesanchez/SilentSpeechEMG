import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------
# Data Loading & Preprocessing
# -------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values   # Labels

    # Encode labels to sequential indices
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Normalize Data
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# -------------------------
# Improved Neural Network
# -------------------------
class EMGNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EMGNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# -------------------------
# Training Function
# -------------------------
def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=30):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        scheduler.step(running_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}, Train Accuracy: {100 * correct / total:.2f}%")

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    file_path = 'emg_data.csv'
    X, y = load_data(file_path)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = EMGNet(input_dim=X.shape[1], num_classes=len(set(y.numpy())))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=30)
