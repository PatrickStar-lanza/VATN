import os
import torch
import pickle
from torch import nn
import pandas as pd
from transformer_v3_1 import Semi_Transformer
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import gzip

device = torch.device("cuda")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
num_epochs = 5

# Load datasets from CSV files
train_df = pd.read_csv('/home/zheng/VATN/completed1/111.csv')
#print(f"train_df:{train_df}")
val_df = pd.read_csv('/home/zheng/VATN/completed1/111.csv')
#print(f"val_df:{val_df}")
train_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in train_df.iloc[:, 0].values]
#print(f"train_files:{train_files}")
val_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in val_df.iloc[:, 0].values]
#print(f"val_files:{val_files}")

action_names = sorted(list(set(train_df.iloc[:, 1].values)))
print(f"action_names:{action_names}")
num_action_classes = len(action_names)

# Define the model and optimizer
model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

def get_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))


def run_training_phase(model, data_files, labels, epoch=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for pkl_file, label in tqdm(zip(data_files, labels), total=len(data_files), desc="Training Phase"):
        with gzip.open(pkl_file, 'rb') as f:
            clip = pickle.load(f).unsqueeze(0).to(device)

        action_idx = action_names.index(label)
        action = torch.tensor([action_idx], device=device)

        with autocast():
            outputs = model(clip)
            loss = criterion(outputs, action)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_acc += get_accuracy(outputs, action).item()

        
        _, preds = torch.max(outputs, dim=1)
        pred_idx = preds.item()

        print(f"Training: action_idx:{action_idx}, pred_idx:{pred_idx}")
        print(f"Training: True label: {label}, Predicted label: {action_names[pred_idx]}")
        
    epoch_loss = running_loss / len(data_files)
    epoch_acc = running_acc / len(data_files)
    print(f"Epoch {epoch}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")



def run_validation_phase(model, data_files, labels, epoch=None):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    correct_predictions = 0  # For tracking correct predictions

    for pkl_file, label in tqdm(zip(data_files, labels), total=len(data_files), desc="Validation Phase"):
        with gzip.open(pkl_file, 'rb') as f:
            clip = pickle.load(f).unsqueeze(0).to(device)

        action_idx = action_names.index(label)
        action = torch.tensor([action_idx], device=device)

        with autocast():
            outputs = model(clip)
            loss = criterion(outputs, action)

        _, preds = torch.max(outputs, dim=1)
        pred_idx = preds.item()
        print(f"pred_idx:{pred_idx}")
        print(f"action_idx:{action_idx}")
        
        if pred_idx == action_idx:
            correct_predictions += 1

        if pred_idx >= len(action_names):
            pred_idx = 0  # Default to 0 for invalid index

        print(f"True label: {label}, Predicted label: {action_names[pred_idx]}")
        print(f"Prediction vector: {outputs.detach().cpu().numpy()}")

        running_loss += loss.item()
        running_acc += get_accuracy(outputs, action).item()

    epoch_loss = running_loss / len(data_files)
    epoch_acc = correct_predictions / len(data_files)  # Calculate accuracy based on correct predictions
    print(f"Epoch {epoch}/{num_epochs} - Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}")



checkpoint_path = "/home/zheng/VATN/checkpoints/last_epoch_model.pth"

# Training Phase
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    run_training_phase(model, train_files, train_df.iloc[:, 1].values, epoch+1)
    if epoch == num_epochs - 1:  
        
        print(f"Saving model to {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)


print(f"Loading model from {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path))
model.eval() 

# Validation Phase
run_validation_phase(model, val_files, val_df.iloc[:, 1].values)





