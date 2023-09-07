import os
import torch
import pickle
import logging
from torch import nn
import pandas as pd
from torch.optim.lr_scheduler import StepLR  # Learning rate scheduler
from transformer_v3_1 import Semi_Transformer  # Your custom transformer model
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm, trange
import gzip

# Initialize logging
logging.basicConfig(filename="training_log.txt", level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Function to log and print
def log_and_print(message):
    print(message)
    logging.info(message)

# Function to save checkpoints
def save_checkpoint(model, path, epoch):
    checkpoint_path = f"{path}_epoch_{epoch}.pth"
    log_and_print(f"Saving model to {checkpoint_path}")
    torch.save(model.state_dict(), checkpoint_path)

# Initialize variables
device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()
num_epochs = 40
checkpoint_base_path = "/home/zheng/VATN/checkpoints/"
start_epoch = 0

# Load training data
train_df = pd.read_csv('/home/zheng/VATN/completed1/111.csv')
train_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in train_df.iloc[:, 0].values]

# Load validation data
val_df = pd.read_csv('/home/zheng/VATN/completed1/111.csv')
val_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in val_df.iloc[:, 0].values]

# Create action names and action classes
action_names = sorted(list(set(train_df.iloc[:, 1].values) | set(val_df.iloc[:, 1].values)))
num_action_classes = len(action_names)

# Initialize model and optimizer
model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device).half()  # Note the .half() here
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scaler = GradScaler()
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)  # Add learning rate scheduler

# Function to get accuracy
def get_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))

# Function to run training phase
def run_training_phase(model, data_files, labels, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for pkl_file, label in tqdm(zip(data_files, labels), total=len(data_files), desc="Training Phase"):
        with gzip.open(pkl_file, 'rb') as f:
            clip = pickle.load(f).unsqueeze(0).to(device).half()  # Note the .half() here
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
    epoch_loss = running_loss / len(data_files)
    epoch_acc = running_acc / len(data_files)
    log_and_print(f"Epoch {epoch}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

# Function to run validation phase
def run_validation_phase(model, val_files, val_labels):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for pkl_file, label in tqdm(zip(val_files, val_labels), total=len(val_files), desc="Validation Phase"):
            with gzip.open(pkl_file, 'rb') as f:
                clip = pickle.load(f).unsqueeze(0).to(device).half()  # Note the .half() here
            action_idx = action_names.index(label)
            action = torch.tensor([action_idx], device=device)
            with autocast():  # Added autocast for validation
                outputs = model(clip)
                loss = criterion(outputs, action)
            val_loss += loss.item()
            val_acc += get_accuracy(outputs, action).item()
    val_loss /= len(val_files)
    val_acc /= len(val_files)
    log_and_print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    return val_loss, val_acc

best_val_loss = float('inf')  # Initialize best validation loss

# Training Phase
for epoch in trange(start_epoch, num_epochs, initial=start_epoch, total=num_epochs, desc="Training Epochs"): 
    run_training_phase(model, train_files, train_df.iloc[:, 1].values, epoch+1)
    val_loss, val_acc = run_validation_phase(model, val_files, val_df.iloc[:, 1].values)
    scheduler.step()
    if val_loss < best_val_loss:  # If this model is so far the best, save it
        best_val_loss = val_loss
        save_checkpoint(model, checkpoint_base_path, epoch)
