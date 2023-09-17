import os
import torch
import pickle
import logging
from torch import nn
import pandas as pd
from transformer_v3_2 import Semi_Transformer
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
num_epochs = 21
start_epoch = 0

# Load data
train_df = pd.read_csv('/home/zheng/VATN/completed1/train.csv')
train_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in train_df.iloc[:, 0].values]
action_names = sorted(list(set(train_df.iloc[:, 1].values)))
num_action_classes = len(action_names)

# Initialize model and optimizer
model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
scaler = GradScaler()

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
    epoch_loss = running_loss / len(data_files)
    epoch_acc = running_acc / len(data_files)
    log_and_print(f"Epoch {epoch}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

checkpoint_base_path = "/proj/speech/ccu_data/transfer-learning-model/dms2313/Self-Supervised-Embedding-Fusion-Transformer/structure_compare/model"

# Define an initial checkpoint path (update this path if you have a pre-existing model)
initial_checkpoint_path = "/proj/speech/ccu_data/transfer-learning-model/dms2313/Self-Supervised-Embedding-Fusion-Transformer/structure_compare/model_epoch_14.pth"
 #"/home/zheng/VATN/checkpoints/model_epoch_13.pth"

# Check if an initial checkpoint exists

if initial_checkpoint_path:
    if os.path.exists(initial_checkpoint_path):
        log_and_print(f"Loading model from {initial_checkpoint_path}")
        model.load_state_dict(torch.load(initial_checkpoint_path))
        start_epoch = int(initial_checkpoint_path.split('_epoch_')[-1].split('.pth')[0]) + 1

# Training Phase
for epoch in trange(start_epoch, num_epochs, initial=start_epoch, total=num_epochs, desc="Training Epochs"): 
    run_training_phase(model, train_files, train_df.iloc[:, 1].values, epoch+1)
    save_checkpoint(model, checkpoint_base_path, epoch)
