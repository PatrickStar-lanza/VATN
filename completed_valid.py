# validation_phase.py
import os
import torch
import pickle
from torch import nn
import pandas as pd
from transformer_v3_1 import Semi_Transformer  # Make sure this module is in your path
from tqdm import tqdm
import gzip

device = torch.device("cuda")

# Load the saved model parameters
checkpoint_path = "/home/zheng/VATN/checkpoints/last_epoch_model.pth"
print(f"Loading model from {checkpoint_path}")
model = Semi_Transformer(num_classes=11, seq_len=30).to(device)  # Change num_classes based on your setup
model = model.half() 
model.load_state_dict(torch.load(checkpoint_path))

# Load data and labels
val_df = pd.read_csv('/home/zheng/VATN/completed1/111.csv')
val_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in val_df.iloc[:, 0].values]
action_names = sorted(list(set(val_df.iloc[:, 1].values)))
num_action_classes = len(action_names)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Initialize variables to store results during the validation phase
val_running_loss = 0.0
correct_predictions = 0

# Start the validation phase
for pkl_file, label in tqdm(zip(val_files, val_df.iloc[:, 1].values), total=len(val_files), desc="Validation Phase"):
    with gzip.open(pkl_file, 'rb') as f:
        clip = pickle.load(f).unsqueeze(0).to(device)

    action_idx = action_names.index(label)
    action = torch.tensor([action_idx], device=device)

    # We don't use autocast and scaler here since we don't do backpropagation
    with torch.no_grad():  # Ensure that gradients are not computed during validation
        outputs = model(clip)
        loss = criterion(outputs, action)

    val_running_loss += loss.item()

    # Compute predicted label and update the number of correct predictions
    _, preds = torch.max(outputs, dim=1)
    pred_idx = preds.item()
    if pred_idx == action_idx:
        correct_predictions += 1

    print(f"Validation: action_idx:{action_idx}, pred_idx:{pred_idx}")
    print(f"Validation: True label: {label}, Predicted label: {action_names[pred_idx]}")

# Compute and display the total loss and accuracy for the validation phase
val_epoch_loss = val_running_loss / len(val_files)
val_epoch_acc = correct_predictions / len(val_files)

print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}")
