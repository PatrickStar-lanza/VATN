# test_phase.py
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
test_df = pd.read_csv('/home/zheng/VATN/completed1/111.csv')
test_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in test_df.iloc[:, 0].values]
action_names = sorted(list(set(test_df.iloc[:, 1].values)))
num_action_classes = len(action_names)

model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)  # Change num_classes based on your setup
model = model.half()
model.load_state_dict(torch.load(checkpoint_path))
# Initialize variables to store results during the test phase
test_running_loss = 0.0
correct_predictions = 0

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Start the test phase
for pkl_file, label in tqdm(zip(test_files, test_df.iloc[:, 1].values), total=len(test_files), desc="Test Phase"):
    with gzip.open(pkl_file, 'rb') as f:
        clip = pickle.load(f).unsqueeze(0).to(device)

    action_idx = action_names.index(label)
    action = torch.tensor([action_idx], device=device)

    with torch.no_grad():  # Ensure that gradients are not computed during testing
        outputs = model(clip)
        loss = criterion(outputs, action)

    test_running_loss += loss.item()

    # Compute predicted label and update the number of correct predictions
    _, preds = torch.max(outputs, dim=1)
    pred_idx = preds.item()
    if pred_idx == action_idx:
        correct_predictions += 1

    print(f"Test: action_idx:{action_idx}, pred_idx:{pred_idx}")
    print(f"Test: True label: {label}, Predicted label: {action_names[pred_idx]}")

# Compute and display the total loss and accuracy for the test phase
test_epoch_loss = test_running_loss / len(test_files)
test_epoch_acc = correct_predictions / len(test_files)

print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.4f}")
