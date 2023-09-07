# validation_phase.py
import os
import torch
import pickle
from torch import nn
import pandas as pd
from transformer_v3_1 import Semi_Transformer  # Make sure this module is in your path
from tqdm import tqdm
import gzip

def log_and_print(msg, log_file):
    print(msg)
    log_file.write(msg + "\n")

device = torch.device("cuda")

# Define the epoch where you want to start
start_epoch = 14  # You can change this value

# Open the log file
with open("valid_log.txt", "a") as log_file:

    # Load data and labels
    val_df = pd.read_csv('/home/zheng/VATN/completed1/valid.csv')
    val_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in val_df.iloc[:, 0].values]
    action_names = sorted(list(set(val_df.iloc[:, 1].values)))
    num_action_classes = len(action_names)

    # Initialize the model and loss function
    model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)  
    model = model.half()
    criterion = nn.CrossEntropyLoss()

    # Dictionary to store validation results
    val_results = {}

    # Loop over epochs starting from start_epoch to 10
    for epoch in range(start_epoch, 28):  # Modify the range here
        checkpoint_path = f"/proj/speech/ccu_data/transfer-learning-model/dms2313/Self-Supervised-Embedding-Fusion-Transformer/checkpoints_completed2/model_epoch_{epoch}.pth"
        log_and_print(f"Loading model from {checkpoint_path}", log_file)
        model.load_state_dict(torch.load(checkpoint_path))

        # Initialize variables to store results during the validation phase
        val_running_loss = 0.0
        correct_predictions = 0

        # Start the validation phase
        for pkl_file, label in tqdm(zip(val_files, val_df.iloc[:, 1].values), total=len(val_files), desc=f"Validation Phase (Epoch {epoch})"):
            with gzip.open(pkl_file, 'rb') as f:
                clip = pickle.load(f).unsqueeze(0).to(device)

            action_idx = action_names.index(label)
            action = torch.tensor([action_idx], device=device)

            with torch.no_grad():
                outputs = model(clip)
                loss = criterion(outputs, action)

            val_running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            pred_idx = preds.item()
            if pred_idx == action_idx:
                correct_predictions += 1

        val_epoch_loss = val_running_loss / len(val_files)
        val_epoch_acc = correct_predictions / len(val_files)
        log_and_print(f"Validation Loss (Epoch {epoch}): {val_epoch_loss:.4f}, Validation Accuracy (Epoch {epoch}): {val_epoch_acc:.4f}", log_file)

        val_results[epoch] = {'Loss': val_epoch_loss, 'Accuracy': val_epoch_acc}

    with open("validation_results.pkl", "wb") as f:
        pickle.dump(val_results, f)
