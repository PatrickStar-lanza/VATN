import os
import torch
import pickle
from torch import nn
import pandas as pd
import gzip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformer_v3_1 import Semi_Transformer  # Ensure this module is in your path

def log_and_print(msg, log_file):
    print(msg)
    log_file.write(msg + "\n")

device = torch.device("cuda")

# Define the model path
model_path = "/proj/speech/ccu_data/transfer-learning-model/dms2313/Self-Supervised-Embedding-Fusion-Transformer/checkpoints_completed/model_epoch_4.pth"

# Open the log file
with open("data_analysis.txt", "a") as log_file:

    # Load data and labels
    val_df = pd.read_csv('/home/zheng/VATN/completed1/test.csv')
    val_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in val_df.iloc[:, 0].values]
    action_names = sorted(list(set(val_df.iloc[:, 1].values)))
    num_action_classes = len(action_names)

    # Initialize the model and loss function
    model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)  
    model = model.half()
    criterion = nn.CrossEntropyLoss()

    log_and_print(f"Loading model from {model_path}", log_file)
    model.load_state_dict(torch.load(model_path))

    # Initialize variables to store results during the validation phase
    val_running_loss = 0.0
    correct_predictions = 0
    correctly_predicted_actions = []
    incorrectly_predicted_actions = []
    action_counts = {name: 0 for name in action_names}
    action_correct_counts = {name: 0 for name in action_names}

    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_action_classes, num_action_classes), dtype=np.int)

    # Start the validation phase
    for pkl_file, label in tqdm(zip(val_files, val_df.iloc[:, 1].values), total=len(val_files), desc="Validation Phase"):
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
        action_counts[label] += 1
        confusion_matrix[action_idx][pred_idx] += 1

        if pred_idx == action_idx:
            correct_predictions += 1
            action_correct_counts[label] += 1
            correctly_predicted_actions.append(label)
        else:
            incorrectly_predicted_actions.append((label, action_names[pred_idx]))

    val_epoch_loss = val_running_loss / len(val_files)
    val_epoch_acc = correct_predictions / len(val_files)
    log_and_print(f"\nValidation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}", log_file)
    
    log_and_print(f"\nCorrectly Predicted Actions: {correctly_predicted_actions}", log_file)
    log_and_print(f"\nIncorrectly Predicted Actions: {incorrectly_predicted_actions}", log_file)
    log_and_print(f"\nAction Accuracy Breakdown:", log_file)
    for action, count in action_counts.items():
        correct_count = action_correct_counts[action]
        accuracy = correct_count / count
        log_and_print(f"Action: {action}, Total: {count}, Correct: {correct_count}, Accuracy: {accuracy:.2f}", log_file)

def plot_confusion_matrix(confusion_matrix, action_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=action_names, yticklabels=action_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_comparison(correct_counts, total_counts, action_names):
    accuracies = [correct_counts[name] / total_counts[name] for name in action_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(len(action_names))

    bar1 = ax.bar(index, accuracies, bar_width, label='Correct Predictions', color='b')
    bar2 = ax.bar(index + bar_width, [1 - acc for acc in accuracies], bar_width, label='Incorrect Predictions', color='r')

    ax.set_xlabel('Action Names')
    ax.set_ylabel('Prediction Accuracy')
    ax.set_title('Comparison of Correct vs Incorrect Predictions')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(action_names, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(confusion_matrix, action_names)
plot_comparison(action_correct_counts, action_counts, action_names)
