import os
import argparse
import random
import torch
import pickle
from torch import nn
import pandas as pd
from transformer_v3_1 import Semi_Transformer  # Replace with the actual import
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import gzip

# Argument Parser
parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs.')
parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate.')
parser.add_argument('--train_csv', type=str, default='/home/zheng/VATN/completed1/111.csv', help='Path to the train.csv file.')
parser.add_argument('--train_data', type=str, default='/home/zheng/VATN/action_pkl_completed', help='Path to the training data.')
parser.add_argument('--checkpoint_path', type=str, default='/home/zheng/VATN/checkpoints_parse/last_epoch_model.pth', help='Path to save the model checkpoint.')
parser.add_argument('--seq_len', type=int, default=30, help='Sequence length for the model.')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
args = parser.parse_args()

# Set the seed for reproducibility
if args.seed is not None:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)

# Initialize
device = torch.device("cuda")
criterion = nn.CrossEntropyLoss()
train_df = pd.read_csv(args.train_csv)
train_files = [os.path.join(args.train_data, fname) for fname in train_df.iloc[:, 0].values]
action_names = sorted(list(set(train_df.iloc[:, 1].values)))
num_action_classes = len(action_names)
model = Semi_Transformer(num_classes=num_action_classes, seq_len=args.seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

    epoch_loss = running_loss / len(data_files)
    epoch_acc = running_acc / len(data_files)
    print(f"Epoch {epoch}/{args.num_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

# Training Phase
for epoch in tqdm(range(args.num_epochs), desc="Training Epochs"):
    run_training_phase(model, train_files, train_df.iloc[:, 1].values, epoch+1)
    if epoch == args.num_epochs - 1:
        print(f"Saving model to {args.checkpoint_path}")
        torch.save(model.state_dict(), args.checkpoint_path)

print(f"Loading model from {args.checkpoint_path}")
