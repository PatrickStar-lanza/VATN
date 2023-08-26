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

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
num_epochs = 10

# Load datasets from CSV files
train_df = pd.read_csv('/home/zheng/VATN/completed1/train.csv')
val_df = pd.read_csv('/home/zheng/VATN/completed1/valid.csv')
test_df = pd.read_csv('/home/zheng/VATN/completed1/test.csv')

train_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in train_df.iloc[:, 0].values]
val_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in val_df.iloc[:, 0].values]
test_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in test_df.iloc[:, 0].values]

action_names = sorted(list(set(train_df.iloc[:, 1].values)))
num_action_classes = len(action_names)

print(f"Number of action classes: {num_action_classes}")
print(f"Action classes: {action_names}")

# Define the model and optimizer
model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

def get_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))

def run_phase(phase, model, data_files, labels, epoch=None):
    class_correct = [0.0 for _ in range(num_action_classes)]
    class_total = [0.0 for _ in range(num_action_classes)]

    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0

    pos = 0 if phase == 'train' else 1
    
    for pkl_file, label in tqdm(zip(data_files, labels), total=len(data_files), desc=f"{phase} Phase", position=pos, leave=True):
        with gzip.open(pkl_file, 'rb') as f:
            clip = pickle.load(f).unsqueeze(0).to(device)

        action_idx = action_names.index(label)
        action = torch.tensor([action_idx], device=device)

        with autocast():
            outputs = model(clip)
            loss = criterion(outputs, action)

        if phase == 'train':
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        _, preds = torch.max(outputs, dim=1)
        correct_tensor = preds.eq(action.data.view_as(preds))
        correct = correct_tensor.item()

        class_correct[action_idx] += correct
        class_total[action_idx] += 1

        running_loss += loss.item()
        running_acc += get_accuracy(outputs, action)

    epoch_loss = running_loss / len(data_files)
    epoch_acc = running_acc / len(data_files)

    if epoch:
        print(f"Epoch {epoch}/{num_epochs} - {phase} Loss: {epoch_loss:.4f}, {phase} Accuracy: {epoch_acc:.4f}")
    else:
        print(f"{phase} Loss: {epoch_loss:.4f}, {phase} Accuracy: {epoch_acc:.4f}")

    if phase != 'train':
        for i in range(num_action_classes):
            if class_total[i] > 0:
                print(f"Accuracy of {action_names[i]} : {100 * class_correct[i] / class_total[i]:.2f}%")
            else:
                print(f"Accuracy of {action_names[i]} : N/A (no samples)")

    return epoch_loss

best_val_acc = 0.0

# Training and Validation Phase
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    run_phase('train', model, train_files, train_df.iloc[:, 1].values, epoch+1)
    

    val_acc = run_phase('val', model, val_files, val_df.iloc[:, 1].values)
    

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"/home/zheng/VATN/checkpoints/model_best.pt")

# Test Phase
# model.load_state_dict(torch.load(f"/home/zheng/VATN/checkpoints/model_best.pt"))
run_phase('test', model, test_files, test_df.iloc[:, 1].values)
