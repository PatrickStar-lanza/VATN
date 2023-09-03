import os
import torch
import pickle
from torch import nn
import pandas as pd
from transformer_v3_1 import Semi_Transformer  # Your custom Transformer model
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import gzip

device = torch.device("cuda")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
num_epochs = 10

# Load datasets from CSV files for training and validation
train_df = pd.read_csv('/home/zheng/VATN/completed1/111.csv')
val_df = pd.read_csv('/home/zheng/VATN/completed1/111.csv')

train_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in train_df.iloc[:, 0].values]
val_files = [os.path.join("/home/zheng/VATN/action_pkl_completed", fname) for fname in val_df.iloc[:, 0].values]

action_names = sorted(list(set(train_df.iloc[:, 1].values)))
num_action_classes = len(action_names)

print(f"Number of action classes: {num_action_classes}")
print(f"Action classes: {action_names}")

# Define the model and optimizer
model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# Print the model structure
#print("Model Structure:")
#print(model)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

#def check_model_weights(model):
#    print(f"Model object ID: {id(model)}")
#    print("Model weights summary:")
#    for name, param in model.named_parameters():
#        print(f"{name}: Mean = {param.data.mean()}, Std = {param.data.std()}")


def get_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))

def run_phase(phase, model, data_files, labels, epoch=None):

#    print(f"Before {phase} phase:")
#    check_model_weights(model)

    class_correct = [0.0 for _ in range(num_action_classes)]
    class_total = [0.0 for _ in range(num_action_classes)]

    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0

    for pkl_file, label in tqdm(zip(data_files, labels), total=len(data_files), desc=f"{phase} Phase", position=0, leave=True):
        with gzip.open(pkl_file, 'rb') as f:
            clip = pickle.load(f).unsqueeze(0).to(device)

        action_idx = action_names.index(label)
        action = torch.tensor([action_idx], device=device)

        with autocast():
            outputs = model(clip)
            loss = criterion(outputs, action)

        if phase == 'val':  # Only print details during validation phase
            print(f"Output shape: {outputs.shape}")  
            print(f"Sample outputs: {outputs[0]}")  
            print(f"Loss: {loss.item()}")  

        if phase == 'train':
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        _, preds = torch.max(outputs, dim=1)

        if phase == 'val':  # Only print details during validation phase
            print(f"Predicted indices: {preds}")  
            print(f"Max values: {torch.max(outputs, 1).values}")  

        pred_idx = preds.item()
        if pred_idx >= len(action_names):
            print(f"warning:invalid index {pred_idx} default is 0")
            pred_idx = 0

        if phase == 'val':  # Only print details during validation phase
            print(f"True label: {label}, Predicted label: {action_names[pred_idx]}")

        correct_tensor = preds.eq(action.data.view_as(preds))
        correct = correct_tensor.item()
        class_correct[action_idx] += correct
        class_total[action_idx] += 1

        running_loss += loss.item()
        running_acc += get_accuracy(outputs, action).item()

    epoch_loss = running_loss / len(data_files)
    epoch_acc = running_acc / len(data_files)
    print(f"Epoch {epoch}/{num_epochs} - {phase} Loss: {epoch_loss:.4f}, {phase} Accuracy: {epoch_acc:.4f}")

    if phase == 'val':
        for i in range(num_action_classes):
            if class_total[i] > 0:
                print(f"Accuracy of {action_names[i]} : {100 * class_correct[i] / class_total[i]:.2f}%")
            else:
                print(f"Accuracy of {action_names[i]} : N/A (no samples)")
#    print(f"After {phase} phase:")
#    check_model_weights(model)
    
    return epoch_acc

# Training and Validation Phases
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    run_phase('train', model, train_files, train_df.iloc[:, 1].values, epoch+1)
    run_phase('val', model, val_files, val_df.iloc[:, 1].values, epoch+1)
