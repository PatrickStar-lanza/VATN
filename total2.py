import os
import torch
import pickle
from torch import nn
import json
from transformer_v3_1 import Semi_Transformer

device = torch.device("cuda")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
num_epochs = 2

# Load data split
with open('data_split.json', 'r') as f:
    data_split = json.load(f)

train_files = [item for sublist in data_split['train'].values() for item in sublist]
val_files = [item for sublist in data_split['val'].values() for item in sublist]
test_files = [item for sublist in data_split['test'].values() for item in sublist]

# Get the names of the actions
action_names = [os.path.splitext(os.path.basename(train_file))[0].rstrip('0123456789') for train_file in train_files]
action_names = list(set(action_names))

# Count the number of action classes
num_action_classes = len(action_names)

print(f"Number of action classes: {num_action_classes}")
print(f"Action classes: {action_names}")

# Define the model and optimizer
model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

def get_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))

def run_phase(phase, model, data_files, epoch=None):
    class_correct = [0.0 for _ in range(num_action_classes)]
    class_total = [0.0 for _ in range(num_action_classes)]

    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0

    for pkl_file in data_files:
        with open(pkl_file, 'rb') as f:
            clip = pickle.load(f).unsqueeze(0).to(device)

        action_name = os.path.splitext(os.path.basename(pkl_file))[0].rstrip('0123456789')
        action_idx = action_names.index(action_name)
        action = torch.tensor([action_idx], device=device)

        outputs = model(clip)
        loss = criterion(outputs, action)

        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    # Print accuracy for each class
    if phase != 'train':
        for i in range(num_action_classes):
            if class_total[i] > 0:
                print(f"Accuracy of {action_names[i]} : {100 * class_correct[i] / class_total[i]:.2f}%")
            else:
                print(f"Accuracy of {action_names[i]} : N/A (no samples)")

    return epoch_loss

# Training Phase
for epoch in range(num_epochs):
    run_phase('train', model, train_files, epoch+1)
    torch.save(model.state_dict(), f"/home/zheng/VATN/checkpoints/model_epoch_{epoch+1}.pt")

# Validation Phase (Once after all training epochs)
run_phase('val', model, val_files)

# Test Phase (Once after all training epochs and validation)
run_phase('test', model, test_files)
