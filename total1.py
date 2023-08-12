import os
import torch
import pickle
from torch import nn
import json
from transformer_v3_1 import Semi_Transformer

device = torch.device("cuda")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
num_epochs = 10
early_stopping_criteria = 5

# Load data split
with open('data_split.json', 'r') as f:
    data_split = json.load(f)

train_files = [item for sublist in data_split['train'].values() for item in sublist]
val_files = [item for sublist in data_split['val'].values() for item in sublist]
test_files = [item for sublist in data_split['test'].values() for item in sublist]

print(f"Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

# Get the names of the actions
action_names = [os.path.splitext(os.path.basename(train_file))[0].rstrip('0123456789') for train_file in train_files]  # Remove digits from the end of each action name
action_names = list(set(action_names))  # Remove duplicates from the list



# Count the number of action classes
num_action_classes = len(action_names)

print(f"Number of action classes: {num_action_classes}")
print(f"Action classes: {action_names}")

# Define the model and optimizer
model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # set learning rate to 0.0001

# Function to evaluate accuracy
def get_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))

# Early stopping initialization
best_val_loss = float('inf')
no_improve_epochs = 0

for epoch in range(num_epochs):
    for phase in ['train', 'val', 'test']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_acc = 0.0

        data_files = train_files if phase == 'train' else (val_files if phase == 'val' else test_files)


        # Iterate through each .pkl file
        for pkl_file in data_files:
            # Load the tensor from the .pkl file
            with open(pkl_file, 'rb') as f:
                clip = pickle.load(f).unsqueeze(0).to(device)

            # Convert action_dir to a tensor and send it to the same device as the clip
            action_name = os.path.splitext(os.path.basename(pkl_file))[0].rstrip('0123456789')
            action = torch.tensor([action_names.index(action_name)], device=device)
            print(f"Label: {action.item()}, Count: {action_names.count(action_name)}")



            # Forward pass
            outputs = model(clip)
            
            print(f"Outputs: {outputs.detach().cpu().numpy()}, Targets: {action.detach().cpu().numpy()}") 

            # Compute loss
            loss = criterion(outputs, action)

            # Backward pass and optimize
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute metrics
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, action)
            
            _, preds = torch.max(outputs, dim=1) 
            print(f"Predictions: {preds.detach().cpu().numpy()}, Actual: {action.detach().cpu().numpy()}")  


        epoch_loss = running_loss / len(data_files)
        epoch_acc = running_acc / len(data_files)

        print(f"Epoch {epoch+1}/{num_epochs}, {phase} Loss: {epoch_loss:.4f}, {phase} Accuracy: {epoch_acc:.4f}")
        
        # Save the model after each epoch
        torch.save(model.state_dict(), f"/home/zheng/VATN/checkpoints/model_epoch_{epoch+1}.pt")
        
        # Early stopping check
        if phase == 'val':
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_criteria:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    break