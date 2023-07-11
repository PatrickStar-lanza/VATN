import os
import torch
from torch import nn
from transformer_v3_1 import Semi_Transformer

device = torch.device("cuda")

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
num_epochs = 1

# Get the directories of all the actions
action_dirs = [os.path.join("/home/zheng/VATN/collective_action", d) for d in os.listdir("/home/zheng/VATN/collective_action")]

# Get the names of the actions
action_names = [os.path.basename(action_dir).rstrip('0123456789') for action_dir in action_dirs]  # Remove digits from the end of each action name
action_names = list(set(action_names))  # Remove duplicates from the list

# Count the number of action classes
num_action_classes = len(action_names)

print(f"Number of action classes: {num_action_classes}")
print(f"Action classes: {action_names}")

# Define the model and optimizer
model = Semi_Transformer(num_classes=num_action_classes, seq_len=30).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # set learning rate to 0.0001

# Get the directories of all the .pt files
pt_files = [os.path.join("/home/zheng/VATN/action_pt", f) for f in os.listdir("/home/zheng/VATN/action_pt") if f.endswith(".pt")]

for epoch in range(num_epochs):
    # Iterate through each .pt file
    for pt_file in pt_files:
        # Load the tensor from the .pt file
        clip = torch.load(pt_file).to(device)

        # Convert action_dir to a tensor and send it to the same device as the clip
        action_name = os.path.basename(pt_file).rstrip('.pt').rstrip('0123456789')
        action = torch.tensor([action_names.index(action_name)], device=device)

        # Forward pass
        outputs = model(clip)

        # Compute loss
        loss = criterion(outputs, action)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Print loss for each iteration
        print(f"Loss: {loss.item()}")
