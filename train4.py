import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from transformer_v3_1 import Semi_Transformer

# Image transformation operations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# Define the placeholder image for padding
padding_img = torch.zeros(3, 224, 224)

for epoch in range(num_epochs):
    # Iterate through each action directory
    for action_dir in action_dirs:
        # Get all frames within the action directory
        frame_files = [os.path.join(action_dir, f) for f in os.listdir(action_dir) if f.isdigit()]
        frame_files.sort(key=lambda x: int(x))

        # Adjust the frames to ensure there are exactly 30 frames
        num_frames = len(frame_files)
        if num_frames > 30:
            start = num_frames // 2 - 15
            frame_files = frame_files[start:start + 30]
        elif num_frames < 30:
            frame_files += ['padding'] * (30 - num_frames)

        # Load and preprocess each frame
        frames = []
        for frame_file in frame_files:
            if frame_file == 'padding':
                frames.append(padding_img)
            else:
                frames.append(transform(Image.open(frame_file)))

        # Stack the frames to form a video clip and add an additional dimension to represent the batch size
        clip = torch.stack(frames).unsqueeze(0).to(device)

        # Convert action_dir to a tensor and send it to the same device as the clip
        action_name = os.path.basename(action_dir).rstrip('0123456789')
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
