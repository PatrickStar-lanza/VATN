# -*- coding: utf-8 -*-
import os
import re


from PIL import Image
from torchvision import transforms
import os
import torch
from transformer_v3_1 import Semi_Transformer
from torch import nn
import torch.optim as optim
import statistics


# Your parent directory
parent_dir = 'clip_frame'

# Get all directories under parent_dir
dirs = [os.path.join(parent_dir, dir) for dir in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, dir))]

# Initialize a list to store the number of numeric jpg files in each directory
num_files = []

# Loop through directories
for dir in dirs:
    action_dirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    # Loop through subdirectories
    for action_dir in action_dirs:
        frame_files = [f for f in os.listdir(action_dir) if f.endswith('.jpg')]
        numeric_frame_files = [f for f in frame_files if os.path.splitext(f)[0].isdigit()]

        num_files.append(len(numeric_frame_files))

# Calculate max, mean, and median
max_num_files = max(num_files)
mean_num_files = statistics.mean(num_files)
median_num_files = statistics.median(num_files)

print(f"Max number of numeric jpg files: {max_num_files}")
print(f"Mean number of numeric jpg files: {mean_num_files}")
print(f"Median number of numeric jpg files: {median_num_files}")





# 图像转换操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 重新调整图像大小，ResNet50的输入大小通常是224x224
    transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对应RGB三个通道，将其归一化至[-1,1]
])

device = torch.device("cuda")

# 获取所有的视频文件夹
video_dirs = [os.path.join("/home/zheng/VATN/clip_frame", d) for d in
              os.listdir("/home/zheng/VATN/clip_frame")]
print(f'There are total {len(video_dirs)} files in clip_frame, they are {[d for d in os.listdir("/home/zheng/VATN/clip_frame")]} ')
# Function to split the action name by the first uppercase letter
def split_by_uppercase(action):
    for i, c in enumerate(action):
        if c.isupper():
            return action[:i]
    return action  # Return the whole string if there is no uppercase letter

# Get a list of all unique action classes
all_actions = []
for video_dir in video_dirs:
    action_dirs = [d for d in os.listdir(video_dir)]
    action_dirs = [split_by_uppercase(d) for d in action_dirs]  # Only keep the part before the first uppercase letter
    all_actions.extend(action_dirs)

# Remove duplicates
all_actions = list(set(all_actions))
print(f"all actions are:{all_actions}, and their number is {len(all_actions)}")
# Create a mapping from actions to integers
action_to_int = {action: i for i, action in enumerate(all_actions)}

# Now you can use the length of all_actions as num_classes
num_classes = len(all_actions)

# Initialize the model
model = Semi_Transformer(num_classes=num_classes, seq_len=9).to(device)  # Adjust the seq_len according to your need

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training code
model.train()  # Set the model to training mode

for epoch in range(10):  # Number of epochs
    running_loss = 0.0

    # Getting each video in the dataset
    for video_dir in video_dirs:
        action_dirs = [os.path.join(video_dir, d) for d in os.listdir(video_dir)]

        for action_dir in action_dirs:
            action = os.path.basename(action_dir)  # Get the action class from the directory name
            action = split_by_uppercase(action)  # Only keep the part before the first uppercase letter
            label = torch.tensor([action_to_int[action]], dtype=torch.long, device=device)  # Convert action to integer label

            frame_files = [os.path.join(action_dir, f) for f in os.listdir(action_dir)]
            frame_files = [f for f in frame_files if os.path.splitext(os.path.basename(f))[0].isdigit()]
            frame_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

            frames = [transform(Image.open(frame_file)) for frame_file in frame_files]
            clip = torch.stack(frames).unsqueeze(0).to(device)  # clip shape: (1, T, C, H, W)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(clip)  # Forward pass
            loss = criterion(outputs, label)  # Calculate loss
            loss.backward()  # Perform backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

    # Print statistics
    print(f'Epoch {epoch+1}, Loss: {running_loss}')

print('Finished Training')
