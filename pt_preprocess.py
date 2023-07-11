import os
import torch
from PIL import Image
from torchvision import transforms

# Image transformation operations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get the directories of all the actions
action_dirs = [os.path.join("/home/zheng/VATN/collective_action", d) for d in os.listdir("/home/zheng/VATN/collective_action")]

# Define the placeholder image for padding
padding_img = torch.zeros(3, 224, 224)

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
    clip = torch.stack(frames).unsqueeze(0)

    # Save the tensor as a .pt file
    output_file = os.path.join("/home/zheng/VATN/action_pt", os.path.basename(action_dir) + ".pt")
    torch.save(clip, output_file)
