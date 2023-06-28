from PIL import Image
from torchvision import transforms
import os
import torch
from transformer_v3_1 import Semi_Transformer

# Image transformation operations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda")

# Get all video directories
video_dirs = [os.path.join("/home/zheng/VATN/clip_frame", d) for d in os.listdir("/home/zheng/VATN/clip_frame")]


# Iterate through each video directory
for video_dir in video_dirs:
    action_dirs = [os.path.join(video_dir, d) for d in os.listdir(video_dir)]
    # Get all action directories within the video directory
    action_dirs_clean = [d[:next((i for i, c in enumerate(d) if c.isupper()), None)] for d in os.listdir(video_dir)]
    action_dirs_clean = list(set(action_dirs_clean))
    print(f"In this file {video_dir}, there are {len(action_dirs_clean)} actions, they are {action_dirs_clean}")
    # Iterate through each action directory
    for action_dir in action_dirs:

        # Get all frames within the action directory
        frame_files = [os.path.join(action_dir, f) for f in os.listdir(action_dir)]

        # Keep only the files where the filename (without extension) is a number
        frame_files = [f for f in frame_files if os.path.splitext(os.path.basename(f))[0].isdigit()]
        frame_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Load and preprocess each frame
        frames = [transform(Image.open(frame_file)) for frame_file in frame_files]

        # Stack the frames to form a video clip and add an additional dimension to represent the batch size
        clip = torch.stack(frames).unsqueeze(0).to(device)
        print(f"clip.size:{clip.size()}")

        # Now you can input the clip to your model for processing
        model = Semi_Transformer(num_classes=len(action_dirs), seq_len=clip.shape[1]).to(device)
        output = model(clip)

    # After processing the first video directory, stop
    break
