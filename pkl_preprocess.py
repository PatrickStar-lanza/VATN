import os
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import pickle

# Image transformation operations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda")

# Define the placeholder image for padding
padding_img = torch.zeros(3, 224, 224)

# Get the directories of all the actions
action_dirs = [os.path.join("/home/zheng/VATN/collective_action", d) for d in os.listdir("/home/zheng/VATN/collective_action")]

for action_dir in action_dirs:
    # Print the action directory name
    #print(f"Action directory: {action_dir}")

    files_in_dir = os.listdir(action_dir)
    #print(f"Files in directory: {files_in_dir}")  # Print all the files in the directory

    # Get all frames within the action directory
    frame_files = []
    for f in files_in_dir:
        #print(f"Checking file: {f}")  # Print the file being checked
        if os.path.splitext(f)[0].isdigit():
            #print(f"File name {f} passed isdigit check.")  # Print the file name that passed the check
            frame_files.append(os.path.join(action_dir, f))
        #else:
            #print(f"File name {f} failed isdigit check.")  # Print the file name that failed the check

    frame_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Adjust the frames to ensure there are exactly 30 frames
    num_frames = len(frame_files)
    if num_frames > 30:
        start = num_frames // 2 - 15
        frame_files = frame_files[start:start + 30]
        padding_start = []
        padding_end = []
    elif num_frames < 30:
        padding_start = [padding_img] * ((30 - num_frames) // 2)
        padding_end = [padding_img] * ((30 - num_frames) - len(padding_start))
        frame_files = padding_start + frame_files + padding_end
    else:
        padding_start = []
        padding_end = []

    # Load and preprocess each frame
    frames = []
    for frame_file in frame_files:
        if isinstance(frame_file, torch.Tensor):
            frames.append(frame_file)
        else:
            frames.append(transform(Image.open(frame_file)))

    frames = torch.stack(frames)  # Stack the frames to form a video clip

    padding_count = len(padding_start) + len(padding_end)
    non_padding_count = len(frame_files) - padding_count

    # Print results
    #print(f'Padded frames: {padding_count}')
    #print(f'Non-padded frames: {non_padding_count}')

    # Print non-zero parts
    #non_padding_frames = frames[len(padding_start):len(padding_start) + non_padding_count]

    #print('Non-padding frames:')
    #print(non_padding_frames)

    # Save the clip to a pickle file
    pkl_file = os.path.join("/home/zheng/VATN/action_pkl", os.path.basename(action_dir) + '.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(frames, f)
