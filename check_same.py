import os
import pickle
import torch

def check_pkl_files(directory):
    pkl_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pkl')]

    if not pkl_files:
        print("No .pkl files found.")
        return

    with open(pkl_files[0], 'rb') as f:
        reference_content = pickle.load(f)

    all_same = True
    for file in pkl_files[1:]:
        with open(file, 'rb') as f:
            current_content = pickle.load(f)
        if not torch.equal(current_content, reference_content):
            all_same = False
            break

    if all_same:
        print("All .pkl files have the same content.")
    else:
        print("Not all .pkl files have the same content.")

directory = "/home/zheng/VATN/action_pkl/"
check_pkl_files(directory)
