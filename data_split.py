import os
import random
import json
from collections import defaultdict

root_dir = "/home/zheng/VATN/action_pkl"

if not os.path.exists(root_dir):
    print(f"Directory {root_dir} does not exist.")
else:
    print(f"Directory {root_dir} exists.")

all_files_dirs = os.listdir(root_dir)
print(f"All files and directories in {root_dir}: {all_files_dirs}")

# Now it's action_files, not action_dirs, and it's looking for .pt files, not directories
action_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pkl')]
print(action_files)
random.shuffle(action_files)  # Randomize the order of action files

# Get action names from files
action_names = [os.path.basename(action_file).split('.')[0].rstrip('0123456789') for action_file in action_files]  # Remove digits from the end of each action name
unique_action_names = list(set(action_names))  # Remove duplicates from the list

# Prepare data split
data_split = defaultdict(lambda: defaultdict(list))

# Ensure at least one instance of each action in the training set
for action_name in unique_action_names:
    # Find the first occurrence of this action
    for action_file in action_files[:]:
        if os.path.basename(action_file).split('.')[0].startswith(action_name):
            data_split['train'][action_name].append(action_file)
            action_files.remove(action_file)  # Remove this file from the action_files list
            break  # Move on to the next unique action name

# Randomly distribute remaining action files across train, val, and test
for action_file in action_files:
    action_name = os.path.basename(action_file).split('.')[0].rstrip('0123456789')
    split = random.choices(['train', 'val', 'test'], weights=[0.7, 0.15, 0.15])[0]  # Modify the weights as necessary
    data_split[split][action_name].append(action_file)

# Save the data split
with open('data_split.json', 'w') as f:
    json.dump(data_split, f)