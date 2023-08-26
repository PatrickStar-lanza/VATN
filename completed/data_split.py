import os
import random
import json
from collections import defaultdict

def split_data_for_action(files):
    total_files = len(files)
    
    if total_files == 1:
        return files, [], []
    elif total_files == 2:
        return [files[0]], [files[1]], []
    else:
        train_count = int(total_files * 0.7)
        val_count = int(total_files * 0.15)
        
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]
        
        # Ensure at least one file in val if there's any left
        if len(test_files) > 0 and len(val_files) == 0:
            val_files = [test_files.pop(0)]

        return train_files, val_files, test_files

root_dir = "/home/zheng/VATN/action_pkl_completed"

if not os.path.exists(root_dir):
    print(f"Directory {root_dir} does not exist.")
else:
    print(f"Directory {root_dir} exists.")

all_files_dirs = os.listdir(root_dir)
print(f"All files and directories in {root_dir}: {all_files_dirs}")

action_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pkl.gz')]
print(action_files)
random.shuffle(action_files)

action_names = [os.path.basename(action_file).split('.')[0].rstrip('0123456789') for action_file in action_files] 
unique_action_names = list(set(action_names))  

data_split = defaultdict(lambda: defaultdict(list))

for action_name in unique_action_names:
    specific_action_files = [f for f in action_files if os.path.basename(f).split('.')[0].rstrip('0123456789') == action_name]
    train_files, val_files, test_files = split_data_for_action(specific_action_files)

    data_split['train'][action_name].extend(train_files)
    data_split['val'][action_name].extend(val_files)
    data_split['test'][action_name].extend(test_files)

    print(f"Action: {action_name} - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

for split, actions in data_split.items():
    # Filter out actions that have no files
    valid_actions = [action for action, files in actions.items() if files]
    print(f"{split} contains {len(valid_actions)} classes: {valid_actions}")

# Saving the results to the specified new directory
output_path = '/home/zheng/VATN/completed/data_split.json'
with open(output_path, 'w') as f:
    json.dump(data_split, f)

print(f"Saved results to {output_path}")
