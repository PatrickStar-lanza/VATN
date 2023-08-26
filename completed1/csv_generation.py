import os
import pandas as pd

# Define target directory and output path
directory_path = "/home/zheng/VATN/action_pkl_completed/"
output_path = "/home/zheng/VATN/completed1/total.csv"

# Get list of files in the directory
file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# Prepare data for ID and label columns
data = {
    "ID": [],
    "label": []
}

for filename in file_list:
    # Only use the filename for the ID column
    data["ID"].append(filename)
    
    # Remove suffixes .gz and .pkl
    label = filename.replace('.gz', '').replace('.pkl', '')
    
    # Remove all digits from the filename
    label = ''.join([i for i in label if not i.isdigit()])
    
    # Add the processed filename to label column
    data["label"].append(label)

# Use pandas to save the data as CSV
df = pd.DataFrame(data)
df.to_csv(output_path, index=False)
