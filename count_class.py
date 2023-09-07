import pandas as pd
import os
import matplotlib.pyplot as plt

def count_and_plot_labels(file_path):
    try:
        df = pd.read_csv(file_path)
        if "label" in df.columns:
            label_counts = df['label'].value_counts()
            
            print(f"In file {file_path.split('/')[-1]}:")
            print("Counts for each label:")
            print(label_counts)
            
            # Finding labels that appear less than 3 times
            rare_labels = label_counts[label_counts < 3]
            print("Labels that appear less than 3 times:")
            print(rare_labels)
            
            # Plotting the data
            plt.figure(figsize=(12, 6))
            label_counts.plot(kind='bar', color='skyblue')
            plt.title(f"Label distribution in {file_path.split('/')[-1]}")
            plt.xlabel("Labels")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()
            
        else:
            print(f"'label' column not found in {file_path.split('/')[-1]}")
    except FileNotFoundError:
        print(f"File {file_path.split('/')[-1]} not found")

# Define the directory and file names
directory_path = "/home/zheng/VATN/completed1/"
file_names = ['total.csv']

# Loop through the CSV files to count labels
for file_name in file_names:
    file_path = os.path.join(directory_path, file_name)
    count_and_plot_labels(file_path)
