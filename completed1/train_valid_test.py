import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/home/zheng/VATN/completed1/total.csv')

# Count the number of samples for each label
value_counts = data['label'].value_counts()

# Identify labels with fewer than 2 samples
to_remove = value_counts[value_counts < 2].index
data_removed = data[data['label'].isin(to_remove)]

print("Removed labels:")
for label in to_remove:
    print(label)

# Save the removed samples to a CSV file
data_removed.to_csv('/home/zheng/VATN/completed1/removed_samples.csv', index=False)

# Exclude the removed labels from the main data
data = data[~data['label'].isin(to_remove)]

# Split 70% of the data for training
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)

# If any labels have only 1 sample in temp_data, then split without stratification, else stratify
if any(temp_data['label'].value_counts() < 2):
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
else:
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Save the splits to CSV files
train_data.to_csv('/home/zheng/VATN/completed1/train.csv', index=False)
valid_data.to_csv('/home/zheng/VATN/completed1/valid.csv', index=False)
test_data.to_csv('/home/zheng/VATN/completed1/test.csv', index=False)
