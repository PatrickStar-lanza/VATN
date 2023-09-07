import pandas as pd
from sklearn.model_selection import train_test_split

def count_labels(df, name):
    label_counts = df['label'].value_counts()
    print(f"Label counts in {name}:")
    print(label_counts)
    print(f"Total unique labels in {name}: {len(label_counts)}")

data = pd.read_csv('/home/zheng/VATN/completed1/total.csv')
value_counts = data['label'].value_counts()

# Remove labels with less than 3 samples
to_remove = value_counts[value_counts < 3].index
data = data[~data['label'].isin(to_remove)]

final_train = pd.DataFrame()
final_valid = pd.DataFrame()
final_test = pd.DataFrame()

for label, count in value_counts.items():
    if label not in to_remove:
        temp = data[data['label'] == label]
        
        if count < 10:
            # If count is less than 10, allocate one sample each to validation and test sets
            train, temp = train_test_split(temp, test_size=2, random_state=42)
            valid, test = train_test_split(temp, test_size=1, random_state=42)
        else:
            # Split in the ratio of 8:1:1
            train, temp = train_test_split(temp, test_size=0.2, random_state=42)
            valid, test = train_test_split(temp, test_size=0.5, random_state=42)
            
        final_train = pd.concat([final_train, train], ignore_index=True)
        final_valid = pd.concat([final_valid, valid], ignore_index=True)
        final_test = pd.concat([final_test, test], ignore_index=True)

# Save to CSV files
final_train.to_csv('/home/zheng/VATN/completed1/train.csv', index=False)
final_valid.to_csv('/home/zheng/VATN/completed1/valid.csv', index=False)
final_test.to_csv('/home/zheng/VATN/completed1/test.csv', index=False)

# Reload and count labels
final_train_reloaded = pd.read_csv('/home/zheng/VATN/completed1/train.csv')
final_valid_reloaded = pd.read_csv('/home/zheng/VATN/completed1/valid.csv')
final_test_reloaded = pd.read_csv('/home/zheng/VATN/completed1/test.csv')

count_labels(final_train_reloaded, 'train.csv')
count_labels(final_valid_reloaded, 'valid.csv')
count_labels(final_test_reloaded, 'test.csv')
