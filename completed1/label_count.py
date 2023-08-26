import pandas as pd

# Read the CSV file
csv_path = "/home/zheng/VATN/completed1/total.csv"
df = pd.read_csv(csv_path)

# Count the occurrences of each label
label_counts = df['label'].value_counts()

# Save the counts to a txt file
with open("/home/zheng/VATN/completed1/label_count.txt", "w") as file:
    for label, count in label_counts.items():
        file.write(f"{label}: {count}\n")
