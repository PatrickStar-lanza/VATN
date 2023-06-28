import os
import re
import matplotlib.pyplot as plt

# path to the collective_action folder
path = '/home/zheng/VATN/collective_action/'

# initialize a set to store action names
action_names = set()

# initialize a list to store number of images in each subdirectory
image_counts = []

# traverse the directories
for subdir in os.listdir(path):
    subdir_path = os.path.join(path, subdir)
    if os.path.isdir(subdir_path):
        # get the lowercase part of the subdir name
        lowercase_part = re.match('^[a-z]*', subdir).group()
        action_names.add(lowercase_part)

        # count the number of images in this subdir
        image_count = len([f for f in os.listdir(subdir_path) if f.endswith('.jpg') or f.endswith('.png')])
        image_counts.append(image_count)

print(f'There are {len(action_names)} kinds of actions. They are:')
print(' '.join(action_names))




import numpy as np

# get the list of action names and counts
action_names_list = list(action_names)
action_counts = [image_counts[i] for i in range(len(action_names_list))]

# sort them by counts
sorted_actions, sorted_counts = zip(*sorted(zip(action_names_list, action_counts), key=lambda x: x[1]))

# draw a bar plot
plt.barh(sorted_actions, sorted_counts)
plt.xlabel('Number of Images')
plt.title('Number of Images for Each Action')
plt.show()
