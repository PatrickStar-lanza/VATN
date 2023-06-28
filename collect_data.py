import os
import shutil
import re

source_dir = '/home/zheng/VATN/clip_frame/'
dest_dir = '/home/zheng/VATN/collective_action/'

# create the destination directory if not exists
os.makedirs(dest_dir, exist_ok=True)

# initialize a dictionary to count the number of each directory name
dir_counter = {}

for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path):
        for subsubdir in os.listdir(subdir_path):
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            if os.path.isdir(subsubdir_path):
                # get the lowercase part of the subsubdir name
                lowercase_part = re.match('^[a-z]*', subsubdir).group()

                # count the number of this name
                if lowercase_part not in dir_counter:
                    dir_counter[lowercase_part] = 1
                else:
                    dir_counter[lowercase_part] += 1

                # create a new name by adding the count to the end of the name
                new_name = lowercase_part + str(dir_counter[lowercase_part])

                # copy the directory
                shutil.copytree(subsubdir_path, os.path.join(dest_dir, new_name))
