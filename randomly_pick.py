import os
import random
import shutil

# Define the paths
dataset_dir = "dataset/dataset_json"
train_set_dir = "dataset/dataset_json/train_set"
validation_set_dir = "dataset/dataset_json/validation_set"

# Create the train and validation set directories if they don't exist
os.makedirs(train_set_dir, exist_ok=True)
os.makedirs(validation_set_dir, exist_ok=True)

# Get a list of all JSON files in the dataset directory and its subfolders
json_files = [os.path.join(root, file) for root, dirs, files in os.walk(dataset_dir) for file in files if file.endswith('.json')]

# Calculate 5% of the total number of JSON files for the validation set
num_validation_files = int(0.05 * len(json_files))

# Randomly select files for the validation set
validation_files = random.sample(json_files, num_validation_files)

# Move the selected validation files to the validation set directory
for file in validation_files:
    shutil.move(file, os.path.join(validation_set_dir, os.path.basename(file)))

# Move the remaining files to the train set directory
for file in json_files:
    if file not in validation_files:
        shutil.move(file, os.path.join(train_set_dir, os.path.basename(file)))

print("Dataset split into train and validation sets successfully.")
