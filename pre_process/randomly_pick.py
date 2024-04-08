from pathlib import Path
import shutil
import random
import math
import sys
import os

# Setting up the environment
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(script_path))
sys.path.append(parent_directory)
import shared.config as configue

PICK_RATIO = configue.PICK_RATIO  # Define the pick ratio from the configuration

# Define directories
dataset_dir = Path("dataset/dataset_json")
unused_dir = dataset_dir / "unused"
train_set_dir = unused_dir / "train_set"
validation_set_dir = unused_dir / "validation_set"
eval_set_dir = unused_dir / "eval_set"

final_train_set_dir = dataset_dir / "train_set"
final_validation_set_dir = dataset_dir / "validation_set"
final_eval_set_dir = dataset_dir / "eval_set"

# Create directories if they do not exist
unused_dir.mkdir(parents=True, exist_ok=True)
train_set_dir.mkdir(parents=True, exist_ok=True)
validation_set_dir.mkdir(parents=True, exist_ok=True)
eval_set_dir.mkdir(parents=True, exist_ok=True)

final_train_set_dir.mkdir(parents=True, exist_ok=True)
final_validation_set_dir.mkdir(parents=True, exist_ok=True)
final_eval_set_dir.mkdir(parents=True, exist_ok=True)

# Get a list of all JSON files
json_files = list(dataset_dir.glob("*.json"))

# Shuffle the file list randomly
random.shuffle(json_files)

# Define the distribution ratios
train_ratio, validation_ratio, eval_ratio = configue.SIZE_OF_TRAIN, configue.SIZE_OF_VAL, configue.SIZE_OF_EVAL

# Distribute files to the unused directory
num_train_files = math.floor(len(json_files) * train_ratio)
num_validation_files = math.floor(len(json_files) * validation_ratio)

train_files = json_files[:num_train_files]
validation_files = json_files[num_train_files:num_train_files + num_validation_files]
eval_files = json_files[num_train_files + num_validation_files:]

# Move files to their respective unused directories
for file in train_files:
    shutil.move(str(file), str(train_set_dir / file.name))
for file in validation_files:
    shutil.move(str(file), str(validation_set_dir / file.name))
for file in eval_files:
    shutil.move(str(file), str(eval_set_dir / file.name))

# Function to pick files from the unused directory
def pick_files(source_dir, target_dir, ratio):
    """Pick a specified ratio of files from the source directory and move them to the target directory."""
    all_files = list(source_dir.glob("*.json"))
    num_pick_files = math.ceil(len(all_files) * ratio)
    picked_files = random.sample(all_files, num_pick_files)
    for file in picked_files:
        shutil.move(str(file), str(target_dir / file.name))

# Pick files from the unused directory to the final dataset_json directory
pick_files(train_set_dir, final_train_set_dir, PICK_RATIO)
pick_files(validation_set_dir, final_validation_set_dir, PICK_RATIO)
pick_files(eval_set_dir, final_eval_set_dir, PICK_RATIO)