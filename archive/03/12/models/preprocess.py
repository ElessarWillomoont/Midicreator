import json
import os

MAX_LENGTH = 256  # Global variable for maximum length

def preprocess_data(input_dir, output_file):
    data_pairs = []
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                for i in range(len(lines) - 1):
                    current_music_data = json.loads(lines[i])
                    next_music_data = json.loads(lines[i + 1])
                    current_ids = current_music_data['ids'][0]  # Assuming we only process the first element
                    next_ids = next_music_data['ids'][0]
                    input_ids = [1] + current_ids + [2]  # Adding BOS and EOS for current line
                    target_ids = [1] + next_ids + [2]  # Adding BOS and EOS for next line
                    # Truncate or pad input_ids and target_ids to MAX_LENGTH
                    input_ids = (input_ids + [0] * MAX_LENGTH)[:MAX_LENGTH]
                    target_ids = (target_ids + [0] * MAX_LENGTH)[:MAX_LENGTH]
                    data_pairs.append((input_ids, target_ids))
    
    with open(output_file, 'w') as f:
        json.dump(data_pairs, f)

preprocess_data('dataset/dataset_json/train_set', 'dataset/train_data.json')
preprocess_data('dataset/dataset_json/validation_set', 'dataset/validation_data.json')
