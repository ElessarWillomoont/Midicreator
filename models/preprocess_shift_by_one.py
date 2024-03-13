import json
import os
from tqdm import tqdm

MAX_LENGTH = 256  # Global variable for maximum length

def preprocess_data(input_dir, output_file, batch_size=8):
    all_files = [os.path.join(subdir, file) for subdir, _, files in os.walk(input_dir) for file in files if file.endswith('.json')]
    total_files = len(all_files)
    batches = [all_files[i:i + batch_size] for i in range(0, total_files, batch_size)]

    with open(output_file, 'w') as f_out:
        for batch_files in tqdm(batches, desc="Processing Batches"):
            data_pairs = []  # Initialize data_pairs for each batch

            for file_path in batch_files:
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    music_data = json.loads(line)
                    token_ids = music_data['ids'][0]

                    for i in range(0, len(token_ids) - MAX_LENGTH + 1):
                        input_ids = token_ids[i:i+MAX_LENGTH-1]
                        target_ids = token_ids[i+1:i+MAX_LENGTH]

                        input_ids = (input_ids + [0] * (MAX_LENGTH - 1 - len(input_ids)))[:MAX_LENGTH-1]
                        target_ids = (target_ids + [0] * (MAX_LENGTH - 1 - len(target_ids)))[:MAX_LENGTH-1]

                        data_pairs.append((input_ids, target_ids))

            # Write the current batch to the output file
            for pair in data_pairs:
                f_out.write(json.dumps(pair) + '\n')  # Write each pair as a new line

            # Optionally clear variables to help with memory management
            del data_pairs

preprocess_data('dataset/dataset_json/train_set', 'dataset/train_data.json')
preprocess_data('dataset/dataset_json/validation_set', 'dataset/validation_data.json')
