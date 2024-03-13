import json
import os
from tqdm import tqdm

MAX_LENGTH = 32  # Global variable for maximum length
FILE_SIZE = 500 * 1024 * 1024  # 500MB in bytes

def preprocess_data(input_dir, prefix, batch_size=8):
    all_files = [os.path.join(subdir, file) for subdir, _, files in os.walk(input_dir) for file in files if file.endswith('.json')]
    total_files = len(all_files)
    batches = [all_files[i:i + batch_size] for i in range(0, total_files, batch_size)]

    file_index = 1
    current_file_size = 0
    output_file = f"{prefix}_{file_index}.json"
    f_out = open(output_file, 'w')

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

        # Check file size and write to a new file if the current one is too large
        for pair in data_pairs:
            pair_str = json.dumps(pair) + '\n'
            if current_file_size + len(pair_str.encode('utf-8')) > FILE_SIZE:
                f_out.close()
                file_index += 1
                output_file = f"{prefix}_{file_index}.json"
                f_out = open(output_file, 'w')
                current_file_size = 0  # Reset file size counter
            f_out.write(pair_str)
            current_file_size += len(pair_str.encode('utf-8'))

        # Optionally clear variables to help with memory management
        del data_pairs

    f_out.close()  # Make sure to close the file when done

preprocess_data('dataset/dataset_json/train_set', 'dataset/train_data', batch_size=8)
preprocess_data('dataset/dataset_json/validation_set', 'dataset/validation_data', batch_size=8)
