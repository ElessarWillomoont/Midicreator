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
                
                for line in lines:
                    music_data = json.loads(line)
                    token_ids = music_data['ids'][0]  # Assuming we only process the first element
                    
                    # Cut the token IDs into 256-token windows
                    for i in range(0, len(token_ids) - MAX_LENGTH, MAX_LENGTH):
                        input_ids = token_ids[i:i+MAX_LENGTH-1]  # Input is the first 255 tokens
                        target_ids = token_ids[i+1:i+MAX_LENGTH]  # Target is the next 255 tokens (shifted by one)
                        
                        # Pad or truncate input_ids and target_ids to MAX_LENGTH
                        input_ids = (input_ids + [0] * (MAX_LENGTH - len(input_ids)))[:MAX_LENGTH]
                        target_ids = (target_ids + [0] * (MAX_LENGTH - len(target_ids)))[:MAX_LENGTH]
                        
                        data_pairs.append((input_ids, target_ids))
    
    with open(output_file, 'w') as f:
        json.dump(data_pairs, f)

preprocess_data('dataset/dataset_json/train_set', 'dataset/train_data.json')
preprocess_data('dataset/dataset_json/validation_set', 'dataset/validation_data.json')
