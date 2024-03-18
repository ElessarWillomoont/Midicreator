import json
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.auto import tqdm

MAX_LENGTH = 8  # Maximum sequence length
FILE_SIZE = 500 * 1024 * 1024  # 500MB in bytes
BATCH_SIZE = 32
PAD_ID = 0  # Assuming 0 is the ID for PAD token

class MusicDataset(Dataset):
    def __init__(self, directory):
        self.token_lists = []
        self.load_data(directory)
        
    def load_data(self, directory):
        files = Path(directory).rglob('*.json')
        files = list(files)  # Convert to list to print the count
        print(f"Found {len(files)} files")  # Debugging line
        for file_path in tqdm(files, desc="Loading Data"):
            print(f"Reading file: {file_path}")  # Debugging line
            with open(file_path, 'r') as f:
                music_data = json.load(f)
                token_ids = music_data['ids'][0]
                self.token_lists.extend([token_ids[i:i+MAX_LENGTH] for i in range(len(token_ids) - MAX_LENGTH + 1)])

    def __len__(self):
        return len(self.token_lists)
    
    def __getitem__(self, idx):
        token_list = self.token_lists[idx]
        # Ensure that every token list is exactly MAX_LENGTH - 1
        padded_list = token_list + [PAD_ID] * (MAX_LENGTH - 1 - len(token_list))
        return {'input_ids': padded_list[:-1], 'labels': padded_list[1:]}

def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

def preprocess_data(input_dir, output_prefix, batch_size=8):
    dataset = MusicDataset(input_dir)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    file_index = 1
    output_file_path = Path(output_prefix)
    output_file_path.mkdir(parents=True, exist_ok=True)
    output_file = output_file_path / f"{output_prefix}_{file_index}.json"
    
    f_out = open(output_file, 'w')
    current_file_size = 0

    for batch in tqdm(loader, desc="Processing Batches"):
        for input_ids, labels in zip(batch['input_ids'], batch['labels']):
            data_pair = json.dumps((input_ids.tolist(), labels.tolist())) + '\n'
            if current_file_size + len(data_pair.encode('utf-8')) > FILE_SIZE:
                f_out.close()
                file_index += 1
                output_file = output_file_path / f"{output_prefix}_{file_index}.json"
                f_out = open(output_file, 'w')
                current_file_size = 0
            f_out.write(data_pair)
            current_file_size += len(data_pair.encode('utf-8'))
    
    f_out.close()