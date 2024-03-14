# data_loader.py
import json
import torch
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data_pairs = json.load(f)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_ids, target_ids = self.data_pairs[idx]
        return torch.tensor(input_ids), torch.tensor(target_ids)

def get_data_loader(data_file, batch_size=96):
    dataset = MusicDataset(data_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
