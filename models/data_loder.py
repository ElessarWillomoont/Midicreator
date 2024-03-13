import json
import torch
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):
    def __init__(self, data_file):
        self.data_pairs = []
        with open(data_file, 'r') as f:
            for line in f:
                input_ids, target_ids = json.loads(line)  # Load each line as a separate json object
                self.data_pairs.append((input_ids, target_ids))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_ids, target_ids = self.data_pairs[idx]
        # Convert lists to tensors
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def get_data_loader(data_file, batch_size, num_workers):
    dataset = MusicDataset(data_file)
    # Ensure shuffle is True to mix up the dataset for better training
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers = num_workers)
