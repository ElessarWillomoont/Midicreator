import json
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm

class MusicDataset(Dataset):
    def __init__(self, file_pattern):
        self.data_pairs = []
        # 使用glob.glob找到所有匹配的文件
        file_list = glob(file_pattern)
        for file_path in tqdm(file_list, desc="Loading data files"):
            with open(file_path, 'r') as f:
                for line in f:
                    input_ids, target_ids = json.loads(line)  # Load each line as a separate json object
                    self.data_pairs.append((input_ids, target_ids))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        input_ids, target_ids = self.data_pairs[idx]
        # Convert lists to tensors
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def get_data_loader(file_pattern, batch_size=96, num_workers=4):
    dataset = MusicDataset(file_pattern)
    # Ensure shuffle is True to mix up the dataset for better training
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
