import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from model_transformer import GPT2LikeTransformer  # 确保这个路径正确
from torch.nn.utils.rnn import pad_sequence

# 数据集目录
dataset_dir = Path('dataset/dataset_json')

# 数据集类
class MIDIDataset(Dataset):
    def __init__(self, json_files):
        self.json_files = json_files

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # 从JSON文件中加载token ids
        with open(self.json_files[idx], 'r') as f:
            data = json.load(f)
        input_ids = torch.tensor(data['input_ids'], dtype=torch.long)
        targets = torch.tensor(data['targets'], dtype=torch.long)
        return input_ids, targets

# 自定义collate_fn函数
def collate_fn(batch):
    input_ids, targets = zip(*batch)
    max_length = 256  # 定义最大序列长度

    # 对截断的序列进行填充
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    return input_ids_padded, targets_padded

# 损失函数
def compute_loss(outputs, targets):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是填充令牌的索引
    return loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

# 初始化模型和优化器
model = GPT2LikeTransformer(vocab_size=30000, n_layer=6, n_head=4, n_emb=16, context_length=256, pad_token_id=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 准备数据
train_json_files = list((dataset_dir / 'train_set').glob('*.json'))
val_json_files = list((dataset_dir / 'validation_set').glob('*.json'))

# 创建DataLoader
train_dataset = MIDIDataset(train_json_files)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

val_dataset = MIDIDataset(val_json_files)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 训练循环
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    for input_ids, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()

# 验证循环（可选）
model.eval()
with torch.no_grad():
    total_loss = 0
    for input_ids, targets in tqdm(val_loader):
        outputs = model(input_ids)
        loss = compute_loss(outputs, targets)
        total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(val_loader)}")
