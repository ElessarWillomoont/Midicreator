import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
import torch.nn as nn
from miditoolkit import MidiFile
from miditok import REMI, MIDITokenizer
from tqdm import tqdm
from model_transformer import GPT2LikeTransformer  # 确保这个路径正确
from torch.nn.utils.rnn import pad_sequence

# 加载tokenizer
tokenizer_path = Path('tokenizer/tokenizer.json')
tokenizer = REMI(params=tokenizer_path)

# 读取MIDI文件并按源文件名分组
splited_midi_dir = Path('dataset/splited')
midi_files = list(splited_midi_dir.glob('*.midi'))
midi_groups = {}
for midi_file in midi_files:
    base_name = midi_file.stem.split('_bar_')[0]
    if base_name not in midi_groups:
        midi_groups[base_name] = []
    midi_groups[base_name].append(midi_file)

# 将组划分为训练组和验证组
group_names = list(midi_groups.keys())
train_group_names, val_group_names = train_test_split(group_names, test_size=0.05, random_state=42)

# 保存验证集组名
with open('validation.json', 'w') as f:
    json.dump(val_group_names, f)

# 创建训练对和验证对
def create_pairs(groups):
    pairs = []
    for group_name in groups:
        sorted_group = sorted(midi_groups[group_name], key=lambda x: x.stem)
        for i in range(len(sorted_group) - 1):
            pairs.append((sorted_group[i], sorted_group[i+1]))
    return pairs

train_pairs = create_pairs(train_group_names)
val_pairs = create_pairs(val_group_names)

# MIDI处理函数
def process_midi_pair(midi_pair):
    midi_data_1 = MidiFile(midi_pair[0])
    midi_data_2 = MidiFile(midi_pair[1])
    tokens_1 = tokenizer(midi_data_1)
    tokens_2 = tokenizer(midi_data_2)
    input_ids = torch.tensor(tokens_1, dtype=torch.long)
    targets = torch.tensor(tokens_2, dtype=torch.long)
    return input_ids, targets

# 数据集类
class MIDIDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return process_midi_pair(self.pairs[idx])

# 自定义collate_fn函数
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids, targets = zip(*batch)
    max_length = 256  # Define your maximum sequence length here

    # Ensure all sequences are truncated to the same length
    input_ids_truncated = [seq[:max_length] for seq in input_ids]
    targets_truncated = [seq[:max_length] for seq in targets]

    # Pad sequences
    input_ids_padded = pad_sequence(input_ids_truncated, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets_truncated, batch_first=True, padding_value=0)

    return input_ids_padded, targets_padded



# 损失函数
def compute_loss(outputs, targets):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是填充令牌的索引
    return loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

# 初始化模型和优化器
model = GPT2LikeTransformer(vocab_size=30000, n_layer=6, n_head=4, n_emb=16, context_length=256, pad_token_id=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建DataLoader
train_dataset = MIDIDataset(train_pairs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 训练循环
model.train()
num_epochs = 3
for epoch in range(num_epochs):
    for input_ids, targets in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
