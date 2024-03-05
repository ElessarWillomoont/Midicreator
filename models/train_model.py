# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.model_transformer import TransformerModel  # 假设你的模型定义在这个文件中
from models.data_loder import get_data_loader

def train_model(train_data_loader, validation_data_loader, model, epochs=1):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for input_ids, target_ids in train_data_loader:
            optimizer.zero_grad()
            output = model(input_ids)
            loss = loss_fn(output, target_ids)
            loss.backward()
            optimizer.step()
        
        # 简化的验证过程
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for input_ids, target_ids in validation_data_loader:
                output = model(input_ids)
                loss = loss_fn(output, target_ids)
                total_loss += loss.item()
            print(f'Epoch {epoch}, Validation Loss: {total_loss / len(validation_data_loader)}')

def compute_loss(outputs, targets):
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 假设0是填充令牌的索引
    return loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

# 加载数据
train_data_loader = get_data_loader('train_data.json')
validation_data_loader = get_data_loader('validation_data.json')

# 初始化模型
model = TransformerModel(vocab_size=30000, n_layer=6, n_head=4, n_emb=16, context_length=256, pad_token_id=0)  # 假设你的模型构造函数不需要任何参数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
model.train()
num_epochs = 1
for epoch in range(num_epochs):
    for input_ids, targets in tqdm(train_data_loader):
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()

# 验证循环（可选）
model.eval()
with torch.no_grad():
    total_loss = 0
    for input_ids, targets in tqdm(validation_data_loader):
        outputs = model(input_ids)
        loss = compute_loss(outputs, targets)
        total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(validation_data_loader)}")
