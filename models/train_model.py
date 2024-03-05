# train_model.py
import torch
from model_transformer import TransformerModel  # 假设你的模型定义在这个文件中
from models.data_loader import get_data_loader

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

# 加载数据
train_data_loader = get_data_loader('train_data.json')
validation_data_loader = get_data_loader('validation_data.json')

# 初始化模型
model = TransformerModel()  # 假设你的模型构造函数不需要任何参数

# 训练模型
train_model(train_data_loader, validation_data_loader, model)