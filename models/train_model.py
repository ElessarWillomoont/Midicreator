# train_model.py
import torch
from model_transformer import TransformerModel  # 假设你的模型定义在这个文件中
from data_loder import get_data_loader

def train_model(train_data_loader, validation_data_loader, model, epochs=1):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for input_ids, target_ids in train_data_loader:
            optimizer.zero_grad()
            output = model(input_ids)
            output = output.permute(0, 2, 1)
            loss = loss_fn(output, target_ids)
            loss.backward()
            optimizer.step()
            output = output.permute(0, 2, 1)
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
train_data_loader = get_data_loader('dataset/train_data.json')
validation_data_loader = get_data_loader('dataset/validation_data.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = TransformerModel(vocab_size=30000, n_layer=3, n_head=4, n_emb=16, context_length=256, pad_token_id=0)  # 假设你的模型构造函数不需要任何参数

# 训练模型
train_model(train_data_loader, validation_data_loader, model)
