# train_model.py
import torch
from model_transformer import TransformerModel  # 假设你的模型定义在这个文件中
from data_loder import get_data_loader
from tqdm import tqdm
import wandb
import time

PROJECT_NAME = 'Midicreator'
ENTITY_NAME = 'candle2587_team'

def train_model(device, train_data_loader, validation_data_loader, model, epochs=1):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs), desc="Epochs Progress"):  # Track progress of epochs
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0

        # Wrap train_data_loader with tqdm for training progress bar
        for input_ids, target_ids in tqdm(train_data_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            output = model(input_ids)
            output = output.permute(0, 2, 1)
            loss = loss_fn(output, target_ids)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()  # Accumulate the training loss
            #wandb.log({"train_loss": loss.item()})
        avg_train_loss = total_train_loss / len(train_data_loader)
        wandb.log({"avg_train_loss": avg_train_loss})
        

        # Validation process with tqdm progress bar
        model.eval()
        total_val_loss = 0
        # Wrap validation_data_loader with tqdm for validation progress bar
        for input_ids, target_ids in tqdm(validation_data_loader, desc=f"Validating Epoch {epoch+1}"):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                output = model(input_ids)
                output = output.permute(0, 2, 1)
                loss = loss_fn(output, target_ids)
                output = output.permute(0, 2, 1)
                total_val_loss += loss.item()
                #wandb.log({"validation loss": loss.item()})
        avg_val_loss = total_val_loss / len(validation_data_loader)
        wandb.log({"avg_validation_loss": avg_val_loss})

        # Print training and validation loss
        print(f'Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - total_start_time
        print(f'Epoch {epoch} completed in {epoch_duration:.2f} seconds.')
        print(f'Total training time up to now: {total_duration:.2f} seconds.')
#Wandb
wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME)

# 加载数据
train_data_loader = get_data_loader('dataset/train_data.json')
validation_data_loader = get_data_loader('dataset/validation_data.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_start_time = time.time()

# 初始化模型
model = TransformerModel(vocab_size=30000, n_layer=3, n_head=4, n_emb=16, context_length=256, pad_token_id=0)  # 假设你的模型构造函数不需要任何参数
model.to(device)
wandb.watch(model, log='all')
# 训练模型
train_model(device ,train_data_loader, validation_data_loader, model)
wandb.finish()
