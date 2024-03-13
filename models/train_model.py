# train_model.py
import torch
from model_transformer import TransformerModel  # 假设你的模型定义在这个文件中
from data_loder import get_data_loader
from tqdm import tqdm
import os
import glob
import wandb
import time

PROJECT_NAME = 'Midicreator'
ENTITY_NAME = 'candle2587_team'
EPOCH_NUM = 1000

# Ensure checkpoint directory exists
checkpoint_dir = "model_output/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def train_model(device, train_data_loader, validation_data_loader, model, epochs=EPOCH_NUM):
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    # Load the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {latest_checkpoint}")
    else:
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs Progress"):
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
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
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
