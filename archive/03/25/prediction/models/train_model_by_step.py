# train_model.py
import torch
from model_transformer import TransformerModel  # 假设你的模型定义在这个文件中
from data_loder import get_data_loader
from tqdm import tqdm
import os
import glob
import wandb
import time
from adaptive_trainer import AdaptiveLRScheduler

PROJECT_NAME = 'Midicreator_Prediction'
ENTITY_NAME = 'candle2587_team'
EPOCH_NUM = 4000
STEP_SIZE = 2000  # 每多少步进行一次检查和存储检查点
BATCH_SIZE = 3500
LOAD_DATA_THREAD = 2

# Ensure checkpoint directory exists
checkpoint_dir = "model_output/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def train_model(device, train_data_loader, validation_data_loader, model, epochs=EPOCH_NUM):
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    #lr_scheduler = AdaptiveLRScheduler(optimizer)

    total_start_time = time.time()

    # Load the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)  # Load global step if available
        print(f"Loaded checkpoint from {latest_checkpoint}")
    else:
        start_epoch = 0
        global_step = 0

    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs Progress"):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0

        for input_ids, target_ids in tqdm(train_data_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            output = model(input_ids)
            output = output.permute(2, 0, 1)
            loss = loss_fn(output, target_ids)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            global_step += 1  # Increment global step

            # Check if it's time to save a checkpoint
            if global_step % STEP_SIZE == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
        avg_train_loss = total_train_loss / len(train_data_loader)
        wandb.log({"avg_train_loss": avg_train_loss, "global_step": global_step})
        

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
        avg_train_loss = total_train_loss / len(train_data_loader)

        #stop_training = lr_scheduler.step(avg_val_loss)
        #if stop_training:
        #    print("Stopping training due to possible overfitting.")
        #    break  # 停止训练

        # 记录当前学习率
        #current_lr = optimizer.param_groups[0]['lr']
        #current_lr = optimizer.param_groups[0]['lr']
        #wandb.log({"learning_rate": current_lr, "epoch": epoch})

        print(f'Epoch {epoch}, Training Loss: {avg_train_loss:.4f}')
        print(f'Epoch {epoch}, Validation Loss: {avg_val_loss:.4f}')
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_duration = epoch_end_time - total_start_time
        print(f'Epoch {epoch} completed in {epoch_duration:.2f} seconds.')
        print(f'Total training time up to now: {total_duration:.2f} seconds.')

#Wandb
wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME)

# 加载数据
train_data_pattern = 'train_data_*.json'  # 匹配所有训练数据文件
validation_data_pattern = 'validation_data_*.json'  # 匹配所有验证数据文件

train_data_loader = get_data_loader(train_data_pattern, BATCH_SIZE, LOAD_DATA_THREAD)
validation_data_loader = get_data_loader(validation_data_pattern, BATCH_SIZE, LOAD_DATA_THREAD)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
total_start_time = time.time()

# 初始化模型
model = TransformerModel(vocab_size=30000, encoder_layer = 1, decoder_layer = 6, n_head=2, n_emb=8, context_length=8, pad_token_id=0)  # context_length=256
model.to(device)
wandb.watch(model, log='all')
# 训练模型
train_model(device ,train_data_loader, validation_data_loader, model)
wandb.finish()