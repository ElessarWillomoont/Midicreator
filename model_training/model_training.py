import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from shared.models import DecoderOnlyTransformer
from preprocess import MusicDataset, custom_collate_fn
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(script_path))
sys.path.append(parent_directory)
import shared.config as configue

# Configuration
PROJECT_NAME = configue.PROJECT_NAME
ENTITY_NAME = configue.ENTITY_NAME
EPOCH_NUM = configue.EPOCH_NUM
STEP_SIZE = configue.STEP_SIZE
BATCH_SIZE = configue.BATCH_SIZE
MAX_LENGTH = configue.MAX_LENGTH
PAD_ID = configue.PAD_ID
CHECK_POINT = configue.CHECK_POINT_CONTINUE

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DecoderOnlyTransformer(vocab_size=configue.vocab_size, decoder_layer=configue.decoder_layer, n_head=configue.n_head, n_emb=configue.n_emb, context_length=MAX_LENGTH, pad_token_id=PAD_ID)
optimizer = Adam(model.parameters(), lr=0.005)
model.to(device)

# Load checkpoint if exists
if os.path.exists(CHECK_POINT):
    model.load_state_dict(torch.load(CHECK_POINT))
    print(f"Loaded checkpoint: {CHECK_POINT}")
else:
    print("No checkpoint found, starting from scratch.")

# Load datasets
train_dataset = MusicDataset('dataset/dataset_json/train_set')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
validation_dataset = MusicDataset('dataset/dataset_json/validation_set')
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# Initialize wandb
wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME)

# Scheduler for learning rate adjustment
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

def validate_model():
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(validation_loader)
    wandb.log({"validation_loss": avg_val_loss})
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss

# Training loop
model.train()
total_steps = 0
for epoch in range(EPOCH_NUM):
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Log training loss to wandb
        wandb.log({"loss": loss.item()})

        # Checkpoint saving logic
        total_steps += 1
        if total_steps % STEP_SIZE == 0:
            checkpoint_path = f"model_output/checkpoints/ckpt_{total_steps}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Validate model after each epoch and adjust learning rate
    val_loss = validate_model()
    scheduler.step(val_loss)
    wandb.log({"EPOCH": epoch})
    wandb.log({"train_rate": loss.item()})
    current_lr = scheduler.optimizer.param_groups[0]['lr']
    wandb.log({"learning_rate": current_lr})
