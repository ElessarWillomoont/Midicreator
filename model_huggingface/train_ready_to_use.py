import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import DecoderOnlyTransformer
from preprocess import MusicDataset, custom_collate_fn
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
PROJECT_NAME = 'Midicreator_Hugging_face'
ENTITY_NAME = 'candle2587_team'
EPOCH_NUM = 4000
STEP_SIZE = 30000
BATCH_SIZE = 1024
MAX_LENGTH = 8
PAD_ID = 0
CHECK_POINT = "NO"  # Specify your checkpoint path

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DecoderOnlyTransformer(vocab_size=30000, decoder_layer=12, n_head=2, n_emb=768, context_length=MAX_LENGTH, pad_token_id=PAD_ID)
optimizer = Adam(model.parameters(), lr=0.001)
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
    current_lr = optimizer.get_last_lr()[0]
    wandb.log({"learning_rate": current_lr})
