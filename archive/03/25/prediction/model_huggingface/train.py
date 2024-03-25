import torch
from torch.optim import Adam
from models import DecoderOnlyTransformer
from preprocess import MusicDataset, custom_collate_fn, preprocess_data
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

PROJECT_NAME = 'Midicreator_Prediction_Hugging_face'
ENTITY_NAME = 'candle2587_team'
EPOCH_NUM = 4000
STEP_SIZE = 200  # 每多少步进行一次检查和存储检查点
BATCH_SIZE = 1024
LOAD_DATA_THREAD = 2
MAX_LENGTH = 8
PAD_ID = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Assuming vocab_size, decoder_layer, n_head, n_emb, context_length, and pad_token_id are defined
model = DecoderOnlyTransformer(vocab_size=30000, decoder_layer=12, n_head=2, n_emb=768, context_length=MAX_LENGTH, pad_token_id=PAD_ID)
optimizer = Adam(model.parameters(), lr=0.001)
model.to(device)
# Load the dataset
train_dataset = MusicDataset('train_set')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

# Initialize wandb
wandb.init(project=PROJECT_NAME, entity=ENTITY_NAME)

# Training loop
model.train()
total_steps = 0
for epoch in range(EPOCH_NUM):
    # Wrap train_loader with tqdm for a progress bar
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Log loss to wandb
        wandb.log({"loss": loss.item()})

        # Print loss (optional, as tqdm also shows progress)
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Checkpoint saving logic
        total_steps += 1
        if total_steps % STEP_SIZE == 0:
            checkpoint_path = f"model_output/checkpoints/ckpt_{total_steps}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
