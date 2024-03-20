import json
import glob
import os
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# [Functions: read_and_process_json, calculate_similarity, create_input_target_pairs, encode_data, MusicDataset, save_checkpoint, load_latest_checkpoint, calculate_accuracy, train, evaluate, save_metrics_to_json]
def calculate_accuracy(preds, labels):
    # Flatten outputs and labels
    preds_flat = preds.argmax(dim=2).flatten()
    labels_flat = labels.flatten()

    # Calculate accuracy
    correct_predictions = (preds_flat == labels_flat).sum().item()
    total_predictions = labels_flat.size(0)

    # Avoid division by zero
    if total_predictions == 0:
        return 0

    return correct_predictions / total_predictions

# Function to read and process a single JSON file
def read_and_process_json(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def calculate_similarity(seq1, seq2):
    set1 = set(seq1)
    set2 = set(seq2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0
    return len(intersection) / len(union)

def create_input_target_pairs(data, similarity_threshold=0.9):
    input_target_pairs = []
    for i in range(len(data) - 1):
        input_seq = data[i]['ids']
        target_seq = data[i + 1]['ids']
        similarity = calculate_similarity(input_seq, target_seq)
        if similarity < similarity_threshold:
            input_target_pairs.append((input_seq, target_seq))
    return input_target_pairs

# Function to encode data
def encode_data(tokenizer, input_target_pairs):
    encoded_inputs = []
    encoded_targets = []
    for input_seq, target_seq in input_target_pairs:
        encoded_input = tokenizer.encode(input_seq, truncation=True, padding='max_length', max_length=512)
        encoded_target = tokenizer.encode(target_seq, truncation=True, padding='max_length', max_length=512)
        encoded_inputs.append(encoded_input)
        encoded_targets.append(encoded_target)
    return {"input_ids": encoded_inputs, "labels": encoded_targets}

# Custom Dataset class
class MusicDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Function to save a checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{timestamp}_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

# Function to load the latest checkpoint
def load_latest_checkpoint(checkpoint_dir, model, optimizer):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt'))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

# Functions to train and evaluate the model
def train(epoch, model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_accuracy = 0

    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += calculate_accuracy(logits, labels)

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)
    return average_loss, average_accuracy

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(logits, labels)

    average_loss = total_loss / len(dataloader)
    average_accuracy = total_accuracy / len(dataloader)
    return average_loss, average_accuracy


# Function to save losses to a JSON file
def save_metrics_to_json(train_losses, val_losses, train_accuracies, val_accuracies, json_file_path):
    metrics = {
        'train_losses': train_losses, 
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    with open(json_file_path, 'w') as json_file:
        json.dump(metrics, json_file)


# Function to plot loss curves
def plot_loss_curves(json_file_path):
    with open(json_file_path, 'r') as json_file:
        losses = json.load(json_file)
    plt.plot(losses['train_losses'], label='Training Loss')
    plt.plot(losses['val_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

# Load training data
directory_path = 'precondition/dataset/'
json_files = glob.glob(directory_path + '*.json')
all_data = []
for file_path in json_files:
    file_data = read_and_process_json(file_path)
    all_data.extend(file_data)

# Creating input-target pairs for training data
input_target_pairs = create_input_target_pairs(all_data, similarity_threshold=0.9)

# Load validation data
validation_directory_path = 'precondition/validation/'
validation_json_files = glob.glob(validation_directory_path + '*.json')
validation_data = []
for file_path in validation_json_files:
    file_data = read_and_process_json(file_path)
    validation_data.extend(file_data)

# Creating input-target pairs for validation data
validation_input_target_pairs = create_input_target_pairs(validation_data, similarity_threshold=0.9)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

# Encoding training and validation data
encodings = encode_data(tokenizer, input_target_pairs)
validation_encodings = encode_data(tokenizer, validation_input_target_pairs)

# Creating datasets and dataloaders
dataset = MusicDataset(encodings)
validation_dataset = MusicDataset(validation_encodings)
dataloader = DataLoader(dataset, batch_size=15, shuffle=True)
val_dataloader = DataLoader(validation_dataset, batch_size=15, shuffle=True)

# Set up optimizer and device
optimizer = AdamW(model.parameters(), lr=5e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Checkpoint directory
checkpoint_dir = 'check_points'
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop with loss and accuracy recording
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
metrics_json_file_path = 'training_validation_metrics.json'
start_epoch = load_latest_checkpoint(checkpoint_dir, model, optimizer)

num_epochs = 15
for epoch in range(start_epoch, num_epochs):
    print("entered traning," + str(epoch))
    train_loss, train_accuracy = train(epoch, model, dataloader, optimizer, device)
    save_checkpoint(model, optimizer, epoch, checkpoint_dir)
    print("checkpoint complete")
    print("train complete, validate")
    val_loss, val_accuracy = evaluate(model, val_dataloader, device)
    print("validate complete")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    print("calculate compete")
    print("train loss:" + str(train_loss) + "validation loss:" + str(val_loss) + "train accuracy:" + str(train_accuracy) + "validation accuracy:" + str(val_accuracy) )
    save_metrics_to_json(train_losses, val_losses, train_accuracies, val_accuracies, metrics_json_file_path)
    print("data saving complete")

# Optional: Function to plot the accuracy curves
def plot_accuracy_curves(json_file_path):
    with open(json_file_path, 'r') as json_file:
        metrics = json.load(json_file)
    plt.plot(metrics['train_accuracies'], label='Training Accuracy')
    plt.plot(metrics['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curves')
    plt.legend()
    plt.show()

# plot_accuracy_curves(metrics_json_file_path) # Uncomment to plot accuracy curves
