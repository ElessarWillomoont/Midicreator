from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoConfig, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Import the AutoConfig class
from transformers import AutoConfig
# Load your dataset
DATASET_PATH = "dataset"
raw_datasets = load_dataset('text', 
                            data_files={
                                'train': f'{DATASET_PATH}/*_train_*.txt', 
                                'validation': f'{DATASET_PATH}/*_valid_*.txt'
                            })

# Initialize the tokenizer with WordLevel model and specify the UNK token
tokenizer = Tokenizer(model=WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()

# Function to yield batches of texts for training the tokenizer
def get_training_corpus():
    for i in range(0, len(raw_datasets["train"]), 1000):
        yield raw_datasets["train"][i: i + 1000]["text"]

# Create a trainer for the tokenizer with special tokens
trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# Train the tokenizer
tokenizer.train_from_iterator(get_training_corpus())

# Save the tokenizer
tokenizer.save("updated_tokenizer.json")

from transformers import PreTrainedTokenizerFast

# Load the tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="updated_tokenizer.json", unk_token="[UNK]", model_max_length=1024)

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Verify the tokenizer's configuration
print("Tokenizer Configuration:", tokenizer)
print("UNK Token:", tokenizer.unk_token)
print("Padding Token:", tokenizer.pad_token)

# Define max_length for your model
max_length = 1024

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Configuring the GPT-2 model
model_config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_positions=max_length
)

model = GPT2LMHeadModel(config=model_config)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
trainer.train()
