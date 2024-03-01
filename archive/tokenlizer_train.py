from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset

# Load your dataset
DATASET_PATH = "dataset"
raw_datasets = load_dataset('text', data_files={'train': f'{DATASET_PATH}/*_train_*.txt'})

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

# Save the tokenizer using the tokenizers library
tokenizer.save("updated_tokenizer.json")
