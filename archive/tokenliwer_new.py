from datasets import load_dataset

DATASET_PATH = "dataset"

raw_datasets = load_dataset('text', 
                            data_files={
                                'train': f'{DATASET_PATH}/*_train_*.txt', 
                                'validation': f'{DATASET_PATH}/*_valid_*.txt'
                            })

sample_10 = raw_datasets["train"]["text"][10]
sample = sample_10[:242]
sample

# Default GPT-2 tokenizer applied to our dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
#print(tokenizer(sample).tokens())

from tokenizers import Tokenizer
from tokenizers.models import WordLevel

# We need to specify the UNK token
new_tokenizer = Tokenizer(model=WordLevel(unk_token="[UNK]"))

# Add pretokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit

new_tokenizer.pre_tokenizer = WhitespaceSplit()

# Let's test our pre_tokenizer
new_tokenizer.pre_tokenizer.pre_tokenize_str(sample)
# Yield batches of 1,000 texts
def get_training_corpus():
  dataset = raw_datasets["train"]
  for i in range(0, len(dataset), 1000):
    yield dataset[i : i + 1000]["text"]

from tokenizers.trainers import WordLevelTrainer

# Add special tokens
trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# Post-processing and updloading it to the Hub
from transformers import PreTrainedTokenizerFast

new_tokenizer.train_from_iterator(get_training_corpus(), trainer) #seems the professor forgot to train the tokenizer

new_tokenizer.save("tokenizer.json")

new_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
new_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

new_tokenizer.push_to_hub("MIDIGPT_MAESTRO_tokenizer")

