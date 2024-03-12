from transformers import PreTrainedTokenizerFast

# Load the saved tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="updated_tokenizer.json", model_max_length=1024, unk_token="[UNK]")

# Check the tokenizer configuration
print("Tokenizer Configuration:", tokenizer)
print("UNK Token:", tokenizer.unk_token)
