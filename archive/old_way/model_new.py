from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoConfig, GPT2LMHeadModel, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
import wandb
# Specify your local dataset path
DATASET_PATH = "dataset"

# Load your local dataset
raw_datasets = load_dataset('text', 
                            data_files={
                                'train': f'{DATASET_PATH}/*_train_*.txt', 
                                'validation': f'{DATASET_PATH}/*_valid_*.txt'
                            })

tk_sample = raw_datasets["train"][1000]
#text_sample = raw_datasets['train']['text'].iloc[1000]
#tk_sample = tokenize({"text": text_sample})
print(tk_sample)
print("Data type:", type(tk_sample))
'''
# Load your local tokenizer
tokenizer = AutoTokenizer.from_pretrained("Lancelort5786/MIDIGPT_MAESTRO_tokenizer")

# Define context length
context_length = 2048

# Function for tokenizing the dataset
def tokenize(element):
    outputs = tokenizer(
        element["text"],
        truncation=True,
        max_length=context_length,
        padding=False
    )
    return {"input_ids": outputs["input_ids"]}

# Tokenize the dataset
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

# Define model configuration
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_positions=context_length,
    n_layer=6,  # Number of transformer layers
    n_head=8,   # Number of multi-head attention heads
    n_embd=512, # Embedding size
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Initialize model
model = GPT2LMHeadModel(config)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # Set to False for causal language modeling (GPT-2 style)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model_output",  # Directory for saving model outputs
    per_device_train_batch_size=3,  # Training batch size per device
    num_train_epochs=3,  # Number of training epochs
    # Add other training arguments as needed
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# You can now start training with `trainer.train()`

from transformers import Trainer, TrainingArguments

# first create a custom trainer to log prediction distribution
SAMPLE_RATE=44100
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # call super class method to get the eval outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # log the prediction distribution using `wandb.Histogram` method.
        if wandb.run is not None:
            input_ids = tokenizer.encode("PIECE_START STYLE=JSFAKES GENRE=JSFAKES TRACK_START", return_tensors="pt").cuda()
            # Generate more tokens.
            voice1_generated_ids = model.generate(
                input_ids,
                max_length=512,
                do_sample=True,
                temperature=0.75,
                eos_token_id=tokenizer.encode("TRACK_END")[0]
            )
            voice2_generated_ids = model.generate(
                voice1_generated_ids,
                max_length=512,
                do_sample=True,
                temperature=0.75,
                eos_token_id=tokenizer.encode("TRACK_END")[0]
            )
            voice3_generated_ids = model.generate(
                voice2_generated_ids,
                max_length=512,
                do_sample=True,
                temperature=0.75,
                eos_token_id=tokenizer.encode("TRACK_END")[0]
            )
            voice4_generated_ids = model.generate(
                voice3_generated_ids,
                max_length=512,
                do_sample=True,
                temperature=0.75,
                eos_token_id=tokenizer.encode("TRACK_END")[0]
            )
            token_sequence = tokenizer.decode(voice4_generated_ids[0])
            note_sequence = token_sequence_to_note_sequence(token_sequence)
            synth = note_seq.fluidsynth
            array_of_floats = synth(note_sequence, sample_rate=SAMPLE_RATE)
            int16_data = note_seq.audio_io.float_samples_to_int16(array_of_floats)
            wandb.log({"Generated_audio": wandb.Audio(int16_data, SAMPLE_RATE)})


        return eval_output
    
from argparse import Namespace

# Get the output directory with timestamp.
output_path = "output"
steps = 50000
# Commented parameters
config = {"output_dir": output_path,
          "num_train_epochs": 6,
          "per_device_train_batch_size": 3,
          "per_device_eval_batch_size": 3,
          "evaluation_strategy": "steps",
          "save_strategy": "steps",
          "eval_steps": steps,
          "logging_steps":steps,
          "logging_first_step": True,
          "save_total_limit": 5,
          "save_steps": steps,
          "lr_scheduler_type": "cosine",
          "learning_rate":5e-4,
          "warmup_ratio": 0.01,
          "weight_decay": 0.01,
          "seed": 1,
          "load_best_model_at_end": True,
          "report_to": "wandb"}

args = Namespace(**config)

train_args = TrainingArguments(**config)

# Use the CustomTrainer created above
trainer = CustomTrainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
'''