WANDB_PROJECT = "MAESTRO_GPT2"
ENTITY = None # set this to team name if working in a team
# = "lms-8bars_raw"
#PROCESSED_DATA_AT = "lms-8bars_processed"
from datasets import load_dataset
from transformers import AutoTokenizer
ds = load_dataset("Lancelort5786/MAESTRO_MMM", split="train")
raw_datasets = ds.train_test_split(test_size=0.1, shuffle=True)
# Change for respective tokenizer
tokenizer = AutoTokenizer.from_pretrained("Lancelort5786/MIDIGPT_MAESTRO_tokenizer")
raw_datasets

train_ds = raw_datasets["train"]
train_ds.set_format("pandas")
df_train = train_ds[:]
df_train.head()

test_ds = raw_datasets["test"]
test_ds.set_format("pandas")
df_test = test_ds[:]
df_test.head()

import pandas as pd
df = pd.concat([df_train, df_test])
df.head()

df['word_count'] = df['text'].str.split().str.len()

# 计算最大、最小和平均单词数
max_words = df['word_count'].max()
min_words = df['word_count'].min()
mean_words = df['word_count'].mean()

# 计算95%百分位数
percentile_95 = df['word_count'].quantile(0.95)

# 打印统计数据
print(f'Max number of words: {max_words}')
print(f'Min number of words: {min_words}')
print(f'Mean number of words: {mean_words}')
print(f'95th percentile of word counts: {percentile_95}')

# Delete the temporary 'word_count' column
df.drop(columns='word_count', inplace=True)
raw_datasets

# Replace this based on Dataset
context_length = 1024

def tokenize(element):
  outputs = tokenizer(
      element['text'],
      truncation=True, #Removing element longer that context size, no effect in JSB
      max_length=context_length,
      padding=False
  )
  return {"input_ids": outputs["input_ids"]}


train_ds.reset_format()
test_ds.reset_format()

# Create tokenize dataset
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

tokenized_datasets

# Convert to df
ds = tokenized_datasets["train"]
ds.set_format("pandas")
df = ds[:]
df.head()

# Get the len of the input_ids in each row
df["Len"] = df["input_ids"].apply(len)
df.head()

# Get statistics
df.describe()

total_num_of_tokens = df["Len"].sum()
print(f"There are {total_num_of_tokens:,d} in the train dataset")

# Let's now reset the format
ds.reset_format()

n_layer=9
n_head=4
n_emb=128

from transformers import AutoConfig, GPT2LMHeadModel

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_positions=context_length,
    n_layer=n_layer,
    n_head=n_head,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_embd=n_emb
)

model = GPT2LMHeadModel(config)
model

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Create the args for out trainer
from argparse import Namespace

# Get the output directory with timestamp.
output_path = "output"
steps = 5000
# Commented parameters correspond to the small model
config = {"output_dir": output_path,
          "num_train_epochs": 100,
          "per_device_train_batch_size": 24,
          "per_device_eval_batch_size": 8,
          "evaluation_strategy": "steps",
          "save_strategy": "steps",
          "eval_steps": steps,
          "logging_steps":steps,
          "logging_first_step": True,
          "save_total_limit": 50,
          "save_steps": steps,
          "lr_scheduler_type": "cosine",
          "learning_rate":5e-4,
          "warmup_ratio": 0.01,
          "weight_decay": 0.01,
          "seed": 1,
          "load_best_model_at_end": True,
          "report_to": "wandb"}

args = Namespace(**config)

from transformers import set_seed
set_seed(args.seed)
import wandb
run = wandb.init(project=WANDB_PROJECT, job_type="training", config=args)

from transformers import Trainer, TrainingArguments
'''
# Code to log also some audio in the raw data
import note_seq

NOTE_LENGTH_16TH_120BPM = 0.25 * 60 / 120
BAR_LENGTH_120BPM = 4.0 * 60 / 120

def token_sequence_to_note_sequence(token_sequence, use_program=True, use_drums=True, instrument_mapper=None, only_piano=False):

    if isinstance(token_sequence, str):
        token_sequence = token_sequence.split()

    note_sequence = empty_note_sequence()

    # Render all notes.
    current_program = 1
    current_is_drum = False
    current_instrument = 0
    track_count = 0
    for token_index, token in enumerate(token_sequence):

        if token == "PIECE_START":
            pass
        elif token == "PIECE_END":
            print("The end.")
            break
        elif token == "TRACK_START":
            current_bar_index = 0
            track_count += 1
            pass
        elif token == "TRACK_END":
            pass
        elif token == "KEYS_START":
            pass
        elif token == "KEYS_END":
            pass
        elif token.startswith("KEY="):
            pass
        elif token.startswith("INST"):
            instrument = token.split("=")[-1]
            if instrument != "DRUMS" and use_program:
                if instrument_mapper is not None:
                    if instrument in instrument_mapper:
                        instrument = instrument_mapper[instrument]
                current_program = int(instrument)
                current_instrument = track_count
                current_is_drum = False
            if instrument == "DRUMS" and use_drums:
                current_instrument = 0
                current_program = 0
                current_is_drum = True
        elif token == "BAR_START":
            current_time = current_bar_index * BAR_LENGTH_120BPM
            current_notes = {}
        elif token == "BAR_END":
            current_bar_index += 1
            pass
        elif token.startswith("NOTE_ON"):
            pitch = int(token.split("=")[-1])
            note = note_sequence.notes.add()
            note.start_time = current_time
            note.end_time = current_time + 4 * NOTE_LENGTH_16TH_120BPM
            note.pitch = pitch
            note.instrument = current_instrument
            note.program = current_program
            note.velocity = 80
            note.is_drum = current_is_drum
            current_notes[pitch] = note
        elif token.startswith("NOTE_OFF"):
            pitch = int(token.split("=")[-1])
            if pitch in current_notes:
                note = current_notes[pitch]
                note.end_time = current_time
        elif token.startswith("TIME_DELTA"):
            delta = float(token.split("=")[-1]) * NOTE_LENGTH_16TH_120BPM
            current_time += delta
        elif token.startswith("DENSITY="):
            pass
        elif token == "[PAD]":
            pass
        else:
            #print(f"Ignored token {token}.")
            pass

    # Make the instruments right.
    instruments_drums = []
    for note in note_sequence.notes:
        pair = [note.program, note.is_drum]
        if pair not in instruments_drums:
            instruments_drums += [pair]
        note.instrument = instruments_drums.index(pair)

    if only_piano:
        for note in note_sequence.notes:
            if not note.is_drum:
                note.instrument = 0
                note.program = 0

    return note_sequence

def empty_note_sequence(qpm=120.0, total_time=0.0):
    note_sequence = note_seq.protobuf.music_pb2.NoteSequence()
    note_sequence.tempos.add().qpm = qpm
    note_sequence.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    note_sequence.total_time = total_time
    return note_sequence

# first create a custom trainer to log prediction distribution
'''
'''
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
'''
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
        # Call superclass method to get the eval outputs
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Additional evaluation or logging logic can be added here if needed

        return eval_output
        
train_args = TrainingArguments(**config)
trainer = CustomTrainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
# Train the model.
trainer.train()
# call wandb.finish() to finish the run
wandb.finish()