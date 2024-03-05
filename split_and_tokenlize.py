from miditoolkit import MidiFile, Instrument
from miditok import REMI, TokenizerConfig
from pathlib import Path
from copy import deepcopy
from math import ceil
import json

def load_midi(midi_path):
    """Load a MIDI file."""
    return MidiFile(midi_path)

def split_midi(midi, max_nb_bar=4):
    """将MIDI文件切分为多个片段，每个片段为一小节"""
    splits = []
    ticks_per_cut = max_nb_bar * midi.ticks_per_beat * 4
    nb_cuts = ceil(midi.max_tick / ticks_per_cut)

    for i in range(nb_cuts):
        midi_short = MidiFile()
        midi_short.ticks_per_beat = midi.ticks_per_beat

        earliest_start = None
        for track in midi.instruments:
            track_short = Instrument(program=track.program, is_drum=track.is_drum, name=track.name)
            notes = [deepcopy(note) for note in track.notes if ticks_per_cut * i <= note.start < ticks_per_cut * (i + 1)]

            for note in notes:
                if earliest_start is None or note.start < earliest_start:
                    earliest_start = note.start

            track_short.notes = notes
            if len(track_short.notes) > 0:
                midi_short.instruments.append(track_short)

        if earliest_start is not None:
            for track in midi_short.instruments:
                for note in track.notes:
                    note.start -= earliest_start
                    note.end -= earliest_start

        if len(midi_short.instruments) > 0:
            splits.append(midi_short)

    return splits


# Create a tokenizer config with your desired parameters
# 加载先前保存的tokenizer参数
tokenizer_path = Path('tokenizer/tokenizer.json')

# 初始化tokenizer，使用保存的参数
tokenizer = REMI(params=tokenizer_path)

# Set input and output directories
input_dir = Path("maestro")
split_dir = Path("dataset/midis/splited")
token_dir = Path("dataset/midis/tokens")
dataset_dir = Path("dataset/dataset_json")
split_dir.mkdir(parents=True, exist_ok=True)
token_dir.mkdir(parents=True, exist_ok=True)
dataset_dir.mkdir(parents=True, exist_ok=True)

# Process each MIDI file
for midi_file_path in input_dir.rglob("*.midi"):
    midi = load_midi(midi_file_path)
    midi_splits = split_midi(midi)

    split_file_paths = []
    for i, midi_split in enumerate(midi_splits):
        split_file_path = split_dir / f"{midi_file_path.stem}_split_{i}.mid"
        midi_split.dump(split_file_path)
        split_file_paths.append(split_file_path)

    # Tokenize the splits
    tokenizer.tokenize_midi_dataset(split_file_paths, token_dir)

    # Combine tokenized outputs into a single JSON file
    combined_tokens = []
    for split_file_path in split_file_paths:
        token_file_path = token_dir / (split_file_path.stem + '.json')
        with open(token_file_path, 'r') as tf:
            tokens = json.load(tf)
            combined_tokens.append(tokens)  # Append each token sequence as a separate row

    combined_json_file_path = dataset_dir / (midi_file_path.stem + ".json")
    with open(combined_json_file_path, 'w') as cjf:
        for token_sequence in combined_tokens:
            cjf.write(json.dumps(token_sequence) + '\n')  # Write each token sequence as a separate line
