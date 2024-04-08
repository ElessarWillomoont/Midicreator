from miditoolkit import MidiFile
from miditok import REMI
from pathlib import Path

def load_midi(midi_path):
    """Load a MIDI file."""
    return MidiFile(midi_path)

# Create a tokenizer config with your desired parameters
tokenizer_path = Path('tokenizer/tokenizer.json')

# Initialize tokenizer using the saved parameters
tokenizer = REMI(params=tokenizer_path)

# Set input and output directories
input_dir = Path("maestro")
token_dir = Path("dataset/midis/tokens")
dataset_dir = Path("dataset/dataset_json")
token_dir.mkdir(parents=True, exist_ok=True)
dataset_dir.mkdir(parents=True, exist_ok=True)

tokenizer.tokenize_midi_dataset(input_dir, dataset_dir)
'''
# Process each MIDI file
for midi_file_path in input_dir.rglob("*.midi"):
    midi = load_midi(midi_file_path)

    # Tokenize the complete MIDI file
    tokenizer.tokenize_midi_dataset(input_dir, dataset_dir)

    
    # Save tokenized output to a JSON file
    output_file_path = token_dir / (midi_file_path.stem + '.json')
    with open(output_file_path, 'w') as tf:
        json.dump({"ids": token_ids}, tf)
    '''

    # Optionally, you can combine all tokenized outputs into a single JSON file
    # combined_json_file_path = dataset_dir / (midi_file_path.stem + ".json")
    # with open(combined_json_file_path, 'w') as cjf:
    #     json.dump({"ids": token_ids}, cjf)
