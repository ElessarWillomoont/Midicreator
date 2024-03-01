from pathlib import Path
from copy import deepcopy
from math import ceil

from miditoolkit import MidiFile
from tqdm import tqdm

# Constants
MAX_NB_BAR = 8
MIN_NB_NOTES = 20
dataset_name = "clean_midi"  # Update this if your subfolder inside 'dataset' has a different name

# Set the paths
current_dir = Path.cwd()
dataset_path = current_dir / 'maestro'
merged_out_dir = dataset_path / f"{dataset_name}-chunked"
merged_out_dir.mkdir(parents=True, exist_ok=True)

# Find all MIDI files in the dataset directory and its subdirectories
midi_paths = list(dataset_path.glob("**/*.mid")) + list(dataset_path.glob("**/*.midi"))

# Process each MIDI file
for i, midi_path in enumerate(tqdm(midi_paths, desc="CHUNKING MIDIS")):
    try:
        # Determine the output directory for this file
        relative_path = midi_path.relative_to(dataset_path)
        output_dir = merged_out_dir / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if chunks already exist
        chunk_paths = list(output_dir.glob(f"{midi_path.stem}_*.mid"))
        if len(chunk_paths) > 0:
            print(f"Chunks for {midi_path} already exist, skipping...")
            continue

        # Load MIDI, merge and save it
        midi = MidiFile(midi_path)
        ticks_per_cut = MAX_NB_BAR * midi.ticks_per_beat * 4
        nb_cuts = ceil(midi.max_tick / ticks_per_cut)
        if nb_cuts < 2:
            continue

        # Add specific MIDI file checks here if necessary

        print(f"Processing {midi_path}")
        midis = [deepcopy(midi) for _ in range(nb_cuts)]

        for j, track in enumerate(midi.instruments):
            track.notes.sort(key=lambda x: x.start)
            for midi_short in midis:
                midi_short.instruments[j].notes = []
            for note in track.notes:
                cut_id = note.start // ticks_per_cut
                note_copy = deepcopy(note)
                note_copy.start -= cut_id * ticks_per_cut
                note_copy.end -= cut_id * ticks_per_cut
                midis[cut_id].instruments[j].notes.append(note_copy)

        # Save the MIDI chunks
        for j, midi_short in enumerate(midis):
            if sum(len(track.notes) for track in midi_short.instruments) < MIN_NB_NOTES:
                continue
            midi_short.dump(output_dir / f"{midi_path.stem}_{j}.mid")

    except Exception as e:
        print(f"An error occurred while processing {midi_path}: {e}")



