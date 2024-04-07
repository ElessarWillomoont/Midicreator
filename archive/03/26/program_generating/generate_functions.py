import mido
import time
import note_seq
from note_seq.protobuf import music_pb2
from transformers import GPT2LMHeadModel, AutoTokenizer
import note_seq
from note_seq.protobuf import music_pb2
from mido import MidiFile, MidiTrack

# Other functions (play_note, detect_sequence) remain the same

NOTE_LENGTH_16TH_120BPM = 0.25 * 60 / 120
BAR_LENGTH_120BPM = 4.0 * 60 / 120

def midi_file_to_token_sequence(midi_file,tokenlizer):
    """ Convert a MIDI file to a token sequence representation. """
    note_sequence = note_seq.midi_file_to_note_sequence(midi_file)

    tokens = ["PIECE_START", "TIME_SIGNATURE=4_4", "GENRE=OTHER", "TRACK_START", "INST=0", "DENSITY=0"]
    current_time = 0
    note_end_times = {}

    for note in note_sequence.notes:
        # Time Delta
        time_delta = note.start_time - current_time
        if time_delta > 0:
            tokens.append(f"TIME_DELTA={time_delta}")
            current_time = note.start_time

        # Note On
        tokens.append(f"NOTE_ON={note.pitch}")

        # Calculating Note End Time
        note_end_time = note.end_time
        note_end_times[note_end_time] = note_end_times.get(note_end_time, [])
        note_end_times[note_end_time].append(note.pitch)

        # Checking for Note Offs that should occur before the next Note On
        for end_time in sorted(note_end_times):
            if end_time <= current_time:
                continue
            if end_time > current_time:
                time_delta = end_time - current_time
                tokens.append(f"TIME_DELTA={time_delta}")
                for pitch in note_end_times[end_time]:
                    tokens.append(f"NOTE_OFF={pitch}")
                current_time = end_time
                note_end_times[end_time] = []

    return ' '.join(tokens)

def empty_note_sequence(qpm=120.0):
    """ Create an empty NoteSequence with a given tempo (QPM). """
    note_sequence = music_pb2.NoteSequence()
    note_sequence.tempos.add().qpm = qpm
    note_sequence.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    return note_sequence

def token_sequence_to_note_sequence(token_sequence):
    """ Convert a sequence of tokens into a NoteSequence. """
    note_sequence = empty_note_sequence()

    current_time = 0
    current_instrument = 0
    current_program = 0
    current_is_drum = False

    for token in token_sequence.split():
        if token.startswith("NOTE_ON"):
            pitch = int(token.split("=")[-1])
            note = note_sequence.notes.add()
            note.start_time = current_time
            note.end_time = current_time + NOTE_LENGTH_16TH_120BPM  # Assuming each note is a 16th note
            note.pitch = pitch
            note.velocity = 80
            note.instrument = current_instrument
            note.program = current_program
            note.is_drum = current_is_drum
        elif token.startswith("TIME_DELTA"):
            delta = float(token.split("=")[-1]) * NOTE_LENGTH_16TH_120BPM
            current_time += delta
        # Add more conditions here for other types of tokens (e.g., NOTE_OFF, BAR_START, etc.)

    return note_sequence

def detect_notes_sequence(inport):
    desired_sequence = [60, 62, 64, 65, 67, 69, 71]  # MIDI notes for CDEFGAB
    pressed_sequence = []

    print("Listening for sequence CDEFGAB...")
    start_time = time.time()
    while time.time() - start_time < 10:  # Listen for 10 seconds
        for msg in inport.iter_pending():
            if msg.type == 'note_on' and msg.velocity > 0 and msg.note in desired_sequence:
                if not pressed_sequence or msg.note != pressed_sequence[-1]:  # Avoid duplicate consecutive notes
                    pressed_sequence.append(msg.note)

                    if pressed_sequence == desired_sequence[:len(pressed_sequence)]:
                        if len(pressed_sequence) == len(desired_sequence):
                            return True
                    else:
                        return False

    return False

def detect_reverse_notes_sequence(inport):
    reverse_sequence = [71, 69, 67, 65, 64, 62, 60]  # MIDI notes for BAGFEDC
    pressed_sequence = []

    print("Listening for reverse sequence BAGFEDC...")
    start_time = time.time()
    while time.time() - start_time < 10:  # Listen for 10 seconds
        for msg in inport.iter_pending():
            if msg.type == 'note_on' and msg.velocity > 0 and msg.note in reverse_sequence:
                if not pressed_sequence or msg.note != pressed_sequence[-1]:
                    pressed_sequence.append(msg.note)

                    if pressed_sequence == reverse_sequence[:len(pressed_sequence)]:
                        if len(pressed_sequence) == len(reverse_sequence):
                            return True
                    else:
                        return False
    return False

def detect_sequence(inport, note, count=3):
    """ Generic function to detect a sequence of the same note repeated 'count' times. """
    pressed_count = 0
    note_pressed = False

    print(f"Listening for sequence of {count} '{note}' notes...")
    start_time = time.time()
    while time.time() - start_time < 10:  # Listen for 10 seconds
        for msg in inport.iter_pending():
            if msg.type == 'note_on' and msg.velocity > 0 and msg.note == note:
                if not note_pressed:  # Count the note only if it wasn't already pressed
                    pressed_count += 1
                    note_pressed = True
                    if pressed_count == count:
                        return True
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note == note:
                    note_pressed = False  # Reset when the note is released

    return False

def record_midi(inport):
    """ Record MIDI input when start sequence is detected and stop when end sequence is detected. """
    recording = []
    recording_started = False
    print("Waiting for start sequence...")

    while True:
        for msg in inport.iter_pending():
            if not recording_started:
                # Start recording if the start sequence is detected
                if detect_notes_sequence(inport):
                    print("Start sequence detected, beginning recording.")
                    recording_started = True
            else:
                # Add messages to recording if recording has started
                recording.append(msg)

                # Stop recording if the end sequence is detected
                if detect_reverse_notes_sequence(inport):
                    print("End sequence detected, stopping recording.")
                    return recording



def play_midi_file(port, filename, play_duration=20):
    """ Play a MIDI file through the specified MIDI port for a specified duration, and turn off all notes at the end. """
    midi_file = mido.MidiFile(filename)

    elapsed_time = 0
    active_notes = set()
    for msg in midi_file.play():
        elapsed_time += msg.time
        if elapsed_time > play_duration:
            break

        if msg.type == 'note_on' and msg.velocity > 0:
            active_notes.add(msg.note)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            active_notes.discard(msg.note)

        port.send(msg)

    # Turn off any remaining active notes
    for note in active_notes:
        off_msg = mido.Message('note_off', note=note)
        port.send(off_msg)

def process_midi(input_midi_file, output_midi_file):
    # Convert input MIDI to tokens
    token_sequence = midi_file_to_token_sequence(input_midi_file)
    print("Input MIDI converted to tokens.")

    # Load the model
    model_path = "checkpoint-140000"  # Modify with your model path
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("Lancelort5786/MIDIGPT_MAESTRO_tokenizer")

    # Prepare input for the model
    inputs = tokenizer(token_sequence, return_tensors="pt")

    # Generate a new series of tokens
    output_sequences = model.generate(
        input_ids=inputs['input_ids'], 
        max_length=1024,  # Adjust the max_length accordingly
        num_return_sequences=1,
        no_repeat_ngram_size=1,
        temperature=2.0,
        top_k=200,
        top_p=0.95,
        # Add other parameters as needed
    )

    # Decode the output sequences to text (tokens)
    generated_token_sequence = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print("Generated new token sequence.")

    # Convert the generated tokens to a MIDI file
    generated_note_sequence = token_sequence_to_note_sequence(generated_token_sequence)
    note_seq.sequence_proto_to_midi_file(generated_note_sequence, output_midi_file)
    print(f"MIDI file saved as {output_midi_file}")

def main():
    print("entered mf at gfs")
    input("say sth")
    ports_output = mido.get_output_names()
    ports_input = mido.get_input_names()
    print("Available MIDI output ports:", ports_output)
    print("Available MIDI input ports:", ports_input)

    port_name_output = [name for name in ports_output if 'Disklavier' in name]
    port_name_input = [name for name in ports_input if 'Disklavier' in name]

    if not port_name_output or not port_name_input:
        print("Disklavier piano not found.")
        return

    with mido.open_output(port_name_output[0]) as port, mido.open_input(port_name_input[0]) as ports_input:
        print(f"Sending notes to port {port_name_output[0]}")
        while True:  # First loop
            if detect_sequence(ports_input, 60, 3):  # Detecting CCC
                print("Sequence CCC detected, entering second loop")

                while True:  # Second loop
                    recorded_notes = record_midi(ports_input)
                    print("Recording complete")
                    midi_file = MidiFile()
                    track = MidiTrack()
                    midi_file.tracks.append(track)
                    for msg in recorded_notes:
                        track.append(msg)
                    midi_file.save("input.mid")
                    print("Recording saved to 'input.mid'")
                    if detect_sequence(ports_input, 71, 3):  # Detecting BBB
                        print("Sequence BBB detected, playing 'output.mid'")
                        process_midi("input.mid", "output_withspace.mid")
                        remove_silence_from_midi("output_withspace.mid", "output.mid")
                        print("prcess completed")
                        try:
                            play_midi_file(port, 'output.mid', 10)
                        except FileNotFoundError:
                            print("File 'output.mid' not found.")

                    if detect_sequence(ports_input, 64, 3):  # Detecting EEE
                        print("Sequence EEE detected, returning to first loop")
                        break  # Exit the second loop

if __name__ == "__main__":
    main()