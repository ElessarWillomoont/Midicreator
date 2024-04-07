# import mido
# import time
# import note_seq
# from note_seq.protobuf import music_pb2
# from transformers import GPT2LMHeadModel, AutoTokenizer
# import note_seq
# from note_seq.protobuf import music_pb2
# from mido import MidiFile, MidiTrack

# Other functions (play_note, detect_sequence) remain the same

NOTE_LENGTH_16TH_120BPM = 0.25 * 60 / 120
BAR_LENGTH_120BPM = 4.0 * 60 / 120


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