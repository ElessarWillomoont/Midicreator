from mido import MidiFile, MidiTrack, Message
import re

def parse_midi_events(input_string):
    # Split the string into individual commands
    commands = re.findall(r'(\w+)=?([-\d._]*)', input_string)

    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)

    # Default values
    current_time = 0

    for command, value in commands:
        if command == 'NOTE_ON':
            note = int(value)
            track.append(Message('note_on', note=note, velocity=64, time=int(current_time)))
            current_time = 0  # Reset delta time after event
        elif command == 'NOTE_OFF':
            note = int(value)
            track.append(Message('note_off', note=note, velocity=64, time=int(current_time)))
            current_time = 0  # Reset delta time after event
        elif command == 'TIME_DELTA':
            # Convert the time delta from beats to ticks
            current_time += int(float(value) * midi_file.ticks_per_beat)

    return midi_file

# Read the input string from a file
input_string = input("please input the output")

# Parse the string and create a MIDI file
midi_file = parse_midi_events(input_string)

# Save the MIDI file
output_path = 'output_music.mid'
midi_file.save(output_path)
print(f"MIDI file saved as {output_path}")
