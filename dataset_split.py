import os
import pretty_midi

def save_bar_as_midi(notes, start_time, end_time, file_path):
    """将一组音符保存成一个新的MIDI文件"""
    # 创建一个新的PrettyMIDI对象
    output_midi = pretty_midi.PrettyMIDI()
    # 创建一个乐器（钢琴），program number 0 是 Acoustic Grand Piano
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    # 添加音符
    for note in notes:
        # 复制音符并转换时间
        new_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start - start_time,
            end=note.end - start_time
        )
        piano.notes.append(new_note)
    # 将乐器添加到PrettyMIDI对象中
    output_midi.instruments.append(piano)
    # 保存MIDI文件
    output_midi.write(file_path)

def midi_to_bars_and_save(midi_file, dataset_dir, beats_per_bar=4):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    bars = []
    beats = midi_data.get_beats()
    for i in range(0, len(beats), beats_per_bar):
        bar_start = beats[i]
        try:
            bar_end = beats[i + beats_per_bar]
        except IndexError:
            bar_end = midi_data.get_end_time()
        notes_in_bar = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                if bar_start <= note.start < bar_end:
                    notes_in_bar.append(note)
        bars.append((notes_in_bar, bar_start, bar_end))

    # 创建训练对并保存为MIDI文件
    for i in range(0, len(bars) - 7, 4):
        input_bars, input_start, input_end = zip(*bars[i:i+4])
        target_bars, target_start, target_end = zip(*bars[i+4:i+8])
        input_notes = [note for bar in input_bars for note in bar]
        target_notes = [note for bar in target_bars for note in bar]
        # 创建文件名
        base_name = os.path.splitext(os.path.basename(midi_file))[0]
        input_file_name = f"{base_name}_input_{i//4}.midi"
        target_file_name = f"{base_name}_target_{i//4}.midi"
        # 保存文件
        save_bar_as_midi(input_notes, input_start[0], input_end[-1], os.path.join(dataset_dir, input_file_name))
        save_bar_as_midi(target_notes, target_start[0], target_end[-1], os.path.join(dataset_dir, target_file_name))

def process_midi_directory_and_save(midi_dir, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for root, _, files in os.walk(midi_dir):
        for file in files:
            if file.lower().endswith(('.midi', '.mid')):
                midi_path = os.path.join(root, file)
                midi_to_bars_and_save(midi_path, dataset_dir)

# Process the 'maestro' directory and save the training pairs in the 'dataset' directory
midi_dir = 'maestro'  # Your MIDI files directory
dataset_dir = 'dataset'  # Directory to save the training pairs
process_midi_directory_and_save(midi_dir, dataset_dir)

