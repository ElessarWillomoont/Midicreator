import os
import pretty_midi
from tqdm import tqdm  # 导入tqdm库

def save_bar_as_midi(notes, start_time, end_time, file_path):
    """将一组音符保存成一个新的MIDI文件"""
    output_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    for note in notes:
        new_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start - start_time,
            end=note.end - start_time
        )
        piano.notes.append(new_note)
    output_midi.instruments.append(piano)
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

    for i in tqdm(range(len(bars) - 1), desc="Processing bars"):  # 保留进度条
        input_notes, input_start, input_end = bars[i][0], bars[i][1], bars[i][2]
        target_notes, target_start, target_end = bars[i + 1][0], bars[i + 1][1], bars[i + 1][2]
        base_name = os.path.splitext(os.path.basename(midi_file))[0]
        input_file_name = f"{base_name}_input_{i}.midi"
        target_file_name = f"{base_name}_target_{i}.midi"
        save_bar_as_midi(input_notes, input_start, input_end, os.path.join(dataset_dir, input_file_name))
        save_bar_as_midi(target_notes, target_start, target_end, os.path.join(dataset_dir, target_file_name))

def process_midi_directory_and_save(midi_dir, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    midi_files = []  # 初始化一个空列表来存储MIDI文件路径
    for root, _, files in os.walk(midi_dir):
        for file in files:
            if file.lower().endswith(('.midi', '.mid')):
                midi_files.append(os.path.join(root, file))  # 使用root构造完整的文件路径

    # 将整个文件处理过程放入tqdm进度条中
    for file in tqdm(midi_files, desc="Processing MIDI files"):
        midi_to_bars_and_save(file, dataset_dir)

midi_dir = 'maestro'  # 您的MIDI文件目录
dataset_dir = 'dataset'  # 用于保存训练对的目录
process_midi_directory_and_save(midi_dir, dataset_dir)
