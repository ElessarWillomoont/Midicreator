import time
from GUI import StatusManager
import generator_founctions as gfs
from miditok import REMI
from pathlib import Path
from copy import deepcopy
from mido import MidiFile, MidiTrack
import mido
import time
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import QThread, pyqtSignal
import sys  # 导入sys用于退出应用
import sys
import time
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

class GenerateThread(QThread):
    status_changed = pyqtSignal(int)
    
    def run(self):
        new_status = 0
        self.status_changed.emit(new_status)
        tokenizer_path = Path('tokenizer/tokenizer.json')
        # Initialize tokenizer using the saved parameters
        tokenizer = REMI(params=tokenizer_path)
        midi_file_input = 'input.mid'
        midi_file_output = 'output.mid'
        vocabulary_json_file = 'shared/vocabulary/vocabulary.json'
        # vocabulary = tokenizer.vocab
        # save_vocabulary_to_json(vocabulary,json_file)
        print("entered mf at gfs")
        ports_output = mido.get_output_names()
        ports_input = mido.get_input_names()
        print("Available MIDI output ports:", ports_output)
        print("Available MIDI input ports:", ports_input)

        port_name_output = [name for name in ports_output if 'Disklavier' in name]
        port_name_input = [name for name in ports_input if 'Disklavier' in name]

        if not port_name_output or not port_name_input:
            print("Disklavier piano not found.")
            return
        with mido.open_input(port_name_input[0]) as ports_input:
            while True:
                print("Waiting...")
                time.sleep(1)  # 每秒打印一次等待信息
                messages = ports_input.iter_pending()
                if any(messages):
                    break  # 如果有任何按键消息，跳出循环
        new_status = 1
        self.status_changed.emit(new_status)
        with mido.open_output(port_name_output[0]) as port, mido.open_input(port_name_input[0]) as ports_input:
            print(f"Sending notes to port {port_name_output[0]}")
            low_c_pressed = False
            high_b_pressed = False
            while True:  # 主循环
                for msg in ports_input.iter_pending():
                    # 处理低音C和高音B的逻辑
                    if msg.type == 'note_on':
                        if msg.note == 36:  # 低音C
                            print("low C detected")
                            low_c_pressed = True
                            print("High B detected")
                        elif msg.note == 83:  # 高音B
                            high_b_pressed = True
                    elif msg.type == 'note_off':
                        if msg.note == 36:
                            low_c_pressed = False
                        elif msg.note == 83:
                            high_b_pressed = False
                    
                    # 检查是否同时按下低音C和高音B
                    if low_c_pressed and high_b_pressed:
                        print("Low C and High B pressed simultaneously. Exiting...")
                        sys.exit()  # 退出程序
                    while True:  # First loop
                        if detect_sequence(ports_input, 60, 1):  # Detecting CCC
                            print("Sequence C detected, entering second loop")
                            new_status = 2
                            self.status_changed.emit(new_status)
                            recording_started = False
                            recording_finished = False 
                            while True:  # Second loop
                                new_status = 2
                                self.status_changed.emit(new_status)
                                recording = []
                                recording_started = False
                                print("Waiting for start sequence...")
                                recording_started = False
                                recording_finished = False
                                while not recording_finished:
                                    for msg in ports_input.iter_pending():
                                        if not recording_started:
                                            # Start recording if the start sequence is detected
                                            if detect_notes_sequence(ports_input):
                                                new_status = 3
                                                self.status_changed.emit(new_status)
                                                print("Start sequence detected, beginning recording.")
                                                recording_started = True
                                        else:
                                            # Add messages to recording if recording has started
                                            recording.append(msg)
                                            # Stop recording if the end sequence is detected
                                            if detect_reverse_notes_sequence(ports_input):
                                                print("End sequence detected, stopping recording.")
                                                recording_finished = True
                                                break
                                        if recording_finished:
                                            break 
                                recorded_notes = recording
                                print("Recorded")
                                new_status = 4
                                self.status_changed.emit(new_status)
                                midi_file = MidiFile()
                                track = MidiTrack()
                                midi_file.tracks.append(track)
                                for msg in recorded_notes:
                                    track.append(msg)
                                midi_file.save(midi_file_input)
                                gfs.process_midi(midi_file_input,midi_file_output,tokenizer,vocabulary_json_file)
                                print("prcess completed")
                                print("Recording saved to 'input.mid'")
                                new_status = 5
                                self.status_changed.emit(new_status)
                                if detect_sequence(ports_input, 60, 3):  # Detecting CCC
                                    new_status = 6
                                    self.status_changed.emit(new_status)
                                    print("Sequence CCC detected, playing 'output.mid'")
                                    try:
                                        play_midi_file(port, 'output.mid', 10)
                                    except FileNotFoundError:
                                        print("File 'output.mid' not found.")

                                    if detect_sequence(ports_input, 71, 3):  # Detecting BBB
                                        print("Sequence BBB detected, returning to first loop")
                                        new_status = 1
                                        self.status_changed.emit(new_status)
                                        break  # Exit the second loop

if __name__ == '__main__':
    app = QApplication(sys.argv)
    manager = StatusManager(0)  # 初始化状态管理器，状态设置为0

    def on_status_changed(status):
        manager.changeStatus(status)

    status_thread = GenerateThread()
    status_thread.status_changed.connect(on_status_changed)
    status_thread.start()

    sys.exit(app.exec_())