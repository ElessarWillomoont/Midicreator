from PyQt5.QtCore import QThread, pyqtSignal
import mido
import time
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEvent, Qt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QEvent, Qt
import queue
    
class SimpleWindow(QMainWindow):
    def __init__(self, threads):
        super().__init__()
        self.threads = threads
        self.setWindowTitle("Simple Window")
        self.setGeometry(100, 100, 600, 400)  # X position, Y position, Width, Height

    def closeEvent(self, event):
        # 请求所有线程退出
        for thread in self.threads:
            thread.requestInterruption()  # 请求线程中断
            thread.quit()  # 请求线程退出
            thread.wait()  # 等待线程退出
        
        super().closeEvent(event)

class CustomApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        # Event filtering logic remains the same...
        return super().eventFilter(obj, event)

class MIDISignalThread(QThread):
    device_found = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.message_queue = queue.Queue()
        self.port_name_input = None
        self.port_name_output = None

    def run(self):
        ports_output = mido.get_output_names()
        ports_input = mido.get_input_names()
        self.port_name_output = [name for name in ports_output if 'Disklavier' in name]
        self.port_name_input = [name for name in ports_input if 'Disklavier' in name]
        if not self.port_name_output or not self.port_name_input:
            print("Disklavier piano not found.")
            self.device_found.emit(False)
            return
        self.device_found.emit(True)
        with mido.open_input(self.port_name_input[0]) as port_input:
            while not self.isInterruptionRequested():
                for msg in port_input.iter_pending():
                    self.message_queue.put(msg)
                time.sleep(0.01)

class GenerateThread(QThread):
    def __init__(self, message_queue):
        super().__init__()
        self.message_queue = message_queue

    def run(self):
        while not self.isInterruptionRequested():
            msg = self.message_queue.get()
            print(f"message in generate thread: {msg}")
            self.message_queue.task_done()

class ExitOnCAndBThread(QThread):
    def __init__(self, message_queue):
        super().__init__()
        self.message_queue = message_queue

    def run(self):
        while not self.isInterruptionRequested():
            msg = self.message_queue.get()
            print(f"message in exit thread: {msg}")
            self.message_queue.task_done()

def main():
    app = CustomApplication(sys.argv)
    
    midi_signal_thread = MIDISignalThread()
    message_queue = midi_signal_thread.message_queue
    
    generate_thread = GenerateThread(message_queue)
    exit_thread = ExitOnCAndBThread(message_queue)

    midi_signal_thread.start()
    generate_thread.start()
    exit_thread.start()

    window = SimpleWindow([midi_signal_thread, generate_thread, exit_thread])
    window.show()

    exit_code = app.exec_()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()