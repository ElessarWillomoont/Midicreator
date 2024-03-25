from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import os
import time

WIDTH = 1920
HEIGHT = 1080
STATUS = 0

class BaseStatusWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(WIDTH, HEIGHT)
        self.setWindowTitle('SVG状态显示')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框

    def loadAndDisplaySVG(self, svg_paths):
        layout = QVBoxLayout()
        self.setLayout(layout)

        for path in svg_paths:
            if os.path.exists(path):
                svg_widget = QSvgWidget(path)
                layout.addWidget(svg_widget)
                # 根据需要调整每个SVG Widget的大小和位置
                # svg_widget.setGeometry(...) 

class StatusWindow0(BaseStatusWindow):
    def __init__(self):
        super().__init__()
        svg_paths = ['/resource/pics/status0/face_0.svg', '/resource/pics/status0/words_0.svg']
        self.loadAndDisplaySVG(svg_paths)
        # 其他状态特有的初始化可以在这里进行

class StatusWindow1(BaseStatusWindow):
    def __init__(self):
        super().__init__()
        svg_paths = ['/resource/pics/status0/face_0.svg', '/resource/pics/status0/words_0.svg']
        self.loadAndDisplaySVG(svg_paths)
        # 其他状态特有的初始化可以在这里进行

class StatusWindow2(BaseStatusWindow):
    def __init__(self):
        super().__init__()
        svg_paths = ['/resource/pics/status0/face_0.svg', '/resource/pics/status0/words_0.svg']
        self.loadAndDisplaySVG(svg_paths)
        # 其他状态特有的初始化可以在这里进行

class StatusWindow3(BaseStatusWindow):
    def __init__(self):
        super().__init__()
        svg_paths = ['/resource/pics/status0/face_0.svg', '/resource/pics/status0/words_0.svg']
        self.loadAndDisplaySVG(svg_paths)
        # 其他状态特有的初始化可以在这里进行

class StatusWindow4(BaseStatusWindow):
    def __init__(self):
        super().__init__()
        svg_paths = ['/resource/pics/status0/face_0.svg', '/resource/pics/status0/words_0.svg']
        self.loadAndDisplaySVG(svg_paths)
        # 其他状态特有的初始化可以在这里进行

class StatusWindow5(BaseStatusWindow):
    def __init__(self):
        super().__init__()
        svg_paths = ['/resource/pics/status0/face_0.svg', '/resource/pics/status0/words_0.svg']
        self.loadAndDisplaySVG(svg_paths)
        # 其他状态特有的初始化可以在这里进行

class StatusManager:
    def __init__(self, initial_status=0):
        self.status_windows = {
            0: StatusWindow0,
            1: StatusWindow1,
            2: StatusWindow2,
            3: StatusWindow3,
            4: StatusWindow4,
            5: StatusWindow5,
            # 为其他状态添加窗口类...
        }
        self.current_window = self.status_windows[initial_status]()

    def changeStatus(self, status):
        if self.current_window:
            self.current_window.close()
        window_class = self.status_windows.get(status, BaseStatusWindow)
        self.current_window = window_class()
        self.current_window.show()

class StatusThread(QThread):
    status_changed = pyqtSignal(int)

    def run(self):
        while True:
            try:
                new_status = int(input("请输入status的值 (0-5): "))
                if 0 <= new_status <= 5:
                    self.status_changed.emit(new_status)
            except ValueError:
                pass
            time.sleep(0.1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    manager = StatusManager(0)  # 初始化状态管理器，状态设置为0

    def on_status_changed(status):
        manager.changeStatus(status)

    status_thread = StatusThread()
    status_thread.status_changed.connect(on_status_changed)
    status_thread.start()

    sys.exit(app.exec_())
