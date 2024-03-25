from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QFont, QPen
import sys
import time

WIDTH = 1920
HEIGHT = 1080

class BaseStatusWindow(QWidget):
    def __init__(self, status):
        super().__init__()
        self.status = status
        self.initUI()

    def initUI(self):
        self.setFixedSize(WIDTH, HEIGHT)
        self.setWindowTitle('SVG状态显示')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setBrush(QColor(0, 0, 0))  # 黑色背景
        qp.drawRect(self.rect())
        
        # 设置文本颜色和字体
        qp.setPen(QColor(255, 255, 255))
        qp.setFont(QFont('Arial', 48))
        
        # 绘制当前状态值
        text = f"状态 {self.status}"
        qp.drawText(self.rect(), Qt.AlignCenter, text)

# 为每个状态定义一个窗口类
class StatusWindow0(BaseStatusWindow):
    def __init__(self):
        super().__init__(0)

class StatusWindow1(BaseStatusWindow):
    def __init__(self):
        super().__init__(1)

class StatusWindow2(BaseStatusWindow):
    def __init__(self):
        super().__init__(2)
class StatusWindow3(BaseStatusWindow):
    def __init__(self):
        super().__init__(3)
class StatusWindow4(BaseStatusWindow):
    def __init__(self):
        super().__init__(4)
class StatusWindow5(BaseStatusWindow):
    def __init__(self):
        super().__init__(5)

# 以此类推，为其他状态定义更多窗口类...

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
        self.current_window = None
        self.changeStatus(initial_status)

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
