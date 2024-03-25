import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setFixedSize(1920, 500)  # 设置窗口大小为1920x500
        self.setStyleSheet("background-color: black;")  # 设置背景颜色为黑色
        self.setWindowFlags(Qt.FramelessWindowHint)  # 设置无边框窗口

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
            self.close()  # 当按下Ctrl + V时关闭窗口

def main():
    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
