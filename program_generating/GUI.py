import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.imageLabel = QLabel(self)  # 创建QLabel作为图像的容器
        self.initUI()

    def initUI(self):
        self.setFixedSize(2560, 1000)  # 更新窗口大小
        self.setStyleSheet("background-color: black;")
        self.setWindowFlags(Qt.FramelessWindowHint)
        pixmap = QPixmap('shared/resources/pics/face.svg')  # 加载图像
        self.imageLabel.setPixmap(pixmap)  # 设置QLabel的图像
        self.updateImagePosition()  # 更新图像位置和大小

    def updateImagePosition(self):
        windowWidth = self.width()
        windowHeight = self.height()
        pixmap = self.imageLabel.pixmap().scaled(windowWidth, int(windowHeight * 0.6), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.adjustSize()
        self.imageLabel.move((windowWidth - self.imageLabel.width()) // 2, int(windowHeight * 0.1))


    def resizeEvent(self, event):
        self.updateImagePosition()  # 窗口大小改变时更新图像位置和大小

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.close()  # 更新为Ctrl + S关闭窗口

def main():
    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
