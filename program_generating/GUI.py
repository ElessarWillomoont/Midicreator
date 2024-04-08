from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsView, QGraphicsScene, QGraphicsScene
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QTransform
from PyQt5.QtSvg import QSvgWidget, QGraphicsSvgItem
import sys
import time
import os
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(script_path))
sys.path.append(parent_directory)
import shared.config as configue

# Constants for window dimensions
WIDTH = configue.WIDTH
HEIGHT = configue.HEIGHT

class BaseStatusWindow(QWidget):
    def __init__(self, status):
        super().__init__()
        self.status = status
        self.initUI()

    def initUI(self):
        self.setFixedSize(WIDTH, HEIGHT)
        self.setWindowTitle('SVG Status Display')
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background
        self.setWindowFlags(Qt.FramelessWindowHint)  # no outline

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setBrush(QColor(0, 0, 0))  # Set background color to black
        qp.drawRect(self.rect())
        
        # Set text color and font
        qp.setPen(QColor(255, 255, 255))
        qp.setFont(QFont('Arial', 48))
        
        # Draw the current status value
        text = f"status {self.status}"
        qp.drawText(self.rect(), Qt.AlignCenter, text)

# set a window class for each status
class StatusWindow0(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('Status 0')
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background
        self.setWindowFlags(Qt.FramelessWindowHint)  # no outline
        self.initUI()

    def initUI(self):
        # Load SVG files
        face_svg_path = 'shared/resources/pics/status0/face_0.svg'
        words_svg_path = 'shared/resources/pics/status0/words_0.svg'
        self.face_svg = QSvgWidget(face_svg_path, self)
        self.words_svg = QSvgWidget(words_svg_path, self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        # Adjust the size based on screen height while maintaining the aspect ratio
        vertical_space = HEIGHT * 0.1  # Space at the top and bottom
        total_height_available = HEIGHT - (vertical_space * 3)  # Total available height, subtracting space for the top, bottom, and between the two SVGs

        # Allocate height to face_svg and words_svg
        face_height = total_height_available * 0.9  # Assume face_svg uses 60% of the available height
        words_height = total_height_available * 0.1  # The remaining 40% for words_svg

        self.scaleAndPositionSvg(self.face_svg, face_height, vertical_space)
        self.scaleAndPositionSvg(self.words_svg, words_height, vertical_space * 2 + face_height)

    def scaleAndPositionSvg(self, svg_widget, target_height, top_margin):
        original_size = svg_widget.renderer().defaultSize()
        scale_factor = target_height / original_size.height()
        new_width = original_size.width() * scale_factor
        new_x = (WIDTH - new_width) / 2
        svg_widget.setGeometry(int(new_x), int(top_margin), int(new_width), int(target_height))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0))  # Draw black background
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow1(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('Status 1')
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background
        self.setWindowFlags(Qt.FramelessWindowHint)  # no outline
        self.initUI()

    def initUI(self):
        # Load SVG files, ensure paths are correct
        self.face_svg = QSvgWidget('shared/resources/pics/status1/face_1.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status1/keys_1.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status1/words_1.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  
        vertical_space_bottom = HEIGHT * 0.1  
        vertical_space_between = HEIGHT * 0.05  

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # Assumed size ratios, adjust as needed
        face_height_ratio = 0.35
        keys_height_ratio = 0.5
        words_height_ratio = 0.15

        face_height = total_height_available * face_height_ratio
        keys_height = total_height_available * keys_height_ratio
        words_height = total_height_available * words_height_ratio

        self.scaleAndPositionSvg(self.face_svg, face_height, vertical_space_top)
        self.scaleAndPositionSvg(self.keys_svg, keys_height, vertical_space_top + face_height + vertical_space_between)
        self.scaleAndPositionSvg(self.words_svg, words_height, vertical_space_top + face_height + keys_height + vertical_space_between * 2)

    def scaleAndPositionSvg(self, svg_widget, target_height, top_margin):
        original_size = svg_widget.renderer().defaultSize()
        scale_factor = target_height / original_size.height()
        new_width = original_size.width() * scale_factor
        new_x = (WIDTH - new_width) / 2
        svg_widget.setGeometry(int(new_x), int(top_margin), int(new_width), int(target_height))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0))  # Set background color to black
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow2(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('Status 2')
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background
        self.setWindowFlags(Qt.FramelessWindowHint)  # no outline
        self.initUI()

    def initUI(self):
        # Load SVG files, ensure paths are correct
        self.face_svg = QSvgWidget('shared/resources/pics/status2/face_2.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status2/keys_2.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status2/words_2.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  
        vertical_space_bottom = HEIGHT * 0.1  
        vertical_space_between = HEIGHT * 0.05  

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # Assumed size ratios, adjust as needed
        face_height_ratio = 0.4
        keys_height_ratio = 0.5
        words_height_ratio = 0.1

        face_height = total_height_available * face_height_ratio
        keys_height = total_height_available * keys_height_ratio
        words_height = total_height_available * words_height_ratio

        self.scaleAndPositionSvg(self.face_svg, face_height, vertical_space_top)
        self.scaleAndPositionSvg(self.keys_svg, keys_height, vertical_space_top + face_height + vertical_space_between)
        self.scaleAndPositionSvg(self.words_svg, words_height, vertical_space_top + face_height + keys_height + vertical_space_between * 2)

    def scaleAndPositionSvg(self, svg_widget, target_height, top_margin):
        original_size = svg_widget.renderer().defaultSize()
        scale_factor = target_height / original_size.height()
        new_width = original_size.width() * scale_factor
        new_x = (WIDTH - new_width) / 2
        svg_widget.setGeometry(int(new_x), int(top_margin), int(new_width), int(target_height))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0))  # Set background color to black
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow3(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('Status 3')
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background
        self.setWindowFlags(Qt.FramelessWindowHint)  # no outline
        self.initUI()

    def initUI(self):
        # Load SVG files, ensure paths are correct
        self.face_svg = QSvgWidget('shared/resources/pics/status3/face_3.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status3/keys_3.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status3/words_3.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  
        vertical_space_bottom = HEIGHT * 0.1  
        vertical_space_between = HEIGHT * 0.05  

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # Assumed size ratios, adjust as needed
        face_height_ratio = 0.3
        keys_height_ratio = 0.5
        words_height_ratio = 0.2

        face_height = total_height_available * face_height_ratio
        keys_height = total_height_available * keys_height_ratio
        words_height = total_height_available * words_height_ratio

        self.scaleAndPositionSvg(self.face_svg, face_height, vertical_space_top)
        self.scaleAndPositionSvg(self.keys_svg, keys_height, vertical_space_top + face_height + vertical_space_between)
        self.scaleAndPositionSvg(self.words_svg, words_height, vertical_space_top + face_height + keys_height + vertical_space_between * 2)

    def scaleAndPositionSvg(self, svg_widget, target_height, top_margin):
        original_size = svg_widget.renderer().defaultSize()
        scale_factor = target_height / original_size.height()
        new_width = original_size.width() * scale_factor
        new_x = (WIDTH - new_width) / 2
        svg_widget.setGeometry(int(new_x), int(top_margin), int(new_width), int(target_height))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0))  # Set background color to black
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow4(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('Status 4')
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background
        self.setWindowFlags(Qt.FramelessWindowHint)  # no outline
        self.initUI()
        self.startAnimation()

    def initUI(self):
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)
        self.view.setGeometry(0, 0, WIDTH, HEIGHT)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent;")

        self.face_svg = self.createSvgItem('shared/resources/pics/status4/face_4.svg')
        self.load_ring_svg = self.createSvgItem('shared/resources/pics/status4/load_ring.svg')
        self.world_svg = self.createSvgItem('shared/resources/pics/status4/words_4.svg')

        self.positionSvgWidgets()

    def createSvgItem(self, file_path):
        svg_item = QGraphicsSvgItem(file_path)
        self.scene.addItem(svg_item)
        return svg_item

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  
        vertical_space_bottom = HEIGHT * 0.1  
        vertical_space_between = HEIGHT * 0.05  

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        face_height = total_height_available * 0.4
        keys_height = total_height_available * 0.5
        words_height = total_height_available * 0.1

        self.scaleAndPositionSvg(self.face_svg, face_height, vertical_space_top)
        self.scaleAndPositionSvg(self.load_ring_svg, keys_height, vertical_space_top + face_height + vertical_space_between)
        self.scaleAndPositionSvg(self.world_svg, words_height, vertical_space_top + face_height + keys_height + vertical_space_between * 2)

    def scaleAndPositionSvg(self, svg_item, target_height, top_margin):
        original_size = svg_item.boundingRect().size()
        scale_factor = target_height / original_size.height()
        new_width = original_size.width() * scale_factor

        new_x = (WIDTH - new_width) / 2

        svg_item.setScale(scale_factor)
        svg_item.setPos(new_x, top_margin)


    def startAnimation(self):
        self.rotation_angle = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.rotateLoadRing)
        self.animation_timer.start(10) 

    def rotateLoadRing(self):
        self.rotation_angle = (self.rotation_angle + 1) % 360

        originalRect = self.load_ring_svg.renderer().defaultSize()
        scale_factor = self.load_ring_svg.scale() 
        scaledWidth = originalRect.width() * scale_factor
        scaledHeight = originalRect.height() * scale_factor

        centerX = scaledWidth / 2
        centerY = scaledHeight / 2

        transform = QTransform()
        transform.translate(centerX, centerY)  
        transform.rotate(self.rotation_angle)  
        transform.translate(-centerX, -centerY)  

        self.load_ring_svg.setTransform(transform)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0))  # Set background color to black
        painter.drawRect(self.rect())

    def showEvent(self, event):
        # 居中显示窗口
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow5(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('Status 5')
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background
        self.setWindowFlags(Qt.FramelessWindowHint)  # no outline
        self.initUI()

    def initUI(self):
        # Load SVG files, ensure paths are correct
        self.face_svg = QSvgWidget('shared/resources/pics/status5/face_5.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status5/keys_5.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status5/words_5.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  
        vertical_space_bottom = HEIGHT * 0.1  
        vertical_space_between = HEIGHT * 0.05  

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  

        # Assumed size ratios, adjust as needed
        face_height_ratio = 0.4
        keys_height_ratio = 0.5
        words_height_ratio = 0.1

        face_height = total_height_available * face_height_ratio
        keys_height = total_height_available * keys_height_ratio
        words_height = total_height_available * words_height_ratio

        self.scaleAndPositionSvg(self.face_svg, face_height, vertical_space_top)
        self.scaleAndPositionSvg(self.keys_svg, keys_height, vertical_space_top + face_height + vertical_space_between)
        self.scaleAndPositionSvg(self.words_svg, words_height, vertical_space_top + face_height + keys_height + vertical_space_between * 2)

    def scaleAndPositionSvg(self, svg_widget, target_height, top_margin):
        original_size = svg_widget.renderer().defaultSize()
        scale_factor = target_height / original_size.height()
        new_width = original_size.width() * scale_factor
        new_x = (WIDTH - new_width) / 2
        svg_widget.setGeometry(int(new_x), int(top_margin), int(new_width), int(target_height))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0))  # Set background color to black
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow6(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('Status 6')
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background
        self.setWindowFlags(Qt.FramelessWindowHint)  # no outline
        self.initUI()

    def initUI(self):
        # Load SVG files, ensure paths are correct
        self.face_svg = QSvgWidget('shared/resources/pics/status6/face_6.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status6/keys_6.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status6/words_6.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  
        vertical_space_bottom = HEIGHT * 0.1  
        vertical_space_between = HEIGHT * 0.05  

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # Assumed size ratios, adjust as needed
        face_height_ratio = 0.5
        keys_height_ratio = 0.3
        words_height_ratio = 0.2

        face_height = total_height_available * face_height_ratio
        keys_height = total_height_available * keys_height_ratio
        words_height = total_height_available * words_height_ratio

        self.scaleAndPositionSvg(self.face_svg, face_height, vertical_space_top)
        self.scaleAndPositionSvg(self.keys_svg, keys_height, vertical_space_top + face_height + vertical_space_between)
        self.scaleAndPositionSvg(self.words_svg, words_height, vertical_space_top + face_height + keys_height + vertical_space_between * 2)

    def scaleAndPositionSvg(self, svg_widget, target_height, top_margin):
        original_size = svg_widget.renderer().defaultSize()
        scale_factor = target_height / original_size.height()
        new_width = original_size.width() * scale_factor
        new_x = (WIDTH - new_width) / 2
        svg_widget.setGeometry(int(new_x), int(top_margin), int(new_width), int(target_height))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0))  # Set background color to black
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusManager:
    def __init__(self, initial_status=0):
        self.status_windows = {
            0: StatusWindow0,
            1: StatusWindow1,
            2: StatusWindow2,
            3: StatusWindow3,
            4: StatusWindow4,
            5: StatusWindow5,
            6: StatusWindow6,
        }
        self.current_window = None
        self.changeStatus(initial_status)

    def changeStatus(self, status):
        if self.current_window:
            self.current_window.close()
        window_class = self.status_windows.get(status, BaseStatusWindow)
        self.current_window = window_class()
        self.current_window.show()

class TestThread(QThread):
    status_changed = pyqtSignal(int)

    def run(self):
        while True:
            try:
                new_status = int(input("please input status index (0-6): "))
                if 0 <= new_status <= 6:
                    self.status_changed.emit(new_status)
            except ValueError:
                pass
            time.sleep(0.1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    manager = StatusManager(0)  # initialize status manager, set index to 0

    def on_status_changed(status):
        manager.changeStatus(status)

    status_thread = TestThread()
    status_thread.status_changed.connect(on_status_changed)
    status_thread.start()

    sys.exit(app.exec_())
