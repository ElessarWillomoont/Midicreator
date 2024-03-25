from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsScene
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QTransform
from PyQt5.QtSvg import QSvgWidget, QSvgRenderer, QGraphicsSvgItem
import sys
import time

WIDTH = 2560
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
class StatusWindow0(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('状态 0')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.initUI()

    def initUI(self):
        # 加载SVG文件
        face_svg_path = 'shared/resources/pics/status0/face_0.svg'
        words_svg_path = 'shared/resources/pics/status0/words_0.svg'
        self.face_svg = QSvgWidget(face_svg_path, self)
        self.words_svg = QSvgWidget(words_svg_path, self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        # 只根据屏幕高度调整大小并保持原始长宽比
        vertical_space = HEIGHT * 0.1  # 顶部和底部的空间
        total_height_available = HEIGHT - (vertical_space * 3)  # 总可用高度，减去顶部、底部和两个SVG之间的空间

        # 分配高度给face_svg和words_svg
        face_height = total_height_available * 0.9  # 假设face_svg使用60%的可用高度
        words_height = total_height_available * 0.1  # 剩余40%给words_svg

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
        painter.setBrush(QColor(0, 0, 0))  # 绘制黑色背景
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow1(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('状态 1')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.initUI()

    def initUI(self):
        # 加载SVG文件，确保路径正确
        self.face_svg = QSvgWidget('shared/resources/pics/status1/face_1.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status1/keys_1.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status1/words_1.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  # 顶部空间
        vertical_space_bottom = HEIGHT * 0.1  # 底部空间
        vertical_space_between = HEIGHT * 0.05  # 两个SVG之间的空间

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # 假设的大小比例，可以根据需求调整
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
        painter.setBrush(QColor(0, 0, 0))  # 黑色背景
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow2(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('状态 2')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.initUI()

    def initUI(self):
        # 加载SVG文件，确保路径正确
        self.face_svg = QSvgWidget('shared/resources/pics/status2/face_2.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status2/keys_2.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status2/words_2.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  # 顶部空间
        vertical_space_bottom = HEIGHT * 0.1  # 底部空间
        vertical_space_between = HEIGHT * 0.05  # 两个SVG之间的空间

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # 假设的大小比例，可以根据需求调整
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
        painter.setBrush(QColor(0, 0, 0))  # 黑色背景
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow3(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('状态 3')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.initUI()

    def initUI(self):
        # 加载SVG文件，确保路径正确
        self.face_svg = QSvgWidget('shared/resources/pics/status3/face_3.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status3/keys_3.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status3/words_3.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  # 顶部空间
        vertical_space_bottom = HEIGHT * 0.1  # 底部空间
        vertical_space_between = HEIGHT * 0.05  # 两个SVG之间的空间

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # 假设的大小比例，可以根据需求调整
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
        painter.setBrush(QColor(0, 0, 0))  # 黑色背景
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow4(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('状态 4')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.initUI()
        self.startAnimation()

    def initUI(self):
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene, self)
        self.view.setGeometry(0, 0, WIDTH, HEIGHT)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent;")

        # 加载 SVG 文件
        self.face_svg = self.createSvgItem('shared/resources/pics/status4/face_4.svg')
        self.load_ring_svg = self.createSvgItem('shared/resources/pics/status4/load_ring.svg')
        self.world_svg = self.createSvgItem('shared/resources/pics/status4/words_4.svg')

        self.positionSvgWidgets()

    def createSvgItem(self, file_path):
        svg_item = QGraphicsSvgItem(file_path)
        self.scene.addItem(svg_item)
        return svg_item

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  # 顶部空间
        vertical_space_bottom = HEIGHT * 0.1  # 底部空间
        vertical_space_between = HEIGHT * 0.05  # 两个SVG之间的空间

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        face_height = total_height_available * 0.3
        keys_height = total_height_available * 0.5
        words_height = total_height_available * 0.2

        self.scaleAndPositionSvg(self.face_svg, face_height, vertical_space_top)
        self.scaleAndPositionSvg(self.load_ring_svg, keys_height, vertical_space_top + face_height + vertical_space_between)
        self.scaleAndPositionSvg(self.world_svg, words_height, vertical_space_top + face_height + keys_height + vertical_space_between * 2)

    def scaleAndPositionSvg(self, svg_item, target_height, top_margin):
        original_size = svg_item.boundingRect().size()
        scale_factor = target_height / original_size.height()
        new_width = original_size.width() * scale_factor

        # 计算新的x位置以便居中显示
        new_x = (WIDTH - new_width) / 2

        # 应用缩放
        svg_item.setScale(scale_factor)
        # 设置位置
        svg_item.setPos(new_x, top_margin)


    def startAnimation(self):
        # 初始化旋转角度
        self.rotation_angle = 0
        # 设置load_ring的旋转动画
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.rotateLoadRing)
        self.animation_timer.start(10)  # 更新间隔为10毫秒，以实现平滑动画

    def rotateLoadRing(self):
        # 更新旋转角度
        self.rotation_angle = (self.rotation_angle + 1) % 360

        # 获取加载环的原始边界矩形（考虑缩放）
        originalRect = self.load_ring_svg.renderer().defaultSize()
        scale_factor = self.load_ring_svg.scale()  # 获取当前应用的缩放因子
        scaledWidth = originalRect.width() * scale_factor
        scaledHeight = originalRect.height() * scale_factor

        # 计算缩放后的中心点
        centerX = scaledWidth / 2
        centerY = scaledHeight / 2

        # 由于QGraphicsSvgItem原点在左上角，需要先移动原点到图形中心，旋转后再移回
        transform = QTransform()
        transform.translate(centerX, centerY)  # 平移到中心
        transform.rotate(self.rotation_angle)  # 旋转
        transform.translate(-centerX, -centerY)  # 平移回原点

        self.load_ring_svg.setTransform(transform)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(0, 0, 0))  # 黑色背景
        painter.drawRect(self.rect())

    def showEvent(self, event):
        # 居中显示窗口
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow5(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('状态 5')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.initUI()

    def initUI(self):
        # 加载SVG文件，确保路径正确
        self.face_svg = QSvgWidget('shared/resources/pics/status5/face_5.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status5/keys_5.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status5/words_5.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  # 顶部空间
        vertical_space_bottom = HEIGHT * 0.1  # 底部空间
        vertical_space_between = HEIGHT * 0.05  # 两个SVG之间的空间

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # 假设的大小比例，可以根据需求调整
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
        painter.setBrush(QColor(0, 0, 0))  # 黑色背景
        painter.drawRect(self.rect())

    def showEvent(self, event):
        screen = QCoreApplication.instance().primaryScreen().geometry()
        self.move((screen.width() - WIDTH) // 2, (screen.height() - HEIGHT) // 2)

class StatusWindow6(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.setWindowTitle('状态 5')
        self.setAttribute(Qt.WA_TranslucentBackground)  # 透明背景
        self.setWindowFlags(Qt.FramelessWindowHint)  # 无边框
        self.initUI()

    def initUI(self):
        # 加载SVG文件，确保路径正确
        self.face_svg = QSvgWidget('shared/resources/pics/status6/face_6.svg', self)
        self.keys_svg = QSvgWidget('shared/resources/pics/status6/keys_6.svg', self)
        self.words_svg = QSvgWidget('shared/resources/pics/status6/words_6.svg', self)
        self.positionSvgWidgets()

    def positionSvgWidgets(self):
        vertical_space_top = HEIGHT * 0.1  # 顶部空间
        vertical_space_bottom = HEIGHT * 0.1  # 底部空间
        vertical_space_between = HEIGHT * 0.05  # 两个SVG之间的空间

        total_height_available = HEIGHT - vertical_space_top - vertical_space_bottom - vertical_space_between * 2  # 总可用高度

        # 假设的大小比例，可以根据需求调整
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
        painter.setBrush(QColor(0, 0, 0))  # 黑色背景
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
                new_status = int(input("请输入status的值 (0-6): "))
                if 0 <= new_status <= 6:
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
