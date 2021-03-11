from PyQt5.QtWidgets import *
from PyQt5.Qt import QVBoxLayout, QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class OKDialog(QDialog):

    def __init__(self, title, message, *args, **kwargs):
        super(OKDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.Ok

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()

        label_err = QLabel(message)
        label_err.setWordWrap(True)
        # label_err.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(label_err)

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        self.exec_()

    def accept(self):
        self.close()


class PlotWindow(QtWidgets.QMainWindow):

    def __init__(self, canvas, *args, **kwargs):
        super(PlotWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle(canvas.view.core_list[0].folder + '/' + canvas.view.core_list[0].file[:-2])

        toolbar = NavigationToolbar(canvas, self)
        canvas.nav_toolbar = toolbar

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


class SelectWindow(QtWidgets.QMainWindow):

    def __init__(self, canvas, *args, **kwargs):
        super(SelectWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('Select NP pattern')

        toolbar = NavigationToolbar(canvas, self)
        canvas.nav_toolbar = toolbar

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


class LoadingWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(LoadingWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('loading')

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(20, 20, 250, 20)
        self.progress_bar.setValue(50)
        self.info = QLabel('Info:')

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.info)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)
