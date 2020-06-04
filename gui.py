from PyQt5 import QtWidgets, QtGui, QtCore

import sys

from core import Core
from view import View


class IntroForm(QtWidgets.QMainWindow):

    def __init__(self, **kwargs):
        super(IntroForm, self).__init__(**kwargs)
        self.setWindowTitle('SPRI browser')
        self.setMinimumSize(300, 400)

        form = QtWidgets.QWidget()
        layout_form = QtWidgets.QVBoxLayout()
        form.setLayout(layout_form)
        self.setCentralWidget(form)

        self.infoLayout = QtWidgets.QHBoxLayout()
        layout_form.addLayout(self.infoLayout)
        self.infoLayout.addWidget(QtWidgets.QLabel("Choose the files to browse:"))
        self.chooseButton = QtWidgets.QPushButton("Find", self)
        self.infoLayout.addWidget(self.chooseButton)



        self.show()

    def setup(self):
        self.open_form = root.open_from
        # self.chooseButton.clicked(self.open_form)

class OpenForm(QtWidgets.QFileDialog):

    def __init__(self, **kwargs):
        super(OpenForm, self).__init__(**kwargs)
        pass

    def setup(self):
        pass


class App(QtWidgets.QApplication):

    def __init__(self):
        super(App, self).__init__(sys.argv)

    def build(self):
        self.intro_form = IntroForm()
        self.open_from = OpenForm()

        self.intro_form.setup()
        self.open_from.setup()
        sys.exit(self.exec_())


root = App()
root.build()


# for fl in ex.files:
#     folder = fl.split('/')[:-1]
#     folder = '/'.join(folder) + '/'
#
#     file = fl.split('/')[-1]
#     file = file.split('.')[0]
#
#     core = Core(folder, file)
#     core.k = 10
#
#     view.add_core(core)
# view.show()
