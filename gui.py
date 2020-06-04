import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from core import Core
from view import View


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.file_name = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # self.openFileNameDialog()
        self.openFileNamesDialog()
        # self.saveFileDialog()

        self.show()

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;Python Files (*.py)", options=options)
        if files:
            self.files = files
        self.close()




    app = QApplication(sys.argv)
    ex = App()
    # sys.exit(app.exec_())

    print(ex.files)

    view = View()

    for fl in ex.files:
        folder = fl.split('/')[:-1]
        folder = '/'.join(folder) + '/'

        file = fl.split('/')[-1]
        file = file.split('.')[0]

        core = Core(folder, file)
        core.k = 10

        view.add_core(core)
    view.show()
