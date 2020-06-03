from PyQt5 import QtWidgets, QtGui
import sys

from forms import App


class Window(QtWidgets.QMainWindow):

    def __init__(self, **kwargs):
        super(Window, self).__init__(**kwargs)

        self.setWindowTitle('Calculator')
        self.init_gui()
        self.show()


    def init_gui(self):
        formular = QtWidgets.QWidget()
        formularLayout = QtWidgets.QVBoxLayout()
        formular.setLayout(formularLayout)

        boxLayout1 = QtWidgets.QHBoxLayout()
        boxLayout2 = QtWidgets.QHBoxLayout()

        formularLayout.addStretch()
        formularLayout.addLayout(boxLayout1)
        formularLayout.addLayout(boxLayout2)
        formularLayout.addStretch()

        self.vysledekLabel = QtWidgets.QLabel("0", self)
        self.vysledekLabel.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Black))
        self.cislo1Edit = QtWidgets.QLineEdit(self)
        self.cislo2Edit = QtWidgets.QLineEdit(self)
        self.vypoctiButton = QtWidgets.QPushButton("Výpočet", self)
        self.operatorComboBox = QtWidgets.QComboBox(self)

        self.operatorComboBox.addItem("+")
        self.operatorComboBox.addItem("-")
        self.operatorComboBox.addItem("/")
        self.operatorComboBox.addItem("*")

        boxLayout1.addWidget(self.cislo1Edit)
        boxLayout1.addWidget(self.operatorComboBox)
        boxLayout1.addWidget(self.cislo2Edit)
        boxLayout1.addWidget(self.vysledekLabel)
        boxLayout2.addWidget(self.vypoctiButton)

        self.setCentralWidget(formular)

        self.vypoctiButton.clicked.connect(self.vypocti)

    def vypocti(self):
        self.ex = App()
        path = self.ex.file_name
        self.ex.close()

        self.vysledekLabel.setText(path[0])




aplikace = QtWidgets.QApplication(sys.argv)
okno = Window()
sys.exit(aplikace.exec_())
