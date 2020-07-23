from PyQt5.QtWidgets import *
from PyQt5.Qt import QVBoxLayout, QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFrame

import sys
import os
from core import Core
from view import View


class CustomDialog(QDialog):

    def __init__(self, *args, **kwargs):
        super(CustomDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("HELLO!")

        QBtn = QDialogButtonBox.Open | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.file_path = str()
        self.file = str()
        self.folder = str()

        self.setWindowTitle("My Awesome App")

        self.open_button = QPushButton(QIcon('icons/folder-open.png'), 'Open file')
        self.open_button.setStatusTip(
            'Open one of the files from the desired measurement. In this stage, the specific channel does not matter. ')
        font = self.open_button.font()
        font.setPointSize(15)
        self.open_button.setFont(font)
        self.open_button.clicked.connect(self.OpenButtonClick)

        self.file_name_label = QLabel()

        self.orientation_checkbox = QCheckBox('Change the orientation')
        self.orientation_checkbox.setStatusTip('Check to rotate the orientation by 90 degrees.')

        self.spr_checkbox = QCheckBox('SPR')
        self.spr_checkbox.setChecked(True)

        self.int_checkbox = QCheckBox('intensity')
        self.nint_checkbox = QCheckBox('norm. int.')
        self.std_checkbox = QCheckBox('std')


        self.channel_checkbox_list = []
        for i in range(1, 5):
            self.channel_checkbox_list.append(QCheckBox('channel {}'.format(i)))

        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setStatusTip(
            'Number of frames integrated. Higher numbers reduce the background noise, but worsen the time resolution. Recommended value is 10 for raw data with 10fps.')
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(100)
        self.k_slider.setSingleStep(1)
        self.k_slider.setValue(10)
        self.k_slider.valueChanged.connect(self.RefreshSliderInfo)

        self.slider_info = QLabel('10')

        self.build_button = QPushButton(QIcon('icons/arrow.png'), 'Build')
        self.build_button.setStatusTip('Builds the view of the data. It usually takes a while.')
        self.build_button.setFont(font)
        self.build_button.clicked.connect(self.BuildButtonClick)
        self.build_button.setDisabled(True)

        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.file_name_label)

        label = QLabel('-- Choose channels to be shown --')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        for channel in self.channel_checkbox_list:
            channel.setDisabled(True)
            layout.addWidget(channel)

        label = QLabel('-- Image settings --')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        layout.addWidget(self.orientation_checkbox)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel('Integration number:'))
        slider_layout.addWidget(self.k_slider)
        slider_layout.addWidget(self.slider_info)
        layout.addLayout(slider_layout)

        label = QLabel('-- Data to plot --')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        plot_layout = QGridLayout()
        plot_layout.addWidget(self.spr_checkbox, 0, 0)
        plot_layout.addWidget(self.int_checkbox, 1, 0)
        plot_layout.addWidget(self.nint_checkbox, 0, 1)
        plot_layout.addWidget(self.std_checkbox, 1, 1)
        layout.addLayout(plot_layout)


        layout.addWidget(self.build_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.setStatusBar(QStatusBar(self))
        self.setCentralWidget(widget)

    def ProcessPath(self, path):
        splitted = path.split('/')
        self.file = splitted[-1].split('.')[0][:-2]
        self.folder = '/'.join(splitted[:-1]) + '/'

        for i in range(4):
            if os.path.isfile(self.folder + self.file + '_{}.tsv'.format(i + 1)):
                self.channel_checkbox_list[i].setDisabled(False)
            else:
                self.channel_checkbox_list[i].setDisabled(True)

    def RefreshSliderInfo(self):
        self.slider_info.setText(str(self.k_slider.value()))

    def OpenButtonClick(self, s):
        dlg = QFileDialog(self)

        self.ProcessPath(dlg.getOpenFileName()[0])
        self.file_name_label.setText(self.file_path)
        self.build_button.setDisabled(False)

    def BuildButtonClick(self, s):
        core_list = []
        for i, channel in enumerate(self.channel_checkbox_list):
            if channel.checkState() == 2:
                core = Core(self.folder, self.file + '_{}'.format(i + 1))
                core.k = 10
                core_list.append(core)
        print('Core, successful')
        # view = View()
        # view.add_core(core)
        # view.show()


app = QApplication(sys.argv)

window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec_()
