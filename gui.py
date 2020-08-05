import sys
import os
import re

from PyQt5.QtWidgets import *
from PyQt5.Qt import QVBoxLayout, QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from core import Core
from view_pyqt import View


def BoolFromCheckBox(value):
    if value.checkState() == 0:
        return False
    else:
        return True


class ErrorDialog(QDialog):

    def __init__(self, error_message, *args, **kwargs):
        super(ErrorDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Error")

        QBtn = QDialogButtonBox.Ok

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()

        label = QLabel('An error has occured:')
        label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(label)

        label_err = QLabel(error_message)
        label_err.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(label_err)

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def accept(self):
        self.close()


class PlotWindow(QtWidgets.QMainWindow):

    def __init__(self, canvas, *args, **kwargs):
        super(PlotWindow, self).__init__(*args, **kwargs)
        toolbar = NavigationToolbar(canvas, self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.img_window = None
        self.plot_window = None

        self.file = str()
        self.folder = str()

        self.setWindowTitle("SPRI Data Viewer")
        self.setWindowIcon(QIcon('icons/film.png'))

        self.open_button = QPushButton(QIcon('icons/folder-open.png'), 'Open file')
        font = self.open_button.font()
        font.setPointSize(15)

        self.open_button.setStatusTip(
            'Open one of the files from the desired measurement. In this stage, the specific channel does not matter. ')
        self.open_button.setFont(font)
        self.open_button.clicked.connect(self.OpenButtonClick)

        self.file_name_label = QLabel()

        self.orientation_checkbox = QCheckBox('Horizontal layout')
        self.orientation_checkbox.clicked.connect(self.RefreshOrientationInfo)
        self.orientation_checkbox.setChecked(True)

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
        label = QLabel('(takes a while)')
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
        self.statusBar().setMinimumSize(400, 40)
        self.statusBar().setStyleSheet("border :1px solid gray;")
        self.setCentralWidget(widget)

    def ProcessPath(self, path):
        splitted = path.split('/')
        file = splitted[-1].split('.')[0]

        if not re.search('^raw.*[1-4]$', file):
            dlg = ErrorDialog('File name has a wrong format.', self)
            dlg.exec_()
            return False

        self.file = file[:-2]
        self.folder = '/'.join(splitted[:-1]) + '/'

        for i in range(4):
            if os.path.isfile(self.folder + self.file + '_{}.tsv'.format(i + 1)):
                self.channel_checkbox_list[i].setDisabled(False)
            else:
                self.channel_checkbox_list[i].setDisabled(True)
                self.channel_checkbox_list[i].setChecked(False)
        return True

    def RefreshSliderInfo(self):
        self.slider_info.setText(str(self.k_slider.value()))

    def RefreshOrientationInfo(self):
        if self.orientation_checkbox.checkState() == 0:
            self.orientation_checkbox.setText('Vertical layout')
        else:
            self.orientation_checkbox.setText('Horizontal layout')

    def OpenButtonClick(self, s):
        dlg = QFileDialog(self)

        if self.ProcessPath(dlg.getOpenFileName()[0]):
            self.file_name_label.setText('folder path: {}\nfile name: {}'.format(self.folder, self.file))
            self.build_button.setDisabled(False)

    def BuildButtonClick(self, s):
        view = View()

        core_list = []
        for i, channel in enumerate(self.channel_checkbox_list):
            if channel.checkState() == 2:
                core = Core(self.folder, self.file + '_{}'.format(i + 1))
                core.k = self.k_slider.value()
                core_list.append(core)
                view.add_core(core)

        view.orientation = BoolFromCheckBox(self.orientation_checkbox)

        canvas_img = view.show_img()
        self.img_window = PlotWindow(canvas_img)
        self.img_window.show()

        canvas_img.setFocusPolicy(QtCore.Qt.ClickFocus)
        canvas_img.setFocus()

        chosen_plots = [
            BoolFromCheckBox(self.spr_checkbox),
            BoolFromCheckBox(self.int_checkbox),
            BoolFromCheckBox(self.nint_checkbox),
            BoolFromCheckBox(self.std_checkbox)
        ]

        print(chosen_plots)
        for i, core in enumerate(view.core_list):
            print('channel {}.'.format(i))
            if chosen_plots[1]:
                core.make_intensity_raw()

            if chosen_plots[2]:
                core.make_intensity_int()

            if chosen_plots[3]:
                core.make_std_int()

        canvas_plot = view.show_plots(chosen_plots)

        self.plot_window = PlotWindow(canvas_plot)
        self.plot_window.show()

        canvas_plot.setFocusPolicy(QtCore.Qt.ClickFocus)
        canvas_plot.setFocus()


app = QApplication(sys.argv)

window = MainWindow()
window.show()
app.exec_()
