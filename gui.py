import sys
import os
import re
import traceback

from PyQt5.QtWidgets import *
from PyQt5.Qt import QVBoxLayout, QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets

import matplotlib
from scipy.ndimage import gaussian_filter

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import tools as tl
import global_var as gv
from core import Core
from view_pyqt import View


def BoolFromCheckBox(value):
    if value.checkState() == 0:
        return False
    else:
        return True


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


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # definitions of all the active widgets:
        self.img_window = None
        self.plot_window = None

        self.view = None
        self.file = str()
        self.folder = str()
        self.chosen_plots = []
        self.width = []
        self.height = []
        self.channels = []
        self.ets = None
        self.avg = None
        self.frame_time = None
        self.num_of_frames = None

        self.loading_window = None
        self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.setWindowTitle("Crnka")
        self.setWindowIcon(QIcon('icons/cat-icon-2.png'))

        self.open_button = QPushButton(QIcon('icons/folder-open.png'), 'Open')
        font = self.open_button.font()
        font.setPointSize(14)

        self.open_button.setStatusTip(
            'Open any raw file from the desired measurement.')
        self.open_button.setFont(font)
        self.open_button.clicked.connect(self.OpenButtonClick)

        self.file_name_label = QLabel('folder: {}\nfile: {}'.format(self.folder, self.file))

        self.orientation_checkbox = QCheckBox('Horizontal layout')
        self.orientation_checkbox.clicked.connect(self.RefreshOrientationInfo)
        self.orientation_checkbox.setChecked(True)

        self.channel_checkbox_list = []
        for i in range(1, 5):
            self.channel_checkbox_list.append(QCheckBox('channel {}'.format(i)))

        self.slider_k = QSlider(Qt.Horizontal)
        self.slider_k.setStatusTip(
            'Number of integrated frames. Recommended value is 10 for 10 fps.')
        self.slider_k.setMinimum(1)
        self.slider_k.setMaximum(100)
        self.slider_k.setSingleStep(1)
        self.slider_k.setValue(10)
        self.slider_k.valueChanged.connect(self.RefreshSliderInfo)

        self.slider_k_info = QLabel('10')

        self.spr_checkbox = QCheckBox('SPR')
        self.spr_checkbox.setChecked(True)
        self.int_checkbox = QCheckBox('intensity')
        self.int_checkbox.setStatusTip('Takes a while')
        self.nint_checkbox = QCheckBox('norm. int.')
        self.nint_checkbox.setStatusTip('Takes a while')
        self.std_checkbox = QCheckBox('std')
        self.std_checkbox.setStatusTip('Takes a while')

        self.filter_gauss_checkbox = QCheckBox('gaussian')
        self.filter_gauss_checkbox.setChecked(False)
        self.filter_gauss_checkbox.clicked.connect(self.RunFilterGaussian)
        self.slider_gauss = QSlider(Qt.Horizontal)
        self.slider_gauss.setStatusTip(
            'Number of integrated frames. Recommended value is 10 for 10 fps.')
        self.slider_gauss.setMinimum(0)
        self.slider_gauss.setMaximum(50)
        self.slider_gauss.setSingleStep(1)
        self.slider_gauss.setValue(10)
        self.slider_gauss_info = QLabel('1')
        self.slider_gauss.valueChanged.connect(self.RefreshSliderGaussInfo)

        self.filter_fourier_checkbox = QCheckBox('fourier')
        self.filter_fourier_checkbox.setStatusTip('short pass')
        self.filter_fourier_checkbox.setChecked(False)
        self.filter_fourier_checkbox.clicked.connect(self.RunFilterFourier)
        self.slider_fourier = QSlider(Qt.Horizontal)
        self.slider_fourier.setStatusTip(
            'Number of integrated frames. Recommended value is 10 for 10 fps.')
        self.slider_fourier.setMinimum(0)
        self.slider_fourier.setMaximum(100)
        self.slider_fourier.setSingleStep(1)
        self.slider_fourier.setValue(10)
        self.slider_fourier_info = QLabel('10')
        self.slider_fourier.valueChanged.connect(self.RefreshSliderFourierInfo)

        self.build_button = QPushButton(QIcon('icons/arrow.png'), 'Build')
        self.build_button.setStatusTip('Builds the view of the data. It usually takes a while.')
        self.build_button.setFont(font)
        self.build_button.clicked.connect(self.BuildButtonClick)
        self.build_button.setDisabled(True)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(200, 80, 250, 20)
        self.progress_bar.setVisible(False)
        # self.progress_bar.setValue(50)

        self.info = QLabel()
        self.info.setVisible(False)
        # toolbar

        toolbar = QToolBar("Main toolbar")
        self.addToolBar(toolbar)

        self.tool_file_info = QAction("File info", self)
        self.tool_file_info.setStatusTip("Info about loaded file.")
        self.tool_file_info.triggered.connect(self.fileInfo)
        toolbar.addAction(self.tool_file_info)
        self.tool_file_info.setDisabled(True)

        self.tool_help = QAction("Help", self)
        self.tool_help.setStatusTip("[help]")
        self.tool_help.triggered.connect(self.toolbarHelp)
        toolbar.addAction(self.tool_help)

        # layout:

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
        slider_layout.addWidget(self.slider_k)
        slider_layout.addWidget(self.slider_k_info)
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

        label = QLabel('-- Image filters --')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        "fourier"
        fourier_layout = QHBoxLayout()
        fourier_layout.addWidget(self.filter_fourier_checkbox)
        fourier_layout.addWidget(self.slider_fourier)
        fourier_layout.addWidget(self.slider_fourier_info)

        layout.addLayout(fourier_layout)

        "gauss"
        gauss_layout = QHBoxLayout()
        gauss_layout.addWidget(self.filter_gauss_checkbox)
        gauss_layout.addWidget(self.slider_gauss)
        gauss_layout.addWidget(self.slider_gauss_info)

        layout.addLayout(gauss_layout)

        layout.addWidget(self.info)
        layout.addWidget(self.progress_bar)

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
            OKDialog('Error', 'File name has a wrong format.', self)
            return False

        self.file = file[:-2]
        self.folder = '/'.join(splitted[:-1]) + '/'
        self.width = []
        self.height = []
        self.channels = []

        info_done = False
        for i in range(4):
            if os.path.isfile(self.folder + self.file + '_{}.tsv'.format(i + 1)):
                self.channel_checkbox_list[i].setDisabled(False)
                if not info_done:
                    w, h, self.frame_time, self.avg, self.num_of_frames, self.ets = tl.read_file_info(
                        self.folder + self.file + '_{}'.format(i + 1))
                    info_done = True

                self.channels.append(i + 1)
                self.width.append(w)
                self.height.append(h)
            else:
                self.channel_checkbox_list[i].setDisabled(True)
        return True

    def toolbarHelp(self):
        OKDialog('help', gv.HELP, self)

    def fileInfo(self):
        fi = str()
        fi += 'file name: {}\n'.format(self.file)
        fi += 'directory: {}\n'.format(self.folder)
        fi += '-' * 40 + '\n'
        fi += 'number of frames: {}\n'.format(self.num_of_frames)
        fi += 'duration: {:.1f} s\n'.format(self.num_of_frames * self.frame_time)
        fi += '\n'
        fi += 'ETS: {:.1f} ms\n'.format(self.ets * 1e3)
        fi += 'AVG: {} \n'.format(self.avg)
        fi += 'frame time: {:.4f} s\n'.format(self.frame_time)
        fi += 'frame rate: {:.1f} fps\n\n'.format(1 / self.frame_time)
        fi += 'channel\twidth\theight\n'
        for c, w, h in zip(self.channels, self.width, self.height):
            fi += '{} \t{} \t{}\n'.format(c, w, h)

        OKDialog('file info', fi, self)

    def RefreshSliderInfo(self):
        self.slider_k_info.setText(str(self.slider_k.value()))

    def RefreshSliderGaussInfo(self):
        self.slider_gauss_info.setText(str(self.slider_gauss.value() / 10))
        self.RunFilterGaussian()

    def RefreshSliderFourierInfo(self):
        self.slider_fourier_info.setText(str(self.slider_fourier.value()))
        self.RunFilterFourier()

    def RefreshOrientationInfo(self):
        if self.orientation_checkbox.checkState() == 0:
            self.orientation_checkbox.setText('Vertical layout')
        else:
            self.orientation_checkbox.setText('Horizontal layout')

    def RunFilterGaussian(self):
        if not self.filter_gauss_checkbox.isChecked():
            for core in self.view.core_list:
                try:
                    del core.postprocessing['b_gauss']
                except KeyError:
                    print('posprocessing key not found')
        else:

            if self.view is None:
                OKDialog('error', 'no image data to work on')
                return

            else:
                for core in self.view.core_list:
                    core.postprocessing['b_gauss'] = lambda img: gaussian_filter(img, self.slider_gauss.value() / 10)
        dict(sorted(core.postprocessing.items()))
        self.view.draw()

    def RunFilterFourier(self):
        if not self.filter_fourier_checkbox.isChecked():
            for core in self.view.core_list:
                try:
                    del core.postprocessing['a_fourier']
                except KeyError:
                    print('posprocessing key not found')
        else:

            if self.view is None:
                OKDialog('error', 'no image data to work on')
                return

            else:
                for core in self.view.core_list:
                    core.postprocessing['a_fourier'] = lambda img: tl.fourier_filter(img, self.slider_fourier.value())
        dict(sorted(core.postprocessing.items()))
        self.view.draw()

    def OpenButtonClick(self, s):
        dlg = QFileDialog(self)

        if self.ProcessPath(dlg.getOpenFileName()[0]):
            self.file_name_label.setText('folder path: ... {}\nfile name: {}'.format(self.folder[-20:], self.file))
            self.build_button.setDisabled(False)
            self.tool_file_info.setDisabled(False)

            if self.width[0] < self.height[0]:
                self.orientation_checkbox.setChecked(True)
            else:
                self.orientation_checkbox.setChecked(False)

            for chch in self.channel_checkbox_list:
                chch.setChecked(False)

    def progress_fn(self, n):
        self.progress_bar.setValue(n)

    def thread_complete(self):
        # print(self.threadpool.activeThreadCount())
        if self.threadpool.activeThreadCount() == 0:
            # print('konec')
            if True in self.chosen_plots:
                canvas_plot = self.view.show_plots(self.chosen_plots)

                self.plot_window = PlotWindow(canvas_plot)
                self.plot_window.show()

                canvas_plot.setFocusPolicy(QtCore.Qt.ClickFocus)
                canvas_plot.setFocus()

            canvas_img = self.view.show_img()
            self.img_window = PlotWindow(canvas_img)
            self.img_window.show()

            canvas_img.setFocusPolicy(QtCore.Qt.ClickFocus)
            canvas_img.setFocus()
            self.progress_bar.setVisible(False)

    def BuildButtonClick(self, s):
        self.view = View()
        core_list = []
        for i, channel in enumerate(self.channel_checkbox_list):
            if channel.checkState() == 2:
                core = Core(self.folder, self.file + '_{}'.format(i + 1))
                core.k = self.slider_k.value()
                core_list.append(core)
                self.view.add_core(core)
        if len(self.view.core_list) == 0:
            OKDialog('error', 'no channels selected')
            return

        self.view.orientation = BoolFromCheckBox(self.orientation_checkbox)

        self.chosen_plots = [
            BoolFromCheckBox(self.spr_checkbox),
            BoolFromCheckBox(self.int_checkbox),
            BoolFromCheckBox(self.nint_checkbox),
            BoolFromCheckBox(self.std_checkbox)
        ]

        self.progress_bar.setVisible(True)

        # self.info.setVisible(True)
        # self.threadpool.stackSize(0)

        for i, core in enumerate(self.view.core_list):
            if self.chosen_plots[1]:
                worker = Worker(core.make_intensity_raw)
                worker.signals.finished.connect(self.thread_complete)
                worker.signals.progress.connect(self.progress_fn)
                self.threadpool.start(worker)
                self.threadpool.waitForDone(100000)

            if self.chosen_plots[2]:
                worker2 = Worker(core.make_intensity_int)
                worker2.signals.finished.connect(self.thread_complete)
                worker2.signals.progress.connect(self.progress_fn)
                self.threadpool.start(worker2)

            if self.chosen_plots[3]:
                worker3 = Worker(core.make_std_int)
                worker3.signals.finished.connect(self.thread_complete)
                worker3.signals.progress.connect(self.progress_fn)
                self.threadpool.start(worker3)

        if self.chosen_plots[0]:
            self.thread_complete()


app = QApplication(sys.argv)
app.setFont(QFont('Courier', 10))

window = MainWindow()
window.show()
app.exec_()
