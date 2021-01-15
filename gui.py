import sys
import os
import re
import traceback

import cv2
from PyQt5.QtWidgets import *
from PyQt5.Qt import QVBoxLayout, QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets

import matplotlib
import scipy.signal
import numpy as np
from scipy.ndimage import gaussian_filter

matplotlib.use('Qt5Agg')

import tools as tl
import global_var as gv
from core import Core
from view_pyqt import View
from gui_windows import OKDialog, PlotWindow, LoadingWindow


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

        self.button_open = QPushButton(QIcon('icons/folder-open.png'), 'Open')
        font = self.button_open.font()
        font.setPointSize(14)

        self.button_open.setStatusTip(
            'Open any raw file from the desired measurement.')
        self.button_open.setFont(font)
        self.button_open.clicked.connect(self.OpenButtonClick)

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

        self.slider_downsample = QSlider(Qt.Horizontal)
        self.slider_downsample.setStatusTip(
            'N-times downsample the raw data in time.')
        self.slider_downsample.setMinimum(1)
        self.slider_downsample.setMaximum(100)
        self.slider_downsample.setSingleStep(1)
        self.slider_downsample.setValue(0)
        self.slider_downsample.valueChanged.connect(self.RefreshSliderDownSampleInfo)
        self.slider_downsample_info = QLabel('0')

        "Data processing"

        self.checkbox_1 = QCheckBox('SPR')
        self.checkbox_1.setChecked(True)
        self.checkbox_2 = QCheckBox('intensity')
        self.checkbox_2.setStatusTip('Takes a while')
        self.checkbox_3 = QCheckBox('norm. int.')
        self.checkbox_3.setStatusTip('Takes a while')
        self.checkbox_4 = QCheckBox('std')
        self.checkbox_4.setStatusTip('Takes a while')

        "Image filters"

        self.filter_gauss_checkbox = QCheckBox('gaussian')
        self.filter_gauss_checkbox.setChecked(False)
        self.filter_gauss_checkbox.clicked.connect(self.RunFilterGaussian)
        self.slider_gauss = QSlider(Qt.Horizontal)

        self.slider_gauss.setMinimum(0)
        self.slider_gauss.setMaximum(50)
        self.slider_gauss.setSingleStep(1)
        self.slider_gauss.setValue(10)
        self.slider_gauss_info = QLabel('1')
        self.slider_gauss.valueChanged.connect(self.RefreshSliderGaussInfo)

        self.filter_erode_checkbox = QCheckBox('erode')
        self.filter_erode_checkbox.setChecked(False)
        self.filter_erode_checkbox.clicked.connect(self.RunFilterErode)
        self.slider_erode = QSlider(Qt.Horizontal)

        self.slider_erode.setMinimum(0)
        self.slider_erode.setMaximum(50)
        self.slider_erode.setSingleStep(1)
        self.slider_erode.setValue(10)
        self.slider_erode_info = QLabel('1')
        self.slider_erode.valueChanged.connect(self.RefreshSliderErodeInfo)

        self.filter_dilate_checkbox = QCheckBox('dilate')
        self.filter_dilate_checkbox.setChecked(False)
        self.filter_dilate_checkbox.clicked.connect(self.RunFilterDilate)
        self.slider_dilate = QSlider(Qt.Horizontal)

        self.slider_dilate.setMinimum(0)
        self.slider_dilate.setMaximum(50)
        self.slider_dilate.setSingleStep(1)
        self.slider_dilate.setValue(10)
        self.slider_dilate_info = QLabel('1')
        self.slider_dilate.valueChanged.connect(self.RefreshSliderDilateInfo)

        self.filter_wiener_checkbox = QCheckBox('wiener')
        self.filter_wiener_checkbox.setChecked(False)
        self.filter_wiener_checkbox.clicked.connect(self.RunFilterWiener)

        self.filter_wiener_label = QLabel('\tsize')
        self.slider_wiener = QSlider(Qt.Horizontal)

        self.slider_wiener.setMinimum(2)
        self.slider_wiener.setMaximum(20)
        self.slider_wiener.setSingleStep(1)
        self.slider_wiener.setValue(10)
        self.slider_wiener_info = QLabel('10')
        self.slider_wiener.valueChanged.connect(self.RefreshSliderWienerInfo)

        self.filter_wiener_noise_label = QLabel('\tnoise')
        self.slider_wiener_noise = QSlider(Qt.Horizontal)

        self.slider_wiener_noise.setMinimum(0)
        self.slider_wiener_noise.setMaximum(400)
        self.slider_wiener_noise.setSingleStep(1)
        self.slider_wiener_noise.setValue(0)
        self.slider_wiener_noise_info = QLabel('auto')
        self.slider_wiener_noise.valueChanged.connect(self.RefreshSliderWienerInfo)

        # Bilateral

        self.filter_bilateral_checkbox = QCheckBox('bilateral')
        self.filter_bilateral_checkbox.setChecked(False)
        self.filter_bilateral_checkbox.clicked.connect(self.RunFilterBilateral)

        self.filter_bilateral_d_label = QLabel('\tdiameter')
        self.slider_bilateral_d = QSlider(Qt.Horizontal)

        self.slider_bilateral_d.setMinimum(2)
        self.slider_bilateral_d.setMaximum(20)
        self.slider_bilateral_d.setSingleStep(1)
        self.slider_bilateral_d.setValue(10)
        self.slider_bilateral_d_info = QLabel('10')
        self.slider_bilateral_d.valueChanged.connect(self.RefreshSliderBilateralInfo)

        self.filter_bilateral_space_label = QLabel('\tspace')
        self.slider_bilateral_space = QSlider(Qt.Horizontal)

        self.slider_bilateral_space.setMinimum(0)
        self.slider_bilateral_space.setMaximum(200)
        self.slider_bilateral_space.setSingleStep(1)
        self.slider_bilateral_space.setValue(10)
        self.slider_bilateral_space_info = QLabel('10')
        self.slider_bilateral_space.valueChanged.connect(self.RefreshSliderBilateralInfo)

        self.filter_bilateral_color_label = QLabel('\tcolor')
        self.slider_bilateral_color = QSlider(Qt.Horizontal)

        self.slider_bilateral_color.setMinimum(0)
        self.slider_bilateral_color.setMaximum(40)
        self.slider_bilateral_color.setSingleStep(1)
        self.slider_bilateral_color.setValue(10)
        self.slider_bilateral_color_info = QLabel('{:.2e}'.format(10 ** (10 / 10 - 5)))
        self.slider_bilateral_color.valueChanged.connect(self.RefreshSliderBilateralInfo)

        self.filters_checkbox = QCheckBox('all filters')
        self.filters_checkbox.setChecked(True)
        self.filters_checkbox.clicked.connect(self.RefreshFilters)

        self.forms_pre_processing = self.channel_checkbox_list + [
            self.orientation_checkbox,
            self.slider_downsample,
            self.slider_downsample_info,
            self.checkbox_1,
            self.checkbox_2,
            self.checkbox_3,
            self.checkbox_4
        ]

        self.button_build = QPushButton(QIcon('icons/arrow.png'), 'Build')
        self.button_build.setStatusTip('Builds the view of the data. It usually takes a while.')
        self.button_build.setFont(font)
        self.button_build.clicked.connect(self.BuildButtonClick)
        self.button_build.setDisabled(True)

        self.button_count = QPushButton(QIcon('icons/count-cat-icon.png'), 'Count NPs')
        self.button_count.setStatusTip('Counts nanoparticles.')
        self.button_count.setFont(font)
        self.button_count.clicked.connect(self.CountButtonClick)
        self.button_count.setDisabled(True)

        self.line_count_start = QLineEdit()
        self.line_count_start.setText('0')
        self.line_count_start.textChanged.connect(self.RefreshCountRange)

        self.line_count_stop = QLineEdit()
        self.line_count_stop.setText('100')
        self.line_count_stop.textChanged.connect(self.RefreshCountRange)

        num = 3
        self.list_slider_count = [QSlider(Qt.Horizontal) for i in range(num)]
        self.list_slider_count_info = [QLabel('5') for i in range(num)]

        for slider in self.list_slider_count:
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setSingleStep(1)
            slider.setValue(5)
            slider.valueChanged.connect(self.RefreshSliderCountInfo)

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

        self.forms_image_filters = [
            self.filter_gauss_checkbox,
            self.slider_gauss_info,
            self.slider_gauss,
            self.filter_wiener_checkbox,
            self.slider_wiener_info,
            self.slider_wiener,
            self.slider_wiener_noise_info,
            self.slider_wiener_noise,
            self.filter_wiener_label,
            self.filter_wiener_noise_label,
            self.filter_bilateral_checkbox,
            self.filter_bilateral_space_label,
            self.filter_bilateral_color_label,
            self.filter_bilateral_d_label,
            self.slider_bilateral_space,
            self.slider_bilateral_color,
            self.slider_bilateral_d,
            self.line_count_stop,
            self.line_count_start,
            self.slider_erode,
            self.slider_erode_info,
            self.filter_erode_checkbox,
            self.slider_dilate,
            self.slider_dilate_info,
            self.filter_dilate_checkbox
        ]

        # layout:

        layout_l = QVBoxLayout()
        layout_l.addWidget(self.button_open)
        layout_l.addWidget(self.file_name_label)

        label = QLabel('-- Choose channels to be shown --')
        label.setAlignment(Qt.AlignCenter)
        layout_l.addWidget(label)

        for channel in self.channel_checkbox_list:
            channel.setDisabled(True)
            layout_l.addWidget(channel)

        label = QLabel('-- Image settings --')
        label.setAlignment(Qt.AlignCenter)
        layout_l.addWidget(label)

        layout_l.addWidget(self.orientation_checkbox)

        slider_downsample_layout = QHBoxLayout()
        slider_downsample_layout.addWidget(QLabel('Downsample:'))
        slider_downsample_layout.addWidget(self.slider_downsample)
        slider_downsample_layout.addWidget(self.slider_downsample_info)
        layout_l.addLayout(slider_downsample_layout)

        label = QLabel('-- Data processing --')
        label.setAlignment(Qt.AlignCenter)
        layout_l.addWidget(label)

        plot_layout = QGridLayout()
        plot_layout.addWidget(self.checkbox_1, 0, 0)
        plot_layout.addWidget(self.checkbox_2, 1, 0)
        plot_layout.addWidget(self.checkbox_3, 0, 1)
        plot_layout.addWidget(self.checkbox_4, 1, 1)
        layout_l.addLayout(plot_layout)

        layout_l.addWidget(self.button_build)

        layout_r = QVBoxLayout()

        label = QLabel('-- Image filters --')
        label.setAlignment(Qt.AlignCenter)
        layout_r.addWidget(label)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel('Integration number:'))
        slider_layout.addWidget(self.slider_k)
        slider_layout.addWidget(self.slider_k_info)
        layout_r.addLayout(slider_layout)

        layout_r.addWidget(self.filters_checkbox)

        wiener_layout_box = QVBoxLayout()

        layout_r.addWidget(self.filter_wiener_checkbox)
        wiener_layout = QHBoxLayout()

        wiener_layout.addWidget(self.filter_wiener_label)
        wiener_layout.addWidget(self.slider_wiener)
        wiener_layout.addWidget(self.slider_wiener_info)

        wiener_layout_noise = QHBoxLayout()
        wiener_layout_noise.addWidget(self.filter_wiener_noise_label)
        wiener_layout_noise.addWidget(self.slider_wiener_noise)
        wiener_layout_noise.addWidget(self.slider_wiener_noise_info)

        wiener_layout_box.addLayout(wiener_layout)
        wiener_layout_box.addLayout(wiener_layout_noise)

        layout_r.addLayout(wiener_layout_box)

        bilateral_layout_box = QVBoxLayout()

        layout_r.addWidget(self.filter_bilateral_checkbox)

        bilateral_d_layout = QHBoxLayout()
        bilateral_d_layout.addWidget(self.filter_bilateral_d_label)
        bilateral_d_layout.addWidget(self.slider_bilateral_d)
        bilateral_d_layout.addWidget(self.slider_bilateral_d_info)

        bilateral_color_layout = QHBoxLayout()
        bilateral_color_layout.addWidget(self.filter_bilateral_color_label)
        bilateral_color_layout.addWidget(self.slider_bilateral_color)
        bilateral_color_layout.addWidget(self.slider_bilateral_color_info)

        bilateral_space_layout = QHBoxLayout()
        bilateral_space_layout.addWidget(self.filter_bilateral_space_label)
        bilateral_space_layout.addWidget(self.slider_bilateral_space)
        bilateral_space_layout.addWidget(self.slider_bilateral_space_info)

        bilateral_layout_box.addLayout(bilateral_d_layout)
        bilateral_layout_box.addLayout(bilateral_color_layout)
        bilateral_layout_box.addLayout(bilateral_space_layout)

        layout_r.addLayout(bilateral_layout_box)

        # gauss
        gauss_layout = QHBoxLayout()
        gauss_layout.addWidget(self.filter_gauss_checkbox)
        gauss_layout.addWidget(self.slider_gauss)
        gauss_layout.addWidget(self.slider_gauss_info)
        layout_r.addLayout(gauss_layout)

        erode_layout = QHBoxLayout()
        erode_layout.addWidget(self.filter_erode_checkbox)
        erode_layout.addWidget(self.slider_erode)
        erode_layout.addWidget(self.slider_erode_info)
        layout_r.addLayout(erode_layout)

        dilate_layout = QHBoxLayout()
        dilate_layout.addWidget(self.filter_dilate_checkbox)
        dilate_layout.addWidget(self.slider_dilate)
        dilate_layout.addWidget(self.slider_dilate_info)
        layout_r.addLayout(dilate_layout)

        for item in self.forms_image_filters + [self.filters_checkbox]:
            item.setDisabled(True)

        label = QLabel('-- NP analysis--')
        label.setAlignment(Qt.AlignCenter)
        layout_r.addWidget(label)

        np_analysis_layout = QHBoxLayout()
        np_analysis_layout.addWidget(QLabel('start:'))
        np_analysis_layout.addWidget(self.line_count_start)
        np_analysis_layout.addWidget(QLabel('stop:'))
        np_analysis_layout.addWidget(self.line_count_stop)
        layout_r.addLayout(np_analysis_layout)

        for i, (slider, info) in enumerate(zip(self.list_slider_count, self.list_slider_count_info)):
            np_analysis_slider_layout = QHBoxLayout()
            np_analysis_slider_layout.addWidget(QLabel('Value {}. '.format(i + 1)))
            np_analysis_slider_layout.addWidget(slider)
            np_analysis_slider_layout.addWidget(info)
            layout_r.addLayout(np_analysis_slider_layout)

        layout_r.addWidget(self.button_count)

        layout_r.addWidget(self.info)
        layout_r.addWidget(self.progress_bar)

        layout = QHBoxLayout()
        layout.addLayout(layout_l)
        layout.addLayout(layout_r)

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
        if self.view is not None:
            for core in self.view.core_list:
                core.k = self.slider_k.value()
            self.view.canvas_img.next_frame(0)

    def RefreshSliderDownSampleInfo(self):
        self.slider_downsample_info.setText(str(self.slider_downsample.value()))

    def RefreshSliderGaussInfo(self):
        self.slider_gauss_info.setText(str(self.slider_gauss.value() / 10))
        self.RunFilterGaussian()

    def RefreshSliderErodeInfo(self):
        self.slider_erode_info.setText(str(self.slider_erode.value()))
        self.RunFilterErode()

    def RefreshSliderDilateInfo(self):
        self.slider_dilate_info.setText(str(self.slider_dilate.value()))
        self.RunFilterDilate()

    def RefreshSliderBilateralInfo(self):
        self.slider_bilateral_d_info.setText(str(self.slider_bilateral_d.value()))
        self.slider_bilateral_space_info.setText(str(self.slider_bilateral_space.value()))
        self.slider_bilateral_color_info.setText('{:.2e}'.format(10 ** (self.slider_bilateral_color.value() / 10 - 5)))
        self.RunFilterBilateral()

    def RefreshSliderWienerInfo(self):
        self.slider_wiener_info.setText('{}'.format(self.slider_wiener.value()))
        if self.slider_wiener_noise.value() == 0:
            self.slider_wiener_noise_info.setText('auto')
        else:
            self.slider_wiener_noise_info.setText('{:.2e}'.format(10 ** (self.slider_wiener_noise.value() / 100 - 8)))

            # self.slider_wiener_noise_info.setText('{:.2e}'.format(self.slider_wiener_noise.value() / 1e3 / 1e6))
            # self.slider_wiener_noise_info.setText('{:.2e}'.format(self.slider_wiener_noise.value() / 1e5 *0.2))
        self.RunFilterWiener()


    def RefreshOrientationInfo(self):
        if self.orientation_checkbox.checkState() == 0:
            self.orientation_checkbox.setText('Vertical layout')
        else:
            self.orientation_checkbox.setText('Horizontal layout')

    def RefreshFilters(self):
        if self.filters_checkbox.isChecked():
            for core in self.view.core_list:
                core.postprocessing = True
            for item in self.forms_image_filters:
                item.setDisabled(False)

        else:
            for core in self.view.core_list:
                core.postprocessing = False
            for item in self.forms_image_filters:
                item.setDisabled(True)
        self.view.canvas_img.next_frame(0)

    def RefreshCountRange(self):
        start = re.fullmatch(r'[0-9]+', self.line_count_start.text()) is None
        stop = re.fullmatch(r'[0-9]+', self.line_count_stop.text()) is None

        if start:
            self.line_count_start.setText('0')
        if stop:
            self.line_count_stop.setText(str(len(self.view.core_list[0])))

        else:
            if int(self.line_count_stop.text()) < int(self.line_count_start.text()) + self.view.core_list[0].k:
                self.line_count_start.setText('0')
            elif int(self.line_count_stop.text()) > len(self.view.core_list[0]):
                self.line_count_stop.setText(str(len(self.view.core_list[0])))

    def RefreshSliderCountInfo(self):
        for slider, info in zip(self.list_slider_count, self.list_slider_count_info):
            info.setText(str(slider.value()))
        self.RunFilterGaussian()
        self.RunFilterBilateral()

    def RunFilterGaussian(self):
        # fn = lambda img: np.transpose(
        #     scipy.signal.decimate(np.transpose(scipy.signal.decimate(img, int(self.slider_gauss.value()/10+1))),
        #                           int(self.slider_gauss.value()/10+1)))
        # fn = lambda img: scipy.signal.medfilt2d(img, self.slider_gauss.value() // 2 * 2 + 1)
        fn = lambda img: gaussian_filter(img, self.slider_gauss.value() / 10)

        self.RunFilter(self.filter_gauss_checkbox.isChecked(), 'c_gauss', fn)

    def RunFilterErode(self):
        fn = lambda img: cv2.erode(np.float32(img), None, iterations=self.slider_erode.value())
        self.RunFilter(self.filter_erode_checkbox.isChecked(), 'y_erode', fn)

    def RunFilterDilate(self):
        fn = lambda img: cv2.dilate(np.float32(img), None, iterations=self.slider_dilate.value())
        self.RunFilter(self.filter_dilate_checkbox.isChecked(), 'z_dilate', fn)

    def RunFilterBilateral(self):
        fn = lambda img: cv2.bilateralFilter(np.float32(img), int(self.slider_bilateral_d.value()),
                                             10 ** (self.slider_bilateral_color.value() / 10 - 5),
                                             self.slider_bilateral_space.value() / 10)
        self.RunFilter(self.filter_bilateral_checkbox.isChecked(), 'b_bilateral', fn)

    def RunFilterWiener(self):
        if self.slider_wiener_noise_info.text() == 'auto':
            fn = lambda img: scipy.signal.wiener(img, self.slider_wiener.value())
        else:
            fn = lambda img: scipy.signal.wiener(img, self.slider_wiener.value(),
                                                 10 ** (self.slider_wiener_noise.value() / 100 - 8)
                                                 )
        self.RunFilter(self.filter_wiener_checkbox.isChecked(), 'a_wiener', fn)

    def RunFilter(self, checked, ftype, fn):
        if not checked:
            for core in self.view.core_list:
                try:
                    del core.postprocessing_filters[ftype]
                except KeyError:
                    print('postprocessing key not found')
        else:

            for core in self.view.core_list:
                core.postprocessing_filters[ftype] = fn
        dict(sorted(core.postprocessing_filters.items()))
        self.view.canvas_img.next_frame(0)

    def OpenButtonClick(self, s):
        dlg = QFileDialog(self)

        if self.ProcessPath(dlg.getOpenFileName()[0]):
            self.file_name_label.setText('folder path: ... {}\nfile name: {}'.format(self.folder[-20:], self.file))
            self.button_build.setDisabled(False)
            self.tool_file_info.setDisabled(False)

            if self.width[0] < self.height[0]:
                self.orientation_checkbox.setChecked(True)
            else:
                self.orientation_checkbox.setChecked(False)

            # for item in self.forms_pre_processing:
            #     item.setDisabled(False)

            for chch in self.channel_checkbox_list:
                chch.setChecked(False)

    def progress_fn(self, n):
        self.progress_bar.setValue(n)

    def thread_complete(self):
        if self.threadpool.activeThreadCount() == 0:
            if True in self.chosen_plots and self.view.core_list[0].spr_time is not None:
                canvas_plot = self.view.show_plots(self.chosen_plots)
                canvas_plot.main_window = self

                self.plot_window = PlotWindow(canvas_plot)
                self.plot_window.show()

                canvas_plot.setFocusPolicy(QtCore.Qt.ClickFocus)
                canvas_plot.setFocus()

            canvas_img = self.view.show_img()
            canvas_img.main_window = self
            self.img_window = PlotWindow(canvas_img)
            self.img_window.show()

            canvas_img.setFocusPolicy(QtCore.Qt.ClickFocus)
            canvas_img.setFocus()
            self.progress_bar.setVisible(False)

    def CountButtonClick(self):
        for core in self.view.core_list:
            core.count_nps(int(self.line_count_start.text()), int(self.line_count_stop.text()),
                           self.list_slider_count[0].value())

    def BuildButtonClick(self, s):

        # try:
        #     del self.view.canvas_plot
        #     del self.view.canvas_img
        #     del self.view
        # except:
        #     pass

        self.view = View()
        for i, channel in enumerate(self.channel_checkbox_list):
            if channel.checkState() == 2:
                core = Core(self.folder, self.file + '_{}'.format(i + 1))
                core.k = self.slider_k.value()
                core.downsample(self.slider_downsample.value())
                self.view.add_core(core)
        if len(self.view.core_list) == 0:
            OKDialog('error', 'no channels selected')
            return

        self.view.orientation = tl.BoolFromCheckBox(self.orientation_checkbox)

        self.chosen_plots = [
            tl.BoolFromCheckBox(self.checkbox_1),
            tl.BoolFromCheckBox(self.checkbox_2),
            tl.BoolFromCheckBox(self.checkbox_3),
            tl.BoolFromCheckBox(self.checkbox_4)
        ]

        self.progress_bar.setVisible(True)

        for item in self.forms_image_filters + [self.filters_checkbox]:
            item.setDisabled(False)

        self.button_count.setDisabled(False)
        self.line_count_stop.setText(str(len(self.view.core_list[0])))

        # for item in self.forms_pre_processing:
        #     item.setDisabled(True)

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

        # if self.chosen_plots[0]:
        self.thread_complete()


app = QApplication(sys.argv)
app.setFont(QFont('Courier', 8))

window = MainWindow()
window.show()
app.exec_()
