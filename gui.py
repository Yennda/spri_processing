import collections
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
from scipy import ndimage
from scipy.ndimage import gaussian_filter

import tools as tl
import global_var as gv
from core import Core
from view_pyqt import View
from gui_windows import OKDialog, PlotWindow, LoadingWindow
import gui_widgets as gw

matplotlib.use('Qt5Agg')


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

        self.img_window = None
        self.plot_window = None

        self.view = None
        self.file = str()
        self.folder = str()

        # info
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

        min_label_width = 150
        min_value_width = 50

        self.setWindowTitle("Crnka")
        self.setWindowIcon(QIcon('icons/cat-icon-2.png'))

        self.button_open = QPushButton(QIcon('icons/folder-open.png'), 'Open')
        self.font = self.button_open.font()
        self.font.setPointSize(14)

        self.button_open.setStatusTip(
            'Open any raw file from the desired measurement.')
        self.button_open.setFont(self.font)
        self.button_open.clicked.connect(self.OpenButtonClick)

        self.file_name_label = QLabel('folder: {}\nfile: {}'.format(self.folder, self.file))

        self.orientation_checkbox = QCheckBox('Horizontal layout')
        self.orientation_checkbox.setStatusTip('Changes the layout of the view.')
        self.orientation_checkbox.clicked.connect(self.RefreshOrientationInfo)
        self.orientation_checkbox.setChecked(True)

        self.channel_checkbox_list = []
        for i in range(1, 5):
            self.channel_checkbox_list.append(QCheckBox('channel {}'.format(i)))

        self.slider_downsample = gw.slider(1, 100, 1, 0, self.RefreshSliderDownSampleInfo)
        self.slider_downsample.setStatusTip(
            'N-times downsample the raw data in time.')
        self.slider_downsample_info = gw.value_label('0')

        self.slider_k = gw.slider(1, 100, 1, 10, self.RefreshSliderInfo)
        self.slider_k.setStatusTip(
            'Number of integrated frames. Recommended value is 10 for 10 fps.')
        self.slider_k_info = gw.value_label('10')

        "Data processing"

        self.checkbox_1 = QCheckBox('SPR')
        self.checkbox_1.setChecked(True)
        self.checkbox_2 = QCheckBox('intensity')
        self.checkbox_2.setStatusTip('!Takes a while to compute!')
        self.checkbox_3 = QCheckBox('histogram')
        self.checkbox_3.setStatusTip('!Slows done online browsing!')
        self.checkbox_4 = QCheckBox('NP counts')
        self.checkbox_4.setStatusTip('Works automatically.')
        self.checkbox_4.setChecked(True)
        self.checkbox_4.setDisabled(True)

        self.filter_gauss_checkbox = gw.checkbox_filter('Gaussian', False, self.RunFilterGaussian)
        self.slider_gauss = gw.slider(0, 50, 1, 10, self.RefreshSliderGaussInfo)
        self.slider_gauss_info = gw.value_label('1')

        self.filter_erode_checkbox = gw.checkbox_filter('maximum', False, self.RunFilterErode)
        self.slider_erode = gw.slider(0, 50, 1, 10, self.RefreshSliderErodeInfo)
        self.slider_erode_info = gw.value_label('1')

        self.filter_threshold_checkbox = gw.checkbox_filter('Threshold', False, self.RunFilterThreshold)
        self.slider_threshold = gw.slider(0, 200, 1, 50, self.RefreshSliderThresholdInfo)
        self.slider_threshold_info = gw.value_label('0.125')

        self.filter_distance_label = QLabel('Min. distance')
        self.filter_distance_label.setMinimumWidth(min_label_width)
        self.slider_distance = gw.slider(0, 20, 1, 3, self.RefreshSliderDistanceInfo)
        self.slider_distance_info = gw.value_label('3')

        self.filter_wiener_checkbox = gw.checkbox_filter('Wiener', False, self.RunFilterWiener)

        self.filter_wiener_label = QLabel('\tsize')
        self.filter_wiener_label.setMinimumWidth(min_label_width)
        self.slider_wiener = gw.slider(2, 20, 1, 6, self.RefreshSliderWienerInfo)
        self.slider_wiener_info = gw.value_label('6')

        self.filter_wiener_noise_label = QLabel('\tnoise')
        self.filter_wiener_noise_label.setMinimumWidth(min_label_width)
        self.slider_wiener_noise = gw.slider(0, 400, 1, 0, self.RefreshSliderWienerInfo)
        self.slider_wiener_noise_info = gw.value_label('auto')

        self.filter_bilateral_checkbox = gw.checkbox_filter('Bilateral', False, self.RunFilterBilateral)

        self.filter_bilateral_d_label = QLabel('\tdiameter')
        self.filter_bilateral_d_label.setMinimumWidth(min_label_width)
        self.slider_bilateral_d = gw.slider(2, 20, 1, 20, self.RefreshSliderBilateralInfo)
        self.slider_bilateral_d_info = gw.value_label('20')

        self.filter_bilateral_space_label = QLabel('\tspace')
        self.filter_bilateral_space_label.setMinimumWidth(min_label_width)
        self.slider_bilateral_space = gw.slider(0, 200, 1, 200, self.RefreshSliderBilateralInfo)
        self.slider_bilateral_space_info = gw.value_label('200')

        self.filter_bilateral_color_label = QLabel('\tcolor')
        self.filter_bilateral_color_label.setMinimumWidth(min_label_width)
        self.slider_bilateral_color = gw.slider(0, 40, 1, 10, self.RefreshSliderBilateralInfo)
        self.slider_bilateral_color_info = gw.value_label('{:.2e}'.format(10 ** (10 / 10 - 5)))

        self.filters_checkbox = gw.checkbox_filter('all filters', True, self.RefreshFilters)

        self.forms_pre_processing = self.channel_checkbox_list + [
            self.orientation_checkbox,
            self.slider_downsample,
            self.slider_downsample_info,
            self.checkbox_1,
            self.checkbox_2,
            self.checkbox_3
        ]

        self.button_build = gw.button('arrow', 'Build', self.font, True, self.BuildButtonClick)
        self.button_build.setStatusTip('Builds the view of the data. It usually takes a while.')

        self.button_correlate = gw.button('arrow', 'Correlation', self.font, True, self.CorrelateButtonClick)

        self.button_export = gw.button(None, 'Export data', self.font, True, self.ExportButtonClick)
        self.button_export_csv = gw.button(None, 'Export NP info as CSV', self.font, True, self.ExportCSVButtonClick)

        self.button_count = gw.button('count-cat-icon', 'Count NPs', self.font, True, self.CountButtonClick)
        self.button_count.setStatusTip('Counts nanoparticles.')

        self.line_count_start = QLineEdit()
        self.line_count_start.setText('0')
        self.line_count_start.textChanged.connect(self.RefreshCountRange)

        self.line_count_stop = QLineEdit()
        self.line_count_stop.setText('100')
        self.line_count_stop.textChanged.connect(self.RefreshCountRange)

        self.line_export_start = QLineEdit()
        self.line_export_start.setText('0')
        self.line_export_start.textChanged.connect(self.RefreshExportRange)

        self.line_export_stop = QLineEdit()
        self.line_export_stop.setText('100')
        self.line_export_stop.textChanged.connect(self.RefreshExportRange)

        self.view_channel_buttons = []

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

        self.np_info_label = QLabel(self.np_info_create())

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
            self.filter_erode_checkbox
        ]
        self.forms_np_recognition = [
            self.slider_threshold,
            self.slider_threshold_info,
            self.filter_threshold_checkbox,
            self.slider_distance_info,
            self.slider_distance,
        ]

        # layout:
        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.openTabUI(), 'Open')  # 0
        self.tabs.addTab(self.filtersTabUI(), 'Filters')  # 1
        self.tabs.addTab(self.NPRecognitionTabUI(), 'NP recognition')  # 2
        self.tabs.addTab(self.NPInfoUI(), 'NP Info')  # 3
        # self.tabs.addTab(self.ViewTabUI(), 'View')  # 4
        self.tabs.addTab(self.ExportsTabUI(), 'Exports')  # 5
        self.tabs.tabBarClicked.connect(self.RefreshNPInfo)
        # self.tabs.currentChanged.connect(self.RefreshNPInfo)
        layout.addWidget(self.tabs)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().setMinimumSize(400, 40)
        self.statusBar().setStyleSheet("border :1px solid gray;")
        self.setCentralWidget(self.tabs)

    def openTabUI(self):

        Tab = QWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.button_open)
        layout.addWidget(self.file_name_label)

        label = QLabel('-- Channels to be shown --')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        for channel in self.channel_checkbox_list:
            channel.setDisabled(True)
            layout.addWidget(channel)

        label = QLabel('-- Image settings --')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        layout.addWidget(self.orientation_checkbox)

        layout.addLayout(gw.layout_slider(
            QLabel('Downsample'),
            self.slider_downsample,
            self.slider_downsample_info)
        )

        label = QLabel('-- Data processing --')
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        plot_layout = QGridLayout()
        plot_layout.addWidget(self.checkbox_1, 0, 0)
        plot_layout.addWidget(self.checkbox_2, 1, 0)
        plot_layout.addWidget(self.checkbox_3, 0, 1)
        plot_layout.addWidget(self.checkbox_4, 1, 1)
        layout.addLayout(plot_layout)

        layout.addWidget(self.button_build)

        layout.addStretch(1)
        Tab.setLayout(layout)
        return Tab

    def filtersTabUI(self):
        Tab = QWidget()

        layout = QVBoxLayout()

        layout.addLayout(gw.layout_slider(
            QLabel('Integration number:'),
            self.slider_k,
            self.slider_k_info
        ))

        layout.addWidget(self.filters_checkbox)

        wiener_layout_box = QVBoxLayout()
        layout.addWidget(self.filter_wiener_checkbox)
        wiener_layout_box.addLayout(gw.layout_slider(
            self.filter_wiener_label,
            self.slider_wiener,
            self.slider_wiener_info
        ))
        wiener_layout_box.addLayout(gw.layout_slider(
            self.filter_wiener_noise_label,
            self.slider_wiener_noise,
            self.slider_wiener_noise_info
        ))

        layout.addLayout(wiener_layout_box)

        bilateral_layout_box = QVBoxLayout()
        layout.addWidget(self.filter_bilateral_checkbox)
        bilateral_layout_box.addLayout(gw.layout_slider(
            self.filter_bilateral_d_label,
            self.slider_bilateral_d,
            self.slider_bilateral_d_info
        ))
        bilateral_layout_box.addLayout(gw.layout_slider(
            self.filter_bilateral_color_label,
            self.slider_bilateral_color,
            self.slider_bilateral_color_info
        ))
        bilateral_layout_box.addLayout(gw.layout_slider(
            self.filter_bilateral_space_label,
            self.slider_bilateral_space,
            self.slider_bilateral_space_info
        ))

        layout.addLayout(bilateral_layout_box)

        layout.addLayout(gw.layout_slider(
            self.filter_gauss_checkbox,
            self.slider_gauss,
            self.slider_gauss_info
        ))

        layout.addLayout(gw.layout_slider(
            self.filter_erode_checkbox,
            self.slider_erode,
            self.slider_erode_info
        ))

        for item in self.forms_image_filters + [self.filters_checkbox] + self.forms_np_recognition:
            item.setDisabled(True)

        layout.addWidget(self.button_correlate)

        layout.addStretch(1)
        Tab.setLayout(layout)
        return Tab

    def NPRecognitionTabUI(self):
        Tab = QWidget()

        layout = QVBoxLayout()

        layout.addLayout(gw.layout_slider(
            self.filter_threshold_checkbox,
            self.slider_threshold,
            self.slider_threshold_info
        ))

        layout.addLayout(gw.layout_slider(
            self.filter_distance_label,
            self.slider_distance,
            self.slider_distance_info
        ))

        np_analysis_layout = QHBoxLayout()
        np_analysis_layout.addWidget(QLabel('start:'))
        np_analysis_layout.addWidget(self.line_count_start)
        np_analysis_layout.addWidget(QLabel('stop:'))
        np_analysis_layout.addWidget(self.line_count_stop)
        layout.addLayout(np_analysis_layout)

        # for i, (slider, info) in enumerate(zip(self.list_slider_count, self.list_slider_count_info)):
        #     np_analysis_slider_layout = QHBoxLayout()
        #     np_analysis_slider_layout.addWidget(QLabel('Value {}. '.format(i + 1)))
        #     np_analysis_slider_layout.addWidget(slider)
        #     np_analysis_slider_layout.addWidget(info)
        #     layout.addLayout(np_analysis_slider_layout)

        layout.addWidget(self.button_count)

        layout.addWidget(self.info)
        layout.addWidget(self.progress_bar)

        layout.addStretch(1)
        Tab.setLayout(layout)
        return Tab

    def ViewTabUI(self):
        Tab = QWidget()

        layout = QVBoxLayout()

        if self.view is None:
            layout.addWidget(QLabel('First, open a file.'))

        else:
            for core in self.view.core_list:
                layout_channel = QHBoxLayout()
                layout_channel.addWidget(QLabel('channel {}.'.format(core.file[-1])))

                for key in self.view.core_list[0]._range.keys():
                    channel = []

                    button = QPushButton(key)

                    font = self.button_open.font()
                    font.setPointSize(8)
                    button.setFont(font)

                    if key == core.type:
                        button.setDisabled(True)
                    button.clicked.connect(self.ViewButtonClick)

                    layout_channel.addWidget(button)
                    channel.append(button)

                self.view_channel_buttons.append(channel)

                layout.addLayout(layout_channel)

        layout.addStretch(1)
        Tab.setLayout(layout)
        return Tab

    def ExportsTabUI(self):
        Tab = QWidget()

        layout = QVBoxLayout()
        np_analysis_layout = QHBoxLayout()
        np_analysis_layout.addWidget(QLabel('start:'))
        np_analysis_layout.addWidget(self.line_export_start)
        np_analysis_layout.addWidget(QLabel('stop:'))
        np_analysis_layout.addWidget(self.line_export_stop)
        layout.addLayout(np_analysis_layout)

        layout.addWidget(self.button_export)
        layout.addWidget(self.button_export_csv)

        layout.addStretch(1)
        Tab.setLayout(layout)
        return Tab

    def NPInfoUI(self):
        Tab = QWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.np_info_label)

        layout.addStretch(1)
        Tab.setLayout(layout)
        return Tab

    def np_info_create(self):
        if self.view == None:
            text = 'Info will be displayed after image data processing.'
        elif self.view.core_list[0].np_container != []:
            text = str()

            for core in self.view.core_list:
                text += 'channel {}.\n'.format(core.file[-1])
                text += '\tpresent: {}\n'.format(len(core.nps_in_frame[self.view.f]))
                text += '\ttotal: {}\n'.format(sum(core.graphs['nps_pos']))
                text += '\t' + '-' * 27 + '\n'
                text += '\tcurrently adsorbed: {}\n'.format(core.graphs['nps_pos'][self.view.f])
                # text += '\tcurrently dedsorbed: {}\n'.format(core.graphs['nps_neg'][self.view.f])
                text += '\t' + '-' * 27 + '\n'

                text += '\tadsorbed up to now: {}\n'.format(sum(core.graphs['nps_pos'][:self.view.f]))
                # text += '\tdesorbed up to now: {}\n'.format(sum(core.graphs['nps_neg'][:self.view.f]))
                text += '\t' + '-' * 27 + '\n'

                # text += '\tbalance: {}\n'.format(
                #     sum(core.graphs['nps_pos'][:self.view.f]) -
                #     sum(core.graphs['nps_neg'][:self.view.f])
                # )

                text += '=' * 35 + '\n'
        else:
            text = 'Info will be displayed after image data processing.'
        return text

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

    def RefreshNPInfo(self):
        if self.tabs.currentIndex() == 2:
            if self.view is not None and self.view.core_list[0]._data_corr is not None:
                pass
                # self.view.change_type(None, 'corr')
                # self.view.set_range()
                # self.view.canvas_img.next_frame(0)

        if self.tabs.currentIndex() == 3:
            self.np_info_label.setText(self.np_info_create())

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

    def RefreshSliderThresholdInfo(self):
        self.slider_threshold_info.setText(str(self.slider_threshold.value() / 400))
        self.RunFilterThreshold()

    def RefreshSliderDistanceInfo(self):
        self.slider_distance_info.setText(str(self.slider_distance.value()))

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

    def RefreshExportRange(self):
        start = re.fullmatch(r'[0-9]+', self.line_export_start.text()) is None
        stop = re.fullmatch(r'[0-9]+', self.line_export_stop.text()) is None

        if start:
            self.line_export_start.setText('0')
        if stop:
            self.line_export_stop.setText(str(len(self.view.core_list[0])))

        else:
            if int(self.line_export_stop.text()) < int(self.line_export_start.text()) + self.view.core_list[0].k:
                self.line_export_start.setText('0')
            elif int(self.line_export_stop.text()) > len(self.view.core_list[0]):
                self.line_export_stop.setText(str(len(self.view.core_list[0])))

    def RefreshSliderCountInfo(self):
        for slider, info in zip(self.list_slider_count, self.list_slider_count_info):
            info.setText(str(slider.value() / 10))

    def RunFilterGaussian(self):
        fn = lambda img: gaussian_filter(img, self.slider_gauss.value() / 10)
        self.RunFilter(self.filter_gauss_checkbox.isChecked(), 'c_gauss', fn)

    def RunFilterErode(self):
        fn = lambda img: ndimage.maximum_filter(img, size=self.slider_erode.value())
        self.RunFilter(self.filter_erode_checkbox.isChecked(), 'y_erode', fn)

    def RunFilterThreshold(self):
        for core in self.view.core_list:
            core.threshold = self.filter_threshold_checkbox.isChecked()
            core.threshold_value = self.slider_threshold.value()/400

        self.view.canvas_img.next_frame(0)

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

        core.postprocessing_filters = collections.OrderedDict(sorted(core.postprocessing_filters.items()))
        self.view.canvas_img.next_frame(0)

    def CorrelateButtonClick(self):
        for core in self.view.core_list:
            core.make_correlation()
        self.view.change_type(None, 'corr')
        self.view.set_range()
        self.view.canvas_img.next_frame(0)
        self.tabs.setCurrentIndex(2)

        for item in self.forms_np_recognition:
            item.setDisabled(False)

    def CountButtonClick(self):
        for core in self.view.core_list:
            core.run_count_nps(int(self.line_count_start.text()), int(self.line_count_stop.text()),
                               self.slider_distance.value())
            # core.type = 'diff'
            self.filters_checkbox.setChecked(False)
            core.postprocessing = False

        self.view.change_type(None, 'diff')
        self.view.set_range()
        self.view.canvas_img.next_frame(0)

        self.chosen_plots = [
            tl.BoolFromCheckBox(self.checkbox_1),
            tl.BoolFromCheckBox(self.checkbox_2),
            tl.BoolFromCheckBox(self.checkbox_3),
            False
        ]
        canvas_plot = self.view.show_plots(self.chosen_plots)
        canvas_plot.main_window = self

        self.plot_window.close()
        self.plot_window = PlotWindow(canvas_plot)
        self.plot_window.show()

        self.tabs.setCurrentIndex(3)

    def ViewButtonClick(self):
        OKDialog('Message', 'Sorry, not implemented yet.', self)

    def ExportButtonClick(self):
        for core in self.view.core_list:
            core.export_data(int(self.line_export_start.text()), int(self.line_export_stop.text()))

    def ExportCSVButtonClick(self):
        if self.view.core_list[0].np_container is not None:
            for core in self.view.core_list:
                core.export_csv()

    def OpenButtonClick(self, s):
        dlg = QFileDialog(self)

        if self.ProcessPath(dlg.getOpenFileName()[0]):
            self.file_name_label.setText('folder path: ... {}\nfile name: {}'.format(self.folder[-20:], self.file))
            self.button_build.setDisabled(False)
            self.button_correlate.setDisabled(False)
            self.button_export.setDisabled(False)
            self.button_export_csv.setDisabled(False)
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

    def BuildButtonClick(self, s):
        self.view = View(self)

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
            False
        ]

        self.progress_bar.setVisible(True)

        for item in self.forms_image_filters + [self.filters_checkbox]:
            item.setDisabled(False)

        self.button_count.setDisabled(False)
        self.line_count_stop.setText(str(len(self.view.core_list[0])))
        self.line_export_stop.setText(str(len(self.view.core_list[0])))

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

            # if self.chosen_plots[2]:
            #     worker2 = Worker(core.make_intensity_int)
            #     worker2.signals.finished.connect(self.thread_complete)
            #     worker2.signals.progress.connect(self.progress_fn)
            #     self.threadpool.start(worker2)

            if self.chosen_plots[3]:
                worker3 = Worker(core.make_std_int)
                worker3.signals.finished.connect(self.thread_complete)
                worker3.signals.progress.connect(self.progress_fn)
                self.threadpool.start(worker3)

        # if self.chosen_plots[0]:
        self.thread_complete()
        self.tabs.setCurrentIndex(1)
        # self.tabs.removeTab(3)
        # self.tabs.insertTab(3, self.ViewTabUI(), 'View')

    # def keyPressEvent(self, e):
    #     print('key pressed {}'.format(e.key()))


app = QApplication(sys.argv)
app.setFont(QFont('Courier', 8))

window = MainWindow()
window.show()
app.exec_()
