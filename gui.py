import collections
import copy
import json
import sys
import os
import re
import time
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

import tools as tl
import global_var as gv
from core import Core
from view_pyqt import View
from gui_windows import OKDialog, PlotWindow
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

        self.font_small = self.button_open.font()
        self.font_small.setPointSize(10)

        self.button_open.setStatusTip(
            'Open any raw file from the desired measurement.')
        self.button_open.setFont(self.font)
        self.button_open.clicked.connect(self.OpenButtonClick)

        self.file_name_label = QLabel('folder: {}\nfile: {}'.format(self.folder, self.file))

        self.orientation_checkbox = QCheckBox('Horizontal layout')
        self.orientation_checkbox.setStatusTip('Changes the layout of the view.')
        self.orientation_checkbox.clicked.connect(self.RefreshOrientationInfo)
        self.orientation_checkbox.setChecked(True)

        self.transpose_checkbox = QCheckBox('Change orientation')

        self.crop_checkbox = QCheckBox('Crop')
        self.crop_checkbox.setChecked(True)

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

        self.filter_fourier_checkbox = gw.checkbox_filter('Fourier', False, self.RunFilterFourier)
        self.slider_fourier = gw.slider(0, 400, 1, 200, self.RefreshSliderFourierInfo)
        self.slider_fourier_info = gw.value_label('0')

        self.filter_threshold_checkbox = gw.checkbox_filter('Threshold', False, self.RunFilterThreshold)
        # self.slider_threshold = gw.slider(0, 200, 1, 50, self.RefreshSliderThresholdInfo)
        self.slider_threshold = QtWidgets.QDoubleSpinBox()
        self.slider_threshold.setValue(0.125)
        self.slider_threshold.setMinimum(0)
        self.slider_threshold.setMaximum(10)
        self.slider_threshold.setSingleStep(0.005)
        self.slider_threshold.setDecimals(3)
        self.slider_threshold.valueChanged.connect(self.RefreshSliderThresholdInfo)

        self.slider_threshold_info = gw.value_label('0.125')

        self.slider_threshold_adaptive = gw.slider(0, 50, 1, 0, self.RefreshSliderThresholdInfo)
        self.slider_threshold_adaptive_info = gw.value_label('0')

        self.label_threshold = QLabel('\tAdaptive coefficient')
        self.label_threshold.setMinimumWidth(min_label_width)

        self.filter_distance_label = QLabel('Min. distance')
        self.filter_distance_label.setMinimumWidth(min_label_width)
        self.slider_distance = gw.slider(0, 20, 1, 0, self.RefreshSliderDistanceInfo)
        self.slider_distance_info = gw.value_label('0')

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

        self.filter_absolute_checkbox = gw.checkbox_filter('Absolute value', False, self.RunFilterAbsolute)

        self.filters_checkbox = gw.checkbox_filter('all filters', True, self.RefreshFilters)

        self.forms_pre_processing = self.channel_checkbox_list + [
            self.orientation_checkbox,
            self.transpose_checkbox,
            self.slider_downsample,
            self.slider_downsample_info,
            self.checkbox_1,
            self.checkbox_2,
            self.checkbox_3
        ]

        self.button_build = gw.button('count-cat-icon', 'Build', self.font, True, self.BuildButtonClick)
        self.button_build.setStatusTip('Builds the view of the data. It usually takes a while.')

        self.fourier_box = QComboBox()
        self.button_fourier = gw.button(None, 'Select', self.font_small, True, self.FourierButtonClick)
        self.button_fourier_clear = gw.button(None, 'Clear', self.font_small, True, self.FourierRemoveButtonClick)

        self.ommit_box = QComboBox()
        self.button_ommit = gw.button(None, 'Select', self.font_small, True, self.OmmitButtonClick)
        self.button_ommit_clear = gw.button(None, 'Clear', self.font_small, True, self.OmmitRemoveButtonClick)

        self.button_correlate = gw.button('brain', 'Correlation', self.font, True, self.CorrelateButtonClick)

        self.select_box = QComboBox()
        self.button_select = gw.button(None, 'Select', self.font_small, True, self.SelectButtonClick)

        self.button_export = gw.button('poison', 'Export data', self.font, True, self.ExportButtonClick)
        self.button_export_csv = gw.button('table', 'Export NP counts', self.font, True,
                                           self.ExportCSVButtonClick)
        self.button_export_parameters = gw.button('application-export', 'Export parameters', self.font, True,
                                                  self.ExportParametersButtonClick)
        self.button_import_parameters = gw.button('application-import', 'Import', self.font, True,
                                                  self.ImportParametersButtonClick)

        self.button_export_nps = gw.button('beans', 'Export NPs', self.font, True,
                                           self.ExportNPsButtonClick)
        self.button_import_nps = gw.button(None, 'Import', self.font, True,
                                           self.ImportNPsButtonClick)
        self.button_import_nps_old = gw.button(None, 'Import NPs old', self.font, True,
                                               self.ImportNPsButtonClickOld)

        self.button_analyse_nps = gw.button('magnifier', 'Analyse NPs', self.font, True,
                                            self.AnalyseNPsButtonClick)

        self.button_export_gif = gw.button('film', 'Create GIF', self.font, True, self.ExportGIFButtonClick)

        self.button_export_video = gw.button('films', 'Create Video', self.font, True, self.ExportVideoButtonClick)

        self.exim_buttons = [
            self.button_export,
            self.button_export_csv,
            self.button_export_parameters,
            self.button_export_nps,
            self.button_import_nps,
            self.button_import_nps_old,
            self.button_analyse_nps,
            self.button_export_gif,
            self.button_export_video
        ]

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

        self.radio_channel_view = []

        self.np_info_label = QLabel(self.np_info_create())

        self.forms_image_filters = [
            self.slider_gauss_info,
            self.slider_gauss,
            self.slider_wiener_info,
            self.slider_wiener,
            self.slider_wiener_noise_info,
            self.slider_wiener_noise,
            self.filter_wiener_label,
            self.filter_wiener_noise_label,
            self.filter_bilateral_space_label,
            self.filter_bilateral_color_label,
            self.filter_bilateral_d_label,
            self.slider_bilateral_space,
            self.slider_bilateral_color,
            self.slider_bilateral_d,
            self.slider_fourier,
            self.slider_fourier_info,
        ]
        self.forms_image_filters_checkoboxes = [
            self.filter_gauss_checkbox,
            self.filter_wiener_checkbox,
            self.filter_bilateral_checkbox,
            self.filter_fourier_checkbox,
            self.filter_absolute_checkbox
        ]

        self.forms_np_recognition = [
            self.slider_threshold,
            self.slider_threshold_info,
            self.filter_threshold_checkbox,
            self.label_threshold,
            self.slider_threshold_adaptive,
            self.slider_threshold_adaptive_info,
            self.filter_distance_label,
            self.slider_distance_info,
            self.slider_distance,
        ]

        layout = QVBoxLayout()

        self.layout_view = QHBoxLayout()
        self.view_cb_list = []

        for i in range(4):
            view_cb = gw.combo_box()
            view_cb.currentIndexChanged.connect(
                lambda event, cbval=view_cb: self.change_view(event, cbval))
            view_cb.setDisabled(True)
            self.view_cb_list.append(view_cb)

            self.layout_view.addWidget(QLabel('ch. {}'.format(i + 1)))
            self.layout_view.addWidget(view_cb)

        self.layout_view.addStretch(1)

        layout.addLayout(self.layout_view)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.openTabUI(), 'Open')  # 0
        self.tabs.addTab(self.filtersTabUI(), 'Filters')  # 1
        self.tabs.addTab(self.NPRecognitionTabUI(), 'NP recognition')  # 2
        self.tabs.addTab(self.NPInfoUI(), 'NP Info')  # 3
        # self.tabs.addTab(self.ViewTabUI(), 'View')  # 4
        self.tabs.addTab(self.ExportsTabUI(), 'Export/Import')  # 5
        self.tabs.tabBarClicked.connect(self.RefreshTabs)

        layout.addWidget(self.tabs)

        widget = QWidget()
        widget.setLayout(layout)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().setMinimumSize(400, 40)
        self.statusBar().setStyleSheet("border :1px solid gray;")
        self.setCentralWidget(widget)

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

        layout_orientation = QHBoxLayout()
        layout_orientation.addWidget(self.orientation_checkbox)
        layout_orientation.addWidget(self.transpose_checkbox)
        layout.addLayout(layout_orientation)

        layout.addWidget(self.crop_checkbox)

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
            self.filter_fourier_checkbox,
            self.slider_fourier,
            self.slider_fourier_info
        ))

        layout.addWidget(self.filter_absolute_checkbox)

        label_fourier = QLabel('Remove spatial freq.:')
        label_fourier.setMinimumWidth(200)
        layout_fourier_buttons = QHBoxLayout()
        layout_fourier_buttons.addWidget(label_fourier)
        layout_fourier_buttons.addWidget(self.fourier_box)
        layout_fourier_buttons.addWidget(self.button_fourier)
        layout_fourier_buttons.addWidget(self.button_fourier_clear)
        layout.addLayout(layout_fourier_buttons)

        label_select = QLabel('Pattern for correlation:')
        label_select.setMinimumWidth(200)
        layout_select_buttons = QHBoxLayout()
        layout_select_buttons.addWidget(label_select)
        layout_select_buttons.addWidget(self.select_box)
        layout_select_buttons.addWidget(self.button_select)
        layout.addLayout(layout_select_buttons)

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
            self.label_threshold,
            self.slider_threshold_adaptive,
            self.slider_threshold_adaptive_info
        ))

        layout.addLayout(gw.layout_slider(
            self.filter_distance_label,
            self.slider_distance,
            self.slider_distance_info
        ))

        layout_ommit_buttons = QHBoxLayout()
        label_ommit = QLabel('Ommit regions:')
        label_ommit.setMinimumWidth(200)
        layout_ommit_buttons.addWidget(label_ommit)
        layout_ommit_buttons.addWidget(self.ommit_box)
        layout_ommit_buttons.addWidget(self.button_ommit)
        layout_ommit_buttons.addWidget(self.button_ommit_clear)
        layout.addLayout(layout_ommit_buttons)

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

                layout_direct = QVBoxLayout()
                for itype in ['raw', 'int', 'diff', 'corr']:
                    channel = []

                    button = QRadioButton(itype)
                    button.toggled.connect(lambda: self.RefreshRadioDirect(itype))
                    channel.append(button)
                    layout_direct.addWidget(button)

                layout_fourier = QVBoxLayout()
                layout_fourier.addWidget(QLabel('Fourier of:'))
                for itype in ['raw', 'int', 'diff']:
                    channel = []
                    button = QRadioButton(itype)
                    button.toggled.connect(lambda: self.RefreshRadioFourier(itype))
                    channel.append(button)
                    layout_fourier.addWidget(button)
                layout_channel.addLayout(layout_direct)
                layout_channel.addLayout(layout_fourier)

                self.radio_channel_view.append(channel)

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

        layout_parameters_ie = QHBoxLayout()
        layout_parameters_ie.addWidget(self.button_export_parameters)
        layout_parameters_ie.addWidget(self.button_import_parameters)
        layout.addLayout(layout_parameters_ie)

        layout_nps_ie = QHBoxLayout()
        layout_nps_ie.addWidget(self.button_export_nps)
        layout_nps_ie.addWidget(self.button_import_nps)
        layout.addLayout(layout_nps_ie)

        layout_nps_ie_old = QHBoxLayout()

        layout_nps_ie_old.addStretch(1)
        layout_nps_ie_old.addWidget(self.button_import_nps_old)

        layout.addLayout(layout_nps_ie_old)

        layout.addWidget(self.button_analyse_nps)
        layout.addWidget(self.button_export_gif)
        layout.addWidget(self.button_export_video)

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

                if core._mask_ommit is not None:
                    text += '\tarea: {:.0f} px \t= {:.1f} mm2\t = {:.1f}%\n'.format(
                        core.active_area,
                        core.active_area * gv.PX ** 2,
                        core.active_area / core.area * 100
                    )
                else:
                    text += '\tarea: {:.0f} px \t= {:.1f} mm2\n'.format(core.area, core.area / 1e6)
                text += '\ttotal: {}\n'.format(sum(core.graphs['nps_pos']))
                text += '\ttotal density: {:.1f} /mm^2\n'.format(
                    sum(core.graphs['nps_pos']) / core.active_area / gv.PX ** 2)
                text += '\t' + '-' * 27 + '\n'

                text += '\tpresent: {}\n'.format(len(core.nps_in_frame[self.view.f]))

                text += '\tcurrently adsorbed: {}\n'.format(core.graphs['nps_pos'][self.view.f])
                # text += '\tcurrently dedsorbed: {}\n'.format(core.graphs['nps_neg'][self.view.f])

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
        if self.tabs.currentIndex() == 3:
            self.np_info_label.setText(self.np_info_create())

    def print(self, string):
        print('gui: {}'.format(string))

    def RefreshTabs(self):
        pass
        # self.print(self.tabs.currentIndex())
        # if self.view is not None and self.view.core_list[0]._data_corr is not None:
        #     if self.tabs.currentIndex() == 2:
        #         self.view.change_type(None, 'corr')
        #         self.view.set_range()
        #         self.view.canvas_img.next_frame(0)
        #         self.view.canvas_img.draw()
        #
        #     if self.tabs.currentIndex() == 1:
        #         self.view.change_type(None, 'diff')
        #         self.view.set_range()
        #         self.view.canvas_img.next_frame(0)
        #         self.view.canvas_img.draw()

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

    def RefreshSliderFourierInfo(self):
        self.slider_fourier_info.setText(str(self.slider_fourier.value() - 200))
        self.RunFilterFourier()

    def RefreshSliderThresholdInfo(self):
        self.slider_threshold_info.setText('{:.4e}'.format(self.slider_threshold.value()))
        self.slider_threshold_adaptive_info.setText(str(self.slider_threshold_adaptive.value() / 10))

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

    def RefreshRadioDirect(self, itype):
        self.print(itype)

    def RefreshRadioFourier(self, itype):
        self.print(itype)

    def RefreshFilters(self):
        if self.view is None:
            if self.filters_checkbox.isChecked():
                for item in self.forms_image_filters + self.forms_image_filters_checkoboxes:
                    item.setDisabled(False)

            else:
                for item in self.forms_image_filters + self.forms_image_filters_checkoboxes:
                    item.setDisabled(True)
            return

        if self.filters_checkbox.isChecked():
            for core in self.view.core_list:
                core.postprocessing = True
            for item in self.forms_image_filters + self.forms_image_filters_checkoboxes:
                item.setDisabled(False)

        else:
            for core in self.view.core_list:
                core.postprocessing = False
            for item in self.forms_image_filters + self.forms_image_filters_checkoboxes:
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

    def RunFilterFourier(self):
        fn = lambda img: tl.fourier_filter_threshold(img, self.slider_fourier.value() - 200)
        self.RunFilter(self.filter_fourier_checkbox.isChecked(), 'a_fourier', fn)

    def RunFilterThreshold(self):
        if self.view is not None:
            for core in self.view.core_list:
                core.threshold = self.filter_threshold_checkbox.isChecked()
                core.threshold_value = self.slider_threshold.value()
                core.threshold_adaptive = self.slider_threshold_adaptive.value() / 10

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

    def RunFilterAbsolute(self):
        fn = lambda img: np.abs(img)
        self.RunFilter(self.filter_absolute_checkbox.isChecked(), 'a_absolute', fn)

    def RunFilter(self, checked, ftype, fn):
        if self.view is None:
            return

        if not checked:
            for core in self.view.core_list:
                try:
                    del core.postprocessing_filters[ftype]
                except KeyError:
                    self.print('postprocessing key not found')
        else:

            for core in self.view.core_list:
                core.postprocessing_filters[ftype] = fn

        core.postprocessing_filters = collections.OrderedDict(sorted(core.postprocessing_filters.items()))
        if self.view.canvas_img is not None:
            self.view.canvas_img.next_frame(0)

    def export_parameters(self):
        if not os.path.isdir(self.folder + gv.FOLDER_IDEAS):
            os.mkdir(self.folder + gv.FOLDER_IDEAS)

        for core in self.view.core_list:
            parameters = {
                'orientation_checkbox': self.orientation_checkbox.isChecked(),
                'transpose_checkbox': self.transpose_checkbox.isChecked(),
                'crop_checkbox': self.crop_checkbox.isChecked(),
                'slider_downsample': self.slider_downsample.value(),
                'slider_k': self.slider_k.value(),
                'filters_checkbox': self.filters_checkbox.isChecked(),
                'filter_wiener_checkbox': self.filter_wiener_checkbox.isChecked(),
                'slider_wiener': self.slider_wiener.value(),
                'slider_wiener_noise': self.slider_wiener_noise.value(),
                'filter_bilateral_checkbox': self.filter_bilateral_checkbox.isChecked(),
                'slider_bilateral_d': self.slider_bilateral_d.value(),
                'slider_bilateral_color': self.slider_bilateral_color.value(),
                'slider_bilateral_space': self.slider_bilateral_space.value(),
                'filter_gauss_checkbox': self.filter_gauss_checkbox.isChecked(),
                'slider_gauss': self.slider_gauss.value(),
                'filter_fourier_checkbox': self.filter_fourier_checkbox.isChecked(),
                'slider_fourier': self.slider_fourier.value(),
                'slider_threshold': self.slider_threshold.value(),
                'slider_threshold_adaptive': self.slider_threshold_adaptive.value(),
                'slider_distance': self.slider_distance.value(),
                'filter_absolute_checkbox': self.filter_absolute_checkbox.isChecked()

            }
            core.save_masks()

            with open(self.folder + gv.FOLDER_IDEAS + '/' + 'parameters_' + core.file + '.json', 'w') as file:
                json.dump(parameters, file)

            self.print('Parameters exported')

    def import_parameters(self):
        for i, channel in enumerate(self.channel_checkbox_list):
            if channel.checkState() == 2:
                with open(
                        self.folder + gv.FOLDER_IDEAS + '/' + 'parameters_' + self.file + '_{}'.format(i + 1) + '.json',
                        'r') as file:
                    p = json.load(file)

                self.orientation_checkbox.setChecked(p['orientation_checkbox'])
                self.transpose_checkbox.setChecked(p['transpose_checkbox'])
                self.crop_checkbox.setChecked(p['crop_checkbox']),
                self.slider_downsample.setValue(p['slider_downsample'])
                self.slider_k.setValue(p['slider_k'])
                self.filter_wiener_checkbox.setChecked(p['filter_wiener_checkbox'])
                self.slider_wiener.setValue(p['slider_wiener'])
                self.slider_wiener_noise.setValue(p['slider_wiener_noise'])
                self.filter_bilateral_checkbox.setChecked(p['filter_bilateral_checkbox'])
                self.slider_bilateral_d.setValue(p['slider_bilateral_d'])
                self.slider_bilateral_color.setValue(p['slider_bilateral_color'])
                self.slider_bilateral_space.setValue(p['slider_bilateral_space'])
                self.filter_gauss_checkbox.setChecked(p['filter_gauss_checkbox'])
                self.slider_gauss.setValue(p['slider_gauss'])
                self.filter_fourier_checkbox.setChecked(p['filter_fourier_checkbox'])
                self.slider_fourier.setValue(p['slider_fourier'])

                if type(p['slider_threshold']) is int:
                    self.slider_threshold.setValue(p['slider_threshold']/200)
                else:
                    self.slider_threshold.setValue(p['slider_threshold'])

                self.slider_threshold_adaptive.setValue(p['slider_threshold_adaptive'])
                self.slider_distance.setValue(p['slider_distance'])
                self.filters_checkbox.setChecked(p['filters_checkbox'])

                if 'filter_absolute_checkbox' in p.keys():
                    self.filter_absolute_checkbox.setChecked(p['filter_absolute_checkbox'])
                self.RefreshFilters()

        if self.view is not None:
            for core in self.view.core_list:
                core.load_masks()

        self.print('Parameters imported')

    def change_view(self, event, view_box):
        i = int(view_box.objectName())
        self.view.change_type(self.view.axes[i], view_box.currentText())
        self.view.set_range()
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
        self.view.change_type(None, 'corr')
        self.view.set_range()
        self.view.canvas_img.next_frame(0)

        for core in self.view.core_list:
            if not core.threshold:
                self.filter_threshold_checkbox.setChecked(True)
                core.threshold = True

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

        if self.plot_window is not None:
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

    def ExportParametersButtonClick(self):
        self.export_parameters()

    def ImportParametersButtonClick(self):
        self.import_parameters()

    def ExportNPsButtonClick(self):
        if self.view.core_list[0].np_container is not None:
            for core in self.view.core_list:
                core.export_np_csv()

    def AnalyseNPsButtonClick(self):
        if self.view.core_list[0].np_container is not None:
            for core in self.view.core_list:
                core.np_analysis()

    def ExportGIFButtonClick(self):
        start = int(self.line_export_start.text())
        stop = int(self.line_export_stop.text())

        self.view.canvas_img.save_gif(start, stop, True)

    def ExportVideoButtonClick(self):
        start = int(self.line_export_start.text())
        stop = int(self.line_export_stop.text())

        self.view.canvas_img.save_gif(start, stop, False)

    def ImportNPsButtonClick(self):
        if self.view.core_list[0].np_container is not None:
            for core in self.view.core_list:
                core.import_np_csv()

            canvas_plot = self.view.show_plots(self.chosen_plots)
            canvas_plot.main_window = self

            self.plot_window.close()
            self.plot_window = PlotWindow(canvas_plot)
            self.plot_window.show()

    def ImportNPsButtonClickOld(self):
        if self.view.core_list[0].np_container is not None:
            for core in self.view.core_list:
                core.import_np_csv_old()

            canvas_plot = self.view.show_plots(self.chosen_plots)
            canvas_plot.main_window = self

            self.plot_window.close()
            self.plot_window = PlotWindow(canvas_plot)
            self.plot_window.show()

    def OpenButtonClick(self, s):
        dlg = QFileDialog(self)

        if self.ProcessPath(dlg.getOpenFileName()[0]):
            self.file_name_label.setText(
                'folder path: {} ... {}\nfile name: {}'.format(self.folder[:10], self.folder[-20:], self.file))
            self.button_build.setDisabled(False)
            self.button_correlate.setDisabled(False)

            self.button_fourier.setDisabled(False)
            self.button_fourier_clear.setDisabled(False)
            self.button_ommit.setDisabled(False)
            self.button_ommit_clear.setDisabled(False)
            self.button_select.setDisabled(False)
            self.button_import_parameters.setDisabled(False)

            self.tool_file_info.setDisabled(False)

            # if self.width[0] < self.height[0]:
            #     self.orientation_checkbox.setChecked(True)
            # else:
            #     self.orientation_checkbox.setChecked(False)

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

    def FourierButtonClick(self):
        if self.button_fourier.text() == 'Select':
            self.button_fourier.setText('OK')
            self.fourier_box.setDisabled(True)
            self.button_fourier_clear.setDisabled(False)
        else:
            self.button_fourier.setText('Select')
            self.fourier_box.setDisabled(False)
            self.button_fourier_clear.setDisabled(True)

        i = self.fourier_box.currentIndex()
        self.view.canvas_img.select_area(self.view.axes[i], 'fourier')

    def FourierRemoveButtonClick(self):
        i = self.fourier_box.currentIndex()
        self.view.canvas_img.mask = np.zeros(self.view.core_list[i].shape_img)
        self.view.canvas_img.mask_img.set_array(
            self.view.canvas_img.mask
        )

        self.view.next_frame(0)

    def OmmitButtonClick(self):
        if self.button_ommit.text() == 'Select':
            self.button_ommit.setText('OK')
            self.ommit_box.setDisabled(True)
            self.button_ommit_clear.setDisabled(False)
        else:
            self.button_ommit.setText('Select')
            self.ommit_box.setDisabled(False)
            self.button_ommit_clear.setDisabled(True)

        i = self.ommit_box.currentIndex()
        self.view.canvas_img.select_area(self.view.axes[i], 'ommit')

    def OmmitRemoveButtonClick(self):
        i = self.ommit_box.currentIndex()
        self.view.canvas_img.mask = np.zeros(self.view.core_list[i].shape_img)
        self.view.canvas_img.mask_img.set_array(
            self.view.canvas_img.mask
        )

        self.view.next_frame(0)

    def SelectButtonClick(self):
        if self.button_select.text() == 'Select':
            self.button_select.setText('OK')
            self.select_box.setDisabled(True)

        else:
            self.button_select.setText('Select')
            self.select_box.setDisabled(False)

        i = self.select_box.currentIndex()
        self.view.canvas_img.select_area(self.view.axes[i], 'np')

    def BuildButtonClick(self, s):
        self.view = View(self)
        self.fourier_box.clear()
        self.ommit_box.clear()
        self.select_box.clear()
        num_of_channel = 0

        [self.view_cb_list[i].setDisabled(True) for i in range(4)]

        for i, channel in enumerate(self.channel_checkbox_list):
            if channel.checkState() == 2:
                self.fourier_box.addItem('ch. {}'.format(i + 1))
                self.ommit_box.addItem('ch. {}'.format(i + 1))
                self.select_box.addItem('ch. {}'.format(i + 1))

                core = Core(self.folder, self.file + '_{}'.format(i + 1))

                self.view_cb_list[i].setDisabled(False)
                self.view_cb_list[i].setObjectName(str(num_of_channel))
                num_of_channel += 1

                if tl.BoolFromCheckBox(self.crop_checkbox):
                    core.crop()
                    core.ref_frame = 0

                if tl.BoolFromCheckBox(self.transpose_checkbox):
                    core._data_raw = np.swapaxes(core._data_raw, 0, 1)
                    core.ref_frame = 0

                core._mask_ommit = np.zeros(core.shape_img)

                core.k = self.slider_k.value()
                core.downsample(self.slider_downsample.value())

                core.noise_analysis(self.avg * self.slider_downsample.value())

                self.view.add_core(core)

        self.layout_view.addStretch(1)

        if len(self.view.core_list) == 0:
            raise FileNotFoundError('No channels selected.')

        self.view.orientation = tl.BoolFromCheckBox(self.orientation_checkbox)

        self.chosen_plots = [
            tl.BoolFromCheckBox(self.checkbox_1),
            tl.BoolFromCheckBox(self.checkbox_2),
            tl.BoolFromCheckBox(self.checkbox_3),
            False
        ]

        self.progress_bar.setVisible(True)

        for item in self.forms_image_filters + [self.filters_checkbox, self.line_count_stop, self.line_count_start]:
            item.setDisabled(False)

        for item in self.forms_image_filters_checkoboxes:
            item.setChecked(False)
            item.setDisabled(False)

        for btn in self.exim_buttons:
            btn.setDisabled(False)

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
        # self.tabs.removeTab(4)
        # self.tabs.insertTab(4, self.ViewTabUI(), 'View')
    #
    # def keyPressEvent(self, e):
    #     self.print('key pressed {}'.format(e.key()))


def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    OKDialog('Error catched', '{}\n\n{}\nMore info in command line'.format(exc_value, '-' * 20))
    print(tb)


sys.excepthook = excepthook
app = QApplication(sys.argv)
app.setFont(QFont('Courier', 8))

window = MainWindow()
window.show()
app.exec_()
