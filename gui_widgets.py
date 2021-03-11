from PyQt5.QtWidgets import *
from PyQt5.Qt import QVBoxLayout, QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets

min_value_width = 50
min_label_width = 150


def value_label(text):
    label = QLabel(text)
    label.setMinimumWidth(min_value_width)
    label.setAlignment(QtCore.Qt.AlignRight)
    return label


def slider(mn, mx, step, value, fn):
    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(mn)
    slider.setMaximum(mx)
    slider.setSingleStep(step)
    slider.setValue(value)
    slider.valueChanged.connect(fn)
    return slider


def checkbox_filter(name, checked, fn):
    checkbox = QCheckBox(name)
    checkbox.setChecked(checked)
    checkbox.setMinimumWidth(min_label_width)
    checkbox.clicked.connect(fn)
    return checkbox


def button(icon, name, font, disabled, fn):
    if icon == None:
        button = QPushButton(name)
    else:
        button = QPushButton(QIcon('icons/{}.png'.format(icon)), name)
    button.setFont(font)
    button.setDisabled(disabled)
    button.clicked.connect(fn)
    return button


def layout_slider(label, slider, info):
    layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(slider)
    layout.addWidget(info)
    return layout

def combo_box():
    cb = QComboBox()
    items = [
        'raw',
        'int',
        'diff',
        'corr',
        'four_r',
        'four_i',
        'four_d',
        'mask'
    ]
    for item in items:
        cb.addItem(item)
    cb.setCurrentText('diff')

    return cb
