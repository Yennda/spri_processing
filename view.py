import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from matplotlib.widgets import RectangleSelector

from global_var import *
import tools as tl

matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Palatino Linotype')
matplotlib.rc('text', usetex='false')
# matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Palatino Linotype'
matplotlib.rcParams['mathtext.it'] = 'Palatino Linotype:italic'
matplotlib.rcParams['mathtext.bf'] = 'BiPalatino Linotype:bold'


class View(object):
    def __init__(self):
        self.core_list = []
        self._f = 0
        self.orientation = True
        self.length = 0

    def add_core(self, core):
        self.core_list.append(core)
        if self.length > len(core) or self.length == 0:
            self.length = len(core)

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, df):
        self._f = (self._f + df) % self.length

    def show(self):
        def next_frame(df):
            self.f = df


        def mouse_scroll(event):
            fig = event.canvas.figure
            if event.button == 'down':
                self.f = 1
            elif event.button == 'up':
                self.f = -1

            fig.suptitle(self.f)
            for i, core in enumerate(self.core_list):
                img_shown[i].set_array(core.frame(self.f)['image'])
            fig.canvas.draw()

        def button_press(event):
            fig = event.canvas.figure
            if event.key == '9':
                self.f = 100
            elif event.key == '7':
                self.f = -100
            elif event.key == '6':
                self.f = 10
            elif event.key == '4':
                self.f = -10
            elif event.key == '3':
                self.f = 1
            elif event.key == '1':
                self.f = -1
            elif event.key == 'i':
                event.inaxes.core.type = 'int'
            elif event.key == 'd':
                event.inaxes.core.type = 'diff'

            fig.suptitle(self.f)
            for i, core in enumerate(self.core_list):
                img_shown[i].set_array(core.frame(self.f)['image'])
            fig.canvas.draw()

        if self.orientation:
            fig, axes = plt.subplots(nrows=len(self.core_list), ncols=1)
        else:
            fig, axes = plt.subplots(ncols=len(self.core_list), nrows=1)

        fig.suptitle(self.f)

        img_shown = []

        for i, core in enumerate(self.core_list):
            frame = core.frame(self.f)
            img_shown.append(
                axes[i].imshow(
                    frame['image'],
                    cmap='gray',
                    zorder=0,
                    vmin=frame['range'][0],
                    vmax=frame['range'][1]
                )
            )
            axes[i].core = core

            if self.orientation:
                axes[i].set_ylabel('{}; {}'.format(core.file[4:], core.type))
            else:
                axes[i].set_xlabel('{}; {}'.format(core.file[4:], core.type))

        fig.canvas.mpl_connect('key_press_event', button_press)
        fig.canvas.mpl_connect('scroll_event', mouse_scroll)
