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
    def __init__(self, core):
        self.core = core
        self.f = 0

    def frame_info(self, f):

        return '{}: {}/{}  t= {} s dt= {:.2f} s'.format(
            self.core.type,
            f,
            len(self.core),
            tl.SecToMin(self.core._time_info[f][0]),
            self.core._time_info[f][1]
        )

    def show(self):
        def button_press(event):
            fig = event.canvas.figure
            ax = fig.axes[0]

            if event.key == '6':
                self.f += 1

                ax.set_title(self.frame_info(ax.index))
                img.set_array(self.core.frame(self.f)['image'])
                fig.canvas.draw()

        fig, ax = plt.subplots()
        ax.index = 0
        ax.set_title(self.core.frame(self.f)['time'])

        img = ax.imshow(
            self.core.frame(self.f)['image'],
            cmap='gray',
            zorder=0
        )

        fig.canvas.mpl_connect('key_press_event', button_press)
