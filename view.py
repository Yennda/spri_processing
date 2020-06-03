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
        self.locations = []

    def add_core(self, core):
        core.synchronize()
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
        def frame_info():
            return '{}/{} |  t = {:.1f} s | dt = {:.2f} s | global time = {:.1f} min'.format(
                self.f,
                self.length,
                self.core_list[0]._time_info[self.f][0],
                self.core_list[0]._time_info[self.f][1],
                self.core_list[0]._time_info[self.f][0]/60 + self.core_list[0].zero_time
            )

        def next_frame(df):
            self.f = df
            for location in self.locations:
                location.xy = [self.f, axes_info[0, 0].get_ylim()[0]]

            fig_info.canvas.draw()

        def add_time_bar(ax):
            rectangle_height = np.abs(
                ax.get_ylim()[1] -
                ax.get_ylim()[0]
            )
            location = mpatches.Rectangle(
                (self.f, ax.get_ylim()[0]),
                1,
                rectangle_height,
                color=gray
            )
            self.locations.append(location)
            ax.add_patch(location)

        def change_type(event, itype):
            if event.inaxes is not None:
                event.inaxes.core.type = itype
                if self.orientation:
                    event.inaxes.set_ylabel('channel {}.; {}'.format(core.file[-1:], event.inaxes.core.type))
                else:
                    event.inaxes.set_xlabel('channel {}.; {}'.format(core.file[-1:], event.inaxes.core.type))

        def mouse_scroll(event):
            fig = event.canvas.figure
            if event.button == 'down':
                next_frame(1)
            elif event.button == 'up':
                next_frame(-1)

            fig.suptitle(frame_info())

            for i, core in enumerate(self.core_list):
                img_shown[i].set_array(core.frame(self.f)['image'])
            fig.canvas.draw()

        def button_press(event):
            fig = event.canvas.figure
            if event.key == '9':
                next_frame(100)
            elif event.key == '7':
                next_frame(-100)
            elif event.key == '6':
                next_frame(10)
            elif event.key == '4':
                next_frame(-10)
            elif event.key == '3':
                next_frame(1)
            elif event.key == '1':
                next_frame(-1)
            elif event.key == '5':
                img = event.inaxes.get_images()[0]
                lim = [i * 1.2 for i in img.get_clim()]
                img.set_clim(lim)
            elif event.key == '8':
                img = event.inaxes.get_images()[0]
                lim = [i / 1.2 for i in img.get_clim()]
                img.set_clim(lim)
            elif event.key == 'ctrl+1':
                change_type(event, 'int')

            elif event.key == 'ctrl+2':
                change_type(event, 'diff')

            elif event.key == 'i':
                if event.inaxes is not None:
                    event.inaxes.core.ref_frame = self.f

            fig.suptitle(frame_info())
            for i, core in enumerate(self.core_list):
                img_shown[i].set_array(core.frame(self.f)['image'])
            fig.canvas.draw()

        if self.orientation:
            fig, axes = plt.subplots(ncols=len(self.core_list), nrows=1)
        else:
            fig, axes = plt.subplots(nrows=len(self.core_list), ncols=1)

        fig.suptitle(frame_info())

        img_shown = []

        for i, core in enumerate(self.core_list):
            frame = core.frame(self.f)
            img_shown.append(
                axes[i].imshow(
                    frame['image'],
                    cmap='gray',
                    zorder=0,
                    vmin=core.range[0],
                    vmax=core.range[1]
                )
            )
            axes[i].core = core

            if self.orientation:
                axes[i].set_ylabel('channel {}.; {}'.format(core.file[-1:], core.type))
            else:
                axes[i].set_xlabel('channel {}.; {}'.format(core.file[-1:], core.type))
            for s in SIDES:
                axes[i].spines[s].set_color(COLORS[i])

        fig.canvas.mpl_connect('key_press_event', button_press)
        fig.canvas.mpl_connect('scroll_event', mouse_scroll)

        fig_info, axes_info = plt.subplots(2, 2, figsize=(10, 3))

        fig_info.suptitle('info')

        axes_info[0 ,0].set_title('spr signal')
        axes_info[0, 0].set_xlabel('frame')
        axes_info[0, 0].set_ylabel('R [a.u.]')

        for i, core in enumerate(self.core_list):
            axes_info[0, 0].plot(
                core.spr_signal - core.spr_signal[0] + 1,
                linewidth=1,
                color=COLORS[i],
                alpha=0.5,
                label='channel {}.'.format(i)
            )

        add_time_bar(axes_info[0, 0])




        axes_info[0, 1].set_title('NP count')
        axes_info[0, 1].set_xlabel('frame')
        axes_info[0, 1].set_ylabel('#')

        for i, core in enumerate(self.core_list):
            axes_info[0, 1].plot(
                core.spr_signal - core.spr_signal[0] + 1,
                linewidth=1,
                color=COLORS[i],
                alpha=0.5,
                label='channel {}.'.format(i)
            )
        add_time_bar(axes_info[0, 1])

