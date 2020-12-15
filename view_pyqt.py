import numpy as np
from PIL import Image
import os

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backend_bases import key_press_handler

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from global_var import *

matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Palatino Linotype')
matplotlib.rc('text', usetex='false')
# matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Palatino Linotype'
matplotlib.rcParams['mathtext.it'] = 'Palatino Linotype:italic'
matplotlib.rcParams['mathtext.bf'] = 'BiPalatino Linotype:bold'


class Canvas(FigureCanvasQTAgg):
    def __init__(self, view, img=True):
        self.view = view
        self.nav_toolbar = None
        if img:
            super(Canvas, self).__init__(view.fig)
            self.mpl_connect('axes_enter_event', self.mouse_enter)
            self.mpl_connect('axes_leave_event', self.mouse_leave)

        else:
            super(Canvas, self).__init__(view.fig_info)
            self.mpl_connect('button_press_event', self.mouse_click_spr)

        self.mpl_connect('key_press_event', self.button_press)
        self.mpl_connect('scroll_event', self.mouse_scroll)


    def next_frame(self, df):
        self.view.f = df
        for location in self.view.locations:
            y = location.get_y()
            location.xy = [self.view.f, y]

        self.view.fig.suptitle(self.view.frame_info())
        for i, core in enumerate(self.view.core_list):
            self.view.img_shown[i].set_array(core.frame(self.view.f))

        self.view.canvas_img.draw()
        if not self.view.canvas_plot is None:
            self.view.canvas_plot.draw()

    def mouse_click_spr(self, event):
        if event.button == 1:
            self.next_frame(int(round(event.xdata)) - self.view.f)

    def mouse_enter(self, event):
        if event.inaxes is not None:
            for s in SIDES:
                event.inaxes.spines[s].set_linewidth(4)
        self.draw()

    def mouse_leave(self, event):
        if event.inaxes is not None:
            for s in SIDES:
                event.inaxes.spines[s].set_linewidth(2)
        self.draw()

    def mouse_scroll(self, event):
        if event.button == 'down':
            self.next_frame(1)
        elif event.button == 'up':
            self.next_frame(-1)

    def save_frame(self, ax):
        """
        checks and eventually creates the folder
        'export_image' in the folder of data
        """
        if not os.path.isdir(ax.core.folder + FOLDER_EXPORTS):
            os.mkdir(ax.core.folder + FOLDER_EXPORTS)

        # creates the name, appends the rigth numeb at the end

        name = '{}/{}_f{:04.0f}'.format(
            ax.core.folder + FOLDER_EXPORTS,
            ax.core.file,
            self.view.f
        )

        i = 1
        while os.path.isfile(name + '_{:02d}.png'.format(i)):
            i += 1
        name += '_{:02d}'.format(i)

        # fig.savefig(
        #     name + '.png',
        #     bbox_inches='tight',
        #     transparent=True,
        #     pad_inches=0,
        #     pi=300
        # )
        img = ax.get_images()[0]
        xlim = [int(i) for i in ax.get_xlim()]
        ylim = [int(i) for i in ax.get_ylim()]

        current = img.get_array()[
                  ylim[1]: ylim[0],
                  xlim[0]: xlim[1]
                  ]

        current = (current - img.get_clim()[0]) / (img.get_clim()[1] - img.get_clim()[0]) * 256
        current = current.astype(np.uint8)

        print(current[20, 20])

        pilimage = Image.fromarray(current)
        pilimage.convert("L")

        pilimage.save(name + '.png', 'png')

        print('File SAVED @{}'.format(name))

    def button_press(self, event):
        key_press_handler(event, self, self.toolbar)

        def set_range(rng):
            img = event.inaxes.get_images()[0]
            img.set_clim(event.inaxes.core.range)

        if event.key == '9':
            self.next_frame(100)
        elif event.key == '7':
            self.next_frame(-100)
        elif event.key == '6':
            self.next_frame(10)
        elif event.key == '4':
            self.next_frame(-10)
        elif event.key == '3':
            self.next_frame(1)
        elif event.key == '1':
            self.next_frame(-1)

        if event.canvas.figure is self.view.fig and event.inaxes is not None:
            if event.key == '5':
                event.inaxes.core.range = [i * 1.2 for i in event.inaxes.core.range]
                print('core: {}, range: {}'.format(event.inaxes.core.file, event.inaxes.core.range))
                set_range(event.inaxes.core.range)

            elif event.key == '8':
                event.inaxes.core.range = [i / 1.2 for i in event.inaxes.core.range]
                print('core: {}, range: {}'.format(event.inaxes.core.file, event.inaxes.core.range))
                set_range(event.inaxes.core.range)

            elif event.key == 'ctrl+1':
                self.view.change_type(event, 'raw')
                set_range(event.inaxes.core.range)
                self.next_frame(0)

            elif event.key == 'ctrl+2':
                self.view.change_type(event, 'int')
                set_range(event.inaxes.core.range)
                self.next_frame(0)

            elif event.key == 'ctrl+3':
                self.view.change_type(event, 'diff')
                set_range(event.inaxes.core.range)
                self.next_frame(0)

            elif event.key == 'i':
                event.inaxes.core.ref_frame = self.view.f
                self.next_frame(0)

            elif event.key == 'a':
                self.save_frame(event.inaxes)
        self.draw()


class View(object):
    def __init__(self):

        self.core_list = []
        self._f = 0
        self.orientation = True
        self.length = 0
        self.locations = []

        self.fig_info = None
        self.axes_info = None
        self.fig = None
        self.axes = None

        self.img_shown = []
        self.canvas_img = None
        self.canvas_plot = None
        self.plots = [
            {
                'key': 'spr_signal',
                'title': 'spr signal',
                'xlabel': 'frame',
                'ylabel': 'R [a.u.]'
            },
            {
                'key': 'intensity_raw',
                'title': 'Raw intensity per px',
                'xlabel': 'frame',
                'ylabel': 'I/area [a.u./px]'
            },
            {
                'key': 'intensity_int',
                'title': 'Int. intensity per px, normalized by laser intensity',
                'xlabel': 'frame',
                'ylabel': 'I/area [a.u./px]'
            },
            {
                'key': 'std_int',
                'title': 'Std of int., normalized by laser intensity',
                'xlabel': 'frame',
                'ylabel': 'std [a.u./]'
            }
        ]

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

    def frame_info(self):
        return '{}/{} |  t = {:.1f} s | dt = {:.2f} s | global time = {:.1f} min'.format(
            self.f,
            self.length,
            self.core_list[0]._time_info[self.f][0],
            self.core_list[0]._time_info[self.f][1],
            self.core_list[0]._time_info[self.f][0] / 60 + self.core_list[0].zero_time
        )

    def change_type(self, event, itype):
        if event.inaxes is not None:
            event.inaxes.core.type = itype
            if self.orientation:
                event.inaxes.set_ylabel('channel {}.; {}'.format(event.inaxes.core.file[-1:], event.inaxes.core.type))
            else:
                event.inaxes.set_xlabel('channel {}.; {}'.format(event.inaxes.core.file[-1:], event.inaxes.core.type))

    def mouse_click_spr(self, event):
        if event.button == 1:
            self.next_frame(int(round(event.xdata)) - self.f)

    def show_img(self):

        if len(self.core_list) == 1:
            self.fig, axes = plt.subplots()
            self.axes = [axes]
        else:
            if self.orientation:
                self.fig, self.axes = plt.subplots(ncols=len(self.core_list), nrows=1)
            else:
                self.fig, self.axes = plt.subplots(nrows=len(self.core_list), ncols=1)

        self.fig.suptitle(self.frame_info())

        for i, core in enumerate(self.core_list):
            self.img_shown.append(
                self.axes[i].imshow(
                    core.frame(self.f),
                    cmap='gray',
                    zorder=0,
                    vmin=core.range[0],
                    vmax=core.range[1]
                )
            )
            self.axes[i].core = core

            fontprops = fm.FontProperties(size=20)
            show_scalebar = AnchoredSizeBar(self.axes[i].transData,
                                            34, '100 $\mu m$', 'lower right',
                                            pad=0.1,
                                            color='black',
                                            frameon=False,
                                            size_vertical=1,
                                            fontproperties=fontprops)

            self.axes[i].add_artist(show_scalebar)

            if self.orientation:
                self.axes[i].set_ylabel('channel {}.; {}'.format(core.file[-1:], core.type))
            else:
                self.axes[i].set_xlabel('channel {}.; {}'.format(core.file[-1:], core.type))
            for s in SIDES:
                self.axes[i].spines[s].set_color(COLORS[i])
                self.axes[i].spines[s].set_linewidth(2)

        self.canvas_img = Canvas(self)
        return self.canvas_img

    def show_plots(self, chosen_plots):
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

        chosen_plot_indices = [i for i in range(len(chosen_plots)) if chosen_plots[i]]
        number_of_plots = len(chosen_plot_indices)

        if number_of_plots == 1:
            self.fig_info, self.axes_info = plt.subplots(figsize=(10, 3))
            axes_plot_list = [self.axes_info]
        elif number_of_plots == 2:
            self.fig_info, self.axes_info = plt.subplots(2, 1, figsize=(10, 3))
            axes_plot_list = self.axes_info[:]
        elif number_of_plots == 3:
            self.fig_info, self.axes_info = plt.subplots(3, 1, figsize=(10, 3))
            axes_plot_list = self.axes_info[:]
        elif number_of_plots == 4:
            self.fig_info, self.axes_info = plt.subplots(2, 2, figsize=(10, 3))
            axes_plot_list = list(self.axes_info[0, :]) + list(self.axes_info[1, :])

        self.fig_info.suptitle('info')

        for k, axes in enumerate(axes_plot_list):
            i = chosen_plot_indices[k]

            axes.set_title(self.plots[i]['title'])
            axes.set_xlabel(self.plots[i]['xlabel'])
            axes.set_ylabel(self.plots[i]['ylabel'])

            for j, core in enumerate(self.core_list):
                if self.plots[i]['key'] == 'spr_signal':
                    axes.plot(
                        core.graphs['spr_signal'] - core.graphs['spr_signal'][0] + 1,
                        linewidth=1,
                        color=COLORS[j],
                        alpha=0.5,
                        label='channel {}.'.format(j)
                    )
                else:
                    axes.plot(
                        core.graphs[self.plots[i]['key']],
                        linewidth=1,
                        color=COLORS[j],
                        alpha=0.5,
                        label='channel {}.'.format(j)
                    )

            add_time_bar(axes)

        # self.axes_info[0, 1].set_title('Raw intensity per px')
        # self.axes_info[0, 1].set_xlabel('frame')
        # self.axes_info[0, 1].set_ylabel('I/area [a.u./px]')
        #
        # for i, core in enumerate(self.core_list):
        #     self.axes_info[0, 1].plot(
        #         core.graphs['intensity_raw'],
        #         linewidth=1,
        #         color=COLORS[i],
        #         alpha=0.5,
        #         label='channel {}.'.format(i)
        #     )
        # add_time_bar(self.axes_info[0, 1])
        #
        # self.axes_info[1, 0].set_title('Int. intensity per px, normalized by laser intensity')
        # self.axes_info[1, 0].set_xlabel('frame')
        # self.axes_info[1, 0].set_ylabel('I/area [a.u./px]')
        #
        # for i, core in enumerate(self.core_list):
        #     self.axes_info[1, 0].plot(
        #         core.graphs['intensity_int'],
        #         linewidth=1,
        #         color=COLORS[i],
        #         alpha=0.5,
        #         label='channel {}.'.format(i)
        #     )
        # add_time_bar(self.axes_info[1, 0])
        #
        # self.axes_info[1, 1].set_title('Std of int., normalized by laser intensity')
        # self.axes_info[1, 1].set_xlabel('frame')
        # self.axes_info[1, 1].set_ylabel('std [a.u./]')
        #
        # for i, core in enumerate(self.core_list):
        #     self.axes_info[1, 1].plot(
        #         core.graphs['std_int'],
        #         linewidth=1,
        #         color=COLORS[i],
        #         alpha=0.5,
        #         label='channel {}.'.format(i)
        #     )
        # add_time_bar(self.axes_info[1, 1])

        self.canvas_plot = Canvas(self, False)

        return self.canvas_plot
