import numpy as np
from PIL import Image
import os

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.widgets import RectangleSelector

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from PyQt5 import QtCore

from global_var import *
from gui_windows import SelectWindow

import tools as tl

matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='Palatino Linotype')
matplotlib.rc('text', usetex='false')
# matplotlib.rcParams.update({'font.size': 30})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Palatino Linotype'
matplotlib.rcParams['mathtext.it'] = 'Palatino Linotype:italic'
matplotlib.rcParams['mathtext.bf'] = 'Palatino Linotype:bold'


class Canvas(FigureCanvasQTAgg):
    def __init__(self, view, img=True):
        self.view = view
        self.nav_toolbar = None
        self.plot_select_window = None
        self.main_window = None

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
        self.view.next_frame(df)

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

    def save_frame(self, axes):
        """
        checks and eventually creates the folder
        'export_image' in the folder of data
        """
        if not os.path.isdir(axes.core.folder + FOLDER_EXPORTS):
            os.mkdir(axes.core.folder + FOLDER_EXPORTS)

        # creates the name, appends the rigth numeb at the end

        name = '{}/{}_f{:04.0f}'.format(
            axes.core.folder + FOLDER_EXPORTS,
            axes.core.file,
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
        img = axes.get_images()[0]
        xlim = [int(i) for i in axes.get_xlim()]
        ylim = [int(i) for i in axes.get_ylim()]

        current = img.get_array()[
                  ylim[1]: ylim[0],
                  xlim[0]: xlim[1]
                  ]

        current = (current - img.get_clim()[0]) / (img.get_clim()[1] - img.get_clim()[0]) * 256
        current = current.astype(np.uint8)

        pilimage = Image.fromarray(current)
        pilimage.convert("L")

        pilimage.save(name + '.png', 'png')

        print('File SAVED @{}'.format(name))

    def select_np_pattern(self, axes):
        xlim = [int(i) for i in axes.get_xlim()]
        ylim = [int(i) for i in axes.get_ylim()]

        img = axes.get_images()[0]
        raw_idea = img.get_array()[
                   ylim[1]: ylim[0],
                   xlim[0]: xlim[1]
                   ]

        def toggle_selector(event):
            pass

        fig_idea, axes_idea = plt.subplots()
        axes_idea.imshow(
            raw_idea,
            cmap='gray'
        )

        canvas_plot_select = FigureCanvasQTAgg(fig_idea)

        toggle_selector.RS = RectangleSelector(
            axes_idea,
            lambda eclick, erelease: self.handle_choose_idea(
                axes,
                (ylim[1], xlim[0]),
                eclick,
                erelease
            ),
            drawtype='box', useblit=True,
            button=[1, 3],  # don't use middle button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True
        )

        plt.connect('key_press_event', toggle_selector)

        self.plot_select_window = SelectWindow(canvas_plot_select)
        self.plot_select_window.show()
        self.plot_select_window.activateWindow()

    def handle_choose_idea(self, axes, shift, eclick, erelease):
        corner_1 = [tl.true_coordinate(b) for b in (eclick.xdata, eclick.ydata)]
        corner_2 = [tl.true_coordinate(e) for e in (erelease.xdata, erelease.ydata)]
        span_x = np.array([
            shift[1] + corner_1[0] + 1 - 2,
            shift[1] + corner_2[0] + 2
        ])
        span_y = np.array([
            shift[0] + corner_1[1] + 1 - 2,
            shift[0] + corner_2[1] + 2
        ])

        idea3d = np.zeros((span_y[1] - span_y[0], span_x[1] - span_x[0], 2 * axes.core.k))

        for i in range(2 * axes.core.k):
            idea3d[:, :, i] = axes.core.frame(self.view.f + axes.core.k - i)[
                              span_y[0]: span_y[1],
                              span_x[0]: span_x[1]]

        axes.core.idea3d = idea3d
        print('Pattern chosen')

        axes.core.save_idea()

    def button_press(self, event):
        key_press_handler(event, self, self.toolbar)

        def set_range(axes):
            img = axes.get_images()[0]
            img.set_clim(axes.core.range)
            for txt, c in zip(self.view.text, self.view.core_list):
                txt.set_text('rng = {:.4f}'.format(c.range[1]))

        if event.key == '9':
            self.next_frame(self.view.core_list[0].k * 10)
        elif event.key == '7':
            self.next_frame(-self.view.core_list[0].k * 10)
        elif event.key == '6':
            self.next_frame(self.view.core_list[0].k)
        elif event.key == '4':
            self.next_frame(-self.view.core_list[0].k)
        elif event.key == '3':
            self.next_frame(1)
        elif event.key == '1':
            self.next_frame(-1)
        elif event.key == 'f':
            self.main_window.filters_checkbox.click()

        if event.canvas.figure is self.view.fig and event.inaxes is not None:
            core_list = [event.inaxes.core]
            axes_list = [event.inaxes]

            if event.key == 'a':
                self.save_frame(event.inaxes)
            elif event.key == 'd':
                self.select_np_pattern(event.inaxes)

        else:
            core_list = self.view.core_list
            axes_list = self.view.axes

        for axes, core in zip(axes_list, core_list):

            if event.key == '5':
                core.range = [i * 1.2 for i in core.range]
                # print('core: {}, range: {}'.format(core.file, core.range))
                set_range(axes)

            elif event.key == '8':
                core.range = [i / 1.2 for i in core.range]
                # print('core: {}, range: {}'.format(core.file, core.range))
                set_range(axes)

            elif event.key == 'ctrl+1':
                self.view.change_type(axes, 'raw')
                set_range(axes)
                self.next_frame(0)

            elif event.key == 'ctrl+2':
                self.view.change_type(axes, 'int')
                set_range(axes)
                self.next_frame(0)

            elif event.key == 'ctrl+3':
                self.view.change_type(axes, 'diff')
                set_range(axes)
                self.next_frame(0)

            elif event.key == 'ctrl+5':
                self.view.change_type(axes, 'diff')
                set_range(axes)
                self.next_frame(0)
                self.main_window.filters_checkbox.click()

            elif event.key == 'ctrl+6':
                self.view.change_type(axes, 'corr')
                set_range(axes)
                self.next_frame(0)
                self.main_window.filters_checkbox.click()

            elif event.key == 'alt+1':
                self.view.change_type(axes, 'four_r')
                set_range(axes)
                self.next_frame(0)

            elif event.key == 'alt+2':
                self.view.change_type(axes, 'four_i')
                set_range(axes)
                self.next_frame(0)

            elif event.key == 'alt+3':
                self.view.change_type(axes, 'four_d')
                set_range(axes)
                self.next_frame(0)

            elif event.key == 'alt+4':
                self.view.change_type(axes, 'mask')
                set_range(axes)
                self.next_frame(0)

            elif event.key == 'ctrl+4':
                self.view.change_type(axes, 'corr')
                set_range(axes)
                self.next_frame(0)

            elif event.key == 'i':
                core.ref_frame = self.view.f
                self.next_frame(0)

        self.draw()


class View(object):
    def __init__(self, main_window):
        self.main_window = main_window
        self.core_list = []
        self._f = 0
        self.orientation = True
        self.length = 0
        self.locations = []

        self.fig_info = None
        self.axes_info = None
        self.fig = None
        self.axes = None
        self.text = []
        self.chosen_plot_indices = None
        self.axes_plot_list = None

        self.idea3d = None

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
                'key': 'histogram',
                'title': 'Histogram',
                'xlabel': 'value',
                'ylabel': 'count'
            },
            {
                'key': 'nps_pos',
                'title': 'Counts of NPs',
                'xlabel': 'frame',
                'ylabel': 'Count'
            }
        ]

    def add_core(self, core):
        # core.synchronize()
        self.core_list.append(core)
        if self.length > len(core) or self.length == 0:
            self.length = len(core)

    def next_frame(self, df):
        self.f = df
        for location in self.locations:
            y = location.get_y()
            location.xy = [self.f, y]

        self.fig.suptitle(self.frame_info())

        for i, core in enumerate(self.core_list):
            self.img_shown[i].set_array(core.frame(self.f))

            if 2 in self.chosen_plot_indices:
                axes = self.axes_plot_list[self.chosen_plot_indices.index(2)]
                for j, core in enumerate(self.core_list):
                    values, counts = core.histogram()
                    axes.clear()
                    axes.bar(
                        values,
                        counts,
                        width=values[1] - values[0],
                        color=COLORS[j],
                        alpha=0.5,
                        label='channel {}.'.format(j)
                    )

            if core.show_nps:
                positions, colors = core.frame_np(self.f)
                [p.remove() for p in self.axes[i].patches]

                for (p, c) in zip(positions, colors):
                    circle = mpatches.Circle(
                        p,
                        5,
                        color=c,
                        fill=False,
                        alpha=0.7,
                        lw=2)
                    self.axes[i].add_patch(circle)

        self.canvas_img.draw()
        if self.canvas_plot is not None:
            self.canvas_plot.draw()

        self.main_window.RefreshNPInfo()

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
            self.core_list[0]._time_info[self.f, 0],
            self.core_list[0]._time_info[self.f, 1],
            self.core_list[0]._time_info[self.f, 0] / 60 + self.core_list[0].zero_time
        )

    def change_type(self, axes, itype):
        if axes is not None:
            axes.core.type = itype
            if self.orientation:
                axes.set_ylabel('channel {}. | {}'.format(axes.core.file[-1:], axes.core.type))
            else:
                axes.set_xlabel('channel {}. | {}'.format(axes.core.file[-1:], axes.core.type))
        # else:
        #     for core, axes in zip(self.core_list, self.axes):
        #         core.type = itype
        #         if self.orientation:
        #             axes.set_ylabel('channel {}. | {}'.format(axes.core.file[-1:], axes.core.type))
        #         else:
        #             axes.set_xlabel('channel {}. | {}'.format(axes.core.file[-1:], axes.core.type))


    def mouse_click_spr(self, event):
        if event.button == 1:
            self.next_frame(int(round(event.xdata)) - self.f)

    def set_range(self):
        for axes in self.axes:
            img = axes.get_images()[0]
            img.set_clim(axes.core.range)
            for txt, c in zip(self.text, self.core_list):
                txt.set_text('rng = {:.4f}'.format(c.range[1]))

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

            self.text.append(self.axes[i].text(
                0.1,
                0.1,
                'rng = {}'.format(core.range[1]),
                color='white',
                backgroundcolor='black',
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.axes[i].transAxes
            ))

            self.axes[i].add_artist(show_scalebar)
            # self.axes[i].add_artist(text)

            axis_font = {'size': '14'}

            if self.orientation:
                self.axes[i].set_ylabel('channel {}. | {}'.format(core.file[-1:], core.type), **axis_font)
            else:
                self.axes[i].set_xlabel('channel {}. | {}'.format(core.file[-1:], core.type), **axis_font)

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

        if self.core_list[0].np_container is not None:
            chosen_plots[3] = True

        self.chosen_plot_indices = [i for i in range(len(chosen_plots)) if chosen_plots[i]]
        number_of_plots = len(self.chosen_plot_indices)

        if number_of_plots == 1:
            self.fig_info, self.axes_info = plt.subplots(figsize=(10, 3))
            self.axes_plot_list = [self.axes_info]
        elif number_of_plots == 2:
            self.fig_info, self.axes_info = plt.subplots(2, 1, figsize=(10, 3))
            self.axes_plot_list = self.axes_info[:]
        elif number_of_plots == 3:
            self.fig_info, self.axes_info = plt.subplots(3, 1, figsize=(10, 3))
            self.axes_plot_list = self.axes_info[:]
        elif number_of_plots == 4:
            self.fig_info, self.axes_info = plt.subplots(2, 2, figsize=(10, 3))
            self.axes_plot_list = list(self.axes_info[0, :]) + list(self.axes_info[1, :])

        self.fig_info.suptitle('info')

        for k, axes in enumerate(self.axes_plot_list):
            i = self.chosen_plot_indices[k]

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
                    add_time_bar(axes)

                elif self.plots[i]['key'] == 'histogram':
                    values, counts = core.histogram()
                    axes.bar(
                        values,
                        counts,
                        color=COLORS[j],
                        alpha=0.5,
                        label='channel {}.'.format(j)
                    )

                elif self.plots[i]['key'] == 'nps_pos':

                    nps_add = [sum(core.graphs['nps_pos'][:i]) for i in range(len(core))]

                    axes.plot(
                        nps_add,
                        linewidth=2,
                        color=COLORS[j],
                        alpha=0.5,
                        label='channel {}.'.format(j)
                    )

                    axes.bar(
                        np.arange(0, len(core), 1),
                        core.graphs['nps_pos'],
                        linewidth=1,
                        color=COLORS[j],
                        alpha=0.5,
                        label='channel {}.'.format(j)
                    )

                    add_time_bar(axes)
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
