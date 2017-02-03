"""
Live plotting in Jupyter notebooks
using the nbagg backend and matplotlib
"""
from collections import Mapping

from qtpy import QtWidgets

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.widgets import Cursor
import mplcursors


import numpy as np
from numpy.ma import masked_invalid, getmask

from .base import BasePlot


class MatPlot(BasePlot):
    """
    Plot x/y lines or x/y/z heatmap data. The first trace may be included
    in the constructor, other traces can be added with MatPlot.add()

    Args:
        *args: shortcut to provide the x/y/z data. See BasePlot.add

        figsize (Tuple[Float, Float]): (width, height) tuple in inches to pass to plt.figure
            default (8, 5)

        interval: period in seconds between update checks

        subplots: either a sequence (args) or mapping (kwargs) to pass to
            plt.subplots. default is a single simple subplot (1, 1)
            you can use this to pass kwargs to the plt.figure constructor

        num: integer or None
            specifies the index of the matplotlib figure window to use. If None
            then open a new window

        **kwargs: passed along to MatPlot.add() to add the first data trace
    """
    def __init__(self, *args, figsize=None, interval=1, subplots=None, num=None,
                 **kwargs):

        super().__init__(interval)

        self._init_plot(subplots, figsize, num=num)
        if args or kwargs:
            self.add(*args, **kwargs)

    def _init_plot(self, subplots=None, figsize=None, num=None):
        if figsize is None:
            figsize = (8, 5)

        if subplots is None:
            subplots = (1, 1)

        if isinstance(subplots, Mapping):
            self.fig, self.subplots = plt.subplots(figsize=figsize, num=num,
                                                   **subplots)
        else:
            self.fig, self.subplots = plt.subplots(*subplots, num=num,
                                                   figsize=figsize)
        if not hasattr(self.subplots, '__len__'):
            self.subplots = (self.subplots,)

        self.title = self.fig.suptitle('')

    def clear(self, subplots=None, figsize=None):
        """
        Clears the plot window and removes all subplots and traces
        so that the window can be reused.
        """
        self.traces = []
        self.fig.clf()
        self._init_plot(subplots, figsize, num=self.fig.number)

    def add_to_plot(self, **kwargs):
        """
        adds one trace to this MatPlot.

        kwargs: with the following exceptions (mostly the data!), these are
            passed directly to the matplotlib plotting routine.

            `subplot`: the 1-based axes number to append to (default 1)

            if kwargs include `z`, we will draw a heatmap (ax.pcolormesh):
                `x`, `y`, and `z` are passed as positional args to pcolormesh

            without `z` we draw a scatter/lines plot (ax.plot):
                `x`, `y`, and `fmt` (if present) are passed as positional args
        """
        # TODO some way to specify overlaid axes?
        ax = self._get_axes(kwargs)
        if 'z' in kwargs:
            plot_object = self._draw_pcolormesh(ax, **kwargs)
        else:
            plot_object = self._draw_plot(ax, **kwargs)

        self._update_labels(ax, kwargs)
        prev_default_title = self.get_default_title()

        self.traces.append({
            'config': kwargs,
            'plot_object': plot_object
        })

        if prev_default_title == self.title.get_text():
            # in case the user has updated title, don't change it anymore
            self.title.set_text(self.get_default_title())

    def _get_axes(self, config):
        return self.subplots[config.get('subplot', 1) - 1]

    def _update_labels(self, ax, config):
        if 'x' in config and not ax.get_xlabel():
            ax.set_xlabel(self.get_label(config['x']))
        if 'y' in config and not ax.get_ylabel():
            ax.set_ylabel(self.get_label(config['y']))

    def update_plot(self):
        """
        update the plot. The DataSets themselves have already been updated
        in update, here we just push the changes to the plot.
        """
        # matplotlib doesn't know how to autoscale to a pcolormesh after the
        # first draw (relim ignores it...) so we have to do this ourselves
        bboxes = dict(zip(self.subplots, [[] for p in self.subplots]))

        for trace in self.traces:
            config = trace['config']
            plot_object = trace['plot_object']
            if 'z' in config:
                # pcolormesh doesn't seem to allow editing x and y data, only z
                # so instead, we'll remove and re-add the data.
                if plot_object:
                    plot_object.remove()

                ax = self._get_axes(config)
                plot_object = self._draw_pcolormesh(ax, **config)
                trace['plot_object'] = plot_object

                if plot_object:
                    bboxes[plot_object.axes].append(
                        plot_object.get_datalim(plot_object.axes.transData))
            else:
                for axletter in 'xy':
                    setter = 'set_' + axletter + 'data'
                    if axletter in config:
                        getattr(plot_object, setter)(config[axletter])

        for ax in self.subplots:
            if ax.get_autoscale_on():
                ax.relim()
                if bboxes[ax]:
                    bbox = Bbox.union(bboxes[ax])
                    if np.all(np.isfinite(ax.dataLim)):
                        # should take care of the case of lines + heatmaps
                        # where there's already a finite dataLim from relim
                        ax.dataLim.set(Bbox.union(ax.dataLim, bbox))
                    else:
                        # when there's only a heatmap, relim gives inf bounds
                        # so just completely overwrite it
                        ax.dataLim = bbox
                ax.autoscale()

        self.fig.canvas.draw()

    def _draw_plot(self, ax, y, x=None, fmt=None, subplot=1, **kwargs):
        # NOTE(alexj)stripping out subplot because which subplot we're in is already
        # described by ax, and it's not a kwarg to matplotlib's ax.plot. But I
        # didn't want to strip it out of kwargs earlier because it should stay
        # part of trace['config'].
        args = [arg for arg in [x, y, fmt] if arg is not None]
        line, = ax.plot(*args, **kwargs)
        return line

    def _draw_pcolormesh(self, ax, z, x=None, y=None, subplot=1, **kwargs):
        # NOTE(alexj)stripping out subplot because which subplot we're in is already
        # described by ax, and it's not a kwarg to matplotlib's ax.plot. But I
        # didn't want to strip it out of kwargs earlier because it should stay
        # part of trace['config'].
        args = [masked_invalid(arg) for arg in [x, y, z]
                if arg is not None]

        for arg in args:
            if np.all(getmask(arg)):
                # if any entire array is masked, don't draw at all
                # there's nothing to draw, and anyway it throws a warning
                return False
        pc = ax.pcolormesh(*args, **kwargs)

        if getattr(ax, 'qcodes_colorbar', None):
            # update_normal doesn't seem to work...
            ax.qcodes_colorbar.update_bruteforce(pc)
        else:
            # TODO: what if there are several colormeshes on this subplot,
            # do they get the same colorscale?
            # We should make sure they do, and have it include
            # the full range of both.
            ax.qcodes_colorbar = self.fig.colorbar(pc, ax=ax)

            # ideally this should have been in _update_labels, but
            # the colorbar doesn't necessarily exist there.
            # I guess we could create the colorbar no matter what,
            # and just give it a dummy mappable to start, so we could
            # put this where it belongs.
            ax.qcodes_colorbar.set_label(self.get_label(z))

        return pc

    def save(self, filename=None):
        """
        Save current plot to filename, by default
        to the location corresponding to the default 
        title.

        Args:
            filename (Optional[str]): Location of the file
        """
        default = "{}.png".format(self.get_default_title())
        filename = filename or default
        self.fig.savefig(filename)


class ClickWidget:
    def __init__(self, dataset):
        self._data = {}
        BasePlot.expand_trace(args=[dataset], kwargs=self._data)
        self._data['xlabel'] = BasePlot.get_label(self._data['x'])
        self._data['ylabel'] = BasePlot.get_label(self._data['y'])
        self._data['zlabel'] = BasePlot.get_label(self._data['z'])
        self._data['xaxis'] = self._data['x'].ndarray[0, :]
        self._data['yaxis'] = self._data['y'].ndarray

        self.fig = plt.figure()

        self._lines = []
        self._datacursor = []
        self._cid = 0

        hbox = QtWidgets.QHBoxLayout()
        self.fig.canvas.setLayout(hbox)
        hspace = QtWidgets.QSpacerItem(0,
                                       0,
                                       QtWidgets.QSizePolicy.Expanding,
                                       QtWidgets.QSizePolicy.Expanding)
        vspace = QtWidgets.QSpacerItem(0,
                                       0,
                                       QtWidgets.QSizePolicy.Minimum,
                                       QtWidgets.QSizePolicy.Expanding)
        hbox.addItem(hspace)

        vbox = QtWidgets.QVBoxLayout()
        self.crossbtn = QtWidgets.QCheckBox('Cross')
        self.sumbtn = QtWidgets.QCheckBox('Show Sum')

        self.crossbtn.toggled.connect(self.toggle_cross)
        self.sumbtn.toggled.connect(self.toggle_sum)
        self.toggle_cross()
        self.toggle_sum()

        vbox.addItem(vspace)
        vbox.addWidget(self.crossbtn)
        vbox.addWidget(self.sumbtn)

        hbox.addLayout(vbox)
        self.fig.tight_layout(rect=(0, 0, 0.9, 1))

    def toggle_cross(self):
        self.remove_plots()
        self.fig.clear()
        if self._cid:
            self.fig.canvas.mpl_disconnect(self._cid)
        if self.crossbtn.isChecked():
            self.ax = np.empty((2, 2), dtype='O')
            self.ax[0, 0] = self.fig.add_subplot(2, 2, 1)
            self.ax[0, 1] = self.fig.add_subplot(2, 2, 2)
            self.ax[1, 0] = self.fig.add_subplot(2, 2, 3)
            print("Axis")
            self._cid = self.fig.canvas.mpl_connect('button_press_event', self._click)
            self._cursor = Cursor(self.ax[0, 0], useblit=True, color='black')
            self.toggle_sum()
        else:
            self.ax = np.empty((1, 1), dtype='O')
            self.ax[0, 0] = self.fig.add_subplot(1, 1, 1)
        self.ax[0, 0].pcolormesh(self._data['x'],
                                 self._data['y'],
                                 self._data['z'])
        self.ax[0, 0].set_xlabel(self._data['xlabel'])
        self.ax[0, 0].set_ylabel(self._data['ylabel'])
        self.fig.tight_layout(rect=(0, 0.07, 0.9, 1))
        self.fig.canvas.draw_idle()

    def toggle_sum(self):
        self.remove_plots()
        if not self.crossbtn.isChecked():
            return
        if self.sumbtn.isChecked():
            self._cursor.set_active(False)
            self.ax[1, 0].set_ylim(0, self._data['z'].sum(axis=0).max() * 1.05)
            self.ax[0, 1].set_xlim(0, self._data['z'].sum(axis=1).max() * 1.05)
            self.ax[1, 0].set_xlabel(self._data['xlabel'])
            self.ax[1, 0].set_ylabel("sum of " + self._data['zlabel'])
            self.ax[0, 1].set_xlabel("sum of " + self._data['zlabel'])
            self.ax[0, 1].set_ylabel(self._data['ylabel'])
            self._lines.append(self.ax[0, 1].plot(self._data['z'].sum(axis=1),
                                                  self._data['yaxis'],
                                                  color='C0',
                                                  marker='.')[0])
            self._lines.append(self.ax[1, 0].plot(self._data['xaxis'],
                                                  self._data['z'].sum(axis=0),
                                                  color='C0',
                                                  marker='.')[0])
            self._datacursor = mplcursors.cursor(self._lines, multiple=False)
        else:
            self._cursor.set_active(True)
            self.ax[1, 0].set_xlabel(self._data['xlabel'])
            self.ax[1, 0].set_ylabel(self._data['zlabel'])
            self.ax[0, 1].set_xlabel(self._data['zlabel'])
            self.ax[0, 1].set_ylabel(self._data['ylabel'])
            self.ax[1, 0].set_ylim(0, self._data['z'].max() * 1.05)
            self.ax[0, 1].set_xlim(0, self._data['z'].max() * 1.05)
        self.fig.canvas.draw_idle()

    def remove_plots(self):
        for line in self._lines:
            line.remove()
        self._lines = []
        if self._datacursor:
            self._datacursor.remove()

    def _click(self, event):

        if event.inaxes == self.ax[0, 0] and not self.sumbtn.isChecked():
            xpos = (abs(self._data['xaxis'] - event.xdata)).argmin()
            ypos = (abs(self._data['yaxis'] - event.ydata)).argmin()
            print("X: {} Y: {}".format(event.xdata, event.ydata))
            print("Clicked on number {},{}".format(xpos, ypos))
            self.remove_plots()

            self._lines.append(self.ax[0, 1].plot(self._data['z'][:, xpos],
                                                  self._data['yaxis'],
                                                  color='C0',
                                                  marker='.')[0])
            self._lines.append(self.ax[1, 0].plot(self._data['xaxis'],
                                                  self._data['z'][ypos, :],
                                                  color='C0',
                                                  marker='.')[0])
            self._datacursor = mplcursors.cursor(self._lines, multiple=False)
            self.fig.canvas.draw()