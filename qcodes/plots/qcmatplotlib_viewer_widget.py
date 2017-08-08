from matplotlib.widgets import Cursor
import mplcursors
import numpy as np
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
from .base import BasePlot

class ClickWidget(BasePlot):

    def __init__(self, dataset):
        super().__init__()
        data = {}
        self.expand_trace(args=[dataset], kwargs=data)
        self.traces = []

        data['xlabel'] = self.get_label(data['x'])
        data['ylabel'] = self.get_label(data['y'])
        data['zlabel'] = self.get_label(data['z'])
        data['xaxis'] = data['x'].ndarray[0, :]
        data['yaxis'] = data['y'].ndarray
        self.traces.append({
            'config': data,
        })
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
        self.crossbtn = QtWidgets.QCheckBox('Cross section')
        self.crossbtn.setToolTip("Display extra subplots with selectable cross sections "
                                 "or sums along axis.")
        self.sumbtn = QtWidgets.QCheckBox('Sum')
        self.sumbtn.setToolTip("Display sums or cross sections.")

        self.savehmbtn = QtWidgets.QPushButton('Save Heatmap')
        self.savehmbtn.setToolTip("Save heatmap as a file (PDF)")
        self.savexbtn = QtWidgets.QPushButton('Save Vert')
        self.savexbtn.setToolTip("Save vertical cross section or sum as a file (PDF)")
        self.saveybtn = QtWidgets.QPushButton('Save Horz')
        self.savexbtn.setToolTip("Save horizontal cross section or sum as a file (PDF)")

        self.crossbtn.toggled.connect(self.toggle_cross)
        self.sumbtn.toggled.connect(self.toggle_sum)

        self.savehmbtn.pressed.connect(self.save_heatmap)
        self.savexbtn.pressed.connect(self.save_subplot_x)
        self.saveybtn.pressed.connect(self.save_subplot_y)

        self.toggle_cross()
        self.toggle_sum()

        vbox.addItem(vspace)
        vbox.addWidget(self.crossbtn)
        vbox.addWidget(self.sumbtn)
        vbox.addWidget(self.savehmbtn)
        vbox.addWidget(self.savexbtn)
        vbox.addWidget(self.saveybtn)

        hbox.addLayout(vbox)

    @staticmethod
    def full_extent(ax, pad=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles."""
        # for text objects we only include them if they are non empty.
        # empty ticks may be rendered outside the figure
        from matplotlib.transforms import Bbox
        items = []
        items += [ax.xaxis.label, ax.yaxis.label, ax.title]
        items = [item for item in items if item.get_text()]
        items.append(ax)
        bbox = Bbox.union([item.get_window_extent() for item in items])

        return bbox.expanded(1.0 + pad, 1.0 + pad)

    def save_subplot(self, axnumber, savename, saveformat='pdf'):
        extent = self.full_extent(self.ax[axnumber]).transformed(self.fig.dpi_scale_trans.inverted())
        full_title = "{}.{}".format(savename, saveformat)
        self.fig.savefig(full_title, bbox_inches=extent)

    def save_subplot_x(self):
        title = self.get_default_title()
        label, unit = self._get_label_and_unit(self.traces[0]['config']['xlabel'])
        if self.sumbtn.isChecked():
            title += " sum over {}".format(label)
        else:
            title += " cross section {} = {} {}".format(label,
                                                        self.traces[0]['config']['xpos'],
                                                        unit)
        self.save_subplot(axnumber=(0, 1), savename=title)

    def save_subplot_y(self):
        title = self.get_default_title()
        label, unit = self._get_label_and_unit(self.traces[0]['config']['ylabel'])
        if self.sumbtn.isChecked():
            title += " sum over {}".format(label)
        else:
            title += " cross section {} = {} {}".format(label,
                                                        self.traces[0]['config']['xpos'],
                                                        unit)
        self.save_subplot(axnumber=(1, 0), savename=title)

    def save_heatmap(self):
        title = self.get_default_title() + " heatmap"
        self.save_subplot(axnumber=(0, 0), savename=title)

    def _update_label(self, ax, axletter, label, extra=None):

        if type(label) == tuple and len(label) == 2:
            label, unit = label
        else:
            unit = ""
        axsetter = getattr(ax, "set_{}label".format(axletter))
        if extra:
            axsetter(extra + "{} ({})".format(label, unit))
        else:
            axsetter("{} ({})".format(label, unit))

    @staticmethod
    def _get_label_and_unit(config):
        if type(config) == tuple and len(config) == 2:
            label, unit = config
        else:
            unit = ""
            label = config
        return label, unit

    def toggle_cross(self):
        self.remove_plots()
        self.fig.clear()
        if self._cid:
            self.fig.canvas.mpl_disconnect(self._cid)
        if self.crossbtn.isChecked():
            self.sumbtn.setEnabled(True)
            self.savexbtn.setEnabled(True)
            self.saveybtn.setEnabled(True)
            self.ax = np.empty((2, 2), dtype='O')
            self.ax[0, 0] = self.fig.add_subplot(2, 2, 1)
            self.ax[0, 1] = self.fig.add_subplot(2, 2, 2)
            self.ax[1, 0] = self.fig.add_subplot(2, 2, 3)
            self._cid = self.fig.canvas.mpl_connect('button_press_event', self._click)
            self._cursor = Cursor(self.ax[0, 0], useblit=True, color='black')
            self.toggle_sum()
            figure_rect = (0, 0, 1, 1)
        else:
            self.sumbtn.setEnabled(False)
            self.savexbtn.setEnabled(False)
            self.saveybtn.setEnabled(False)
            self.ax = np.empty((1, 1), dtype='O')
            self.ax[0, 0] = self.fig.add_subplot(1, 1, 1)
            figure_rect = (0, 0.0, 0.75, 1)
        self.ax[0, 0].pcolormesh(self.traces[0]['config']['x'],
                                 self.traces[0]['config']['y'],
                                 self.traces[0]['config']['z'],
                                 edgecolor='face')
        self._update_label(self.ax[0, 0], 'x', self.traces[0]['config']['xlabel'])
        self._update_label(self.ax[0, 0], 'y', self.traces[0]['config']['ylabel'])
        self.fig.tight_layout(rect=figure_rect)
        self.fig.canvas.draw_idle()

    def toggle_sum(self):
        self.remove_plots()
        if not self.crossbtn.isChecked():
            return
        self.ax[1, 0].clear()
        self.ax[0, 1].clear()
        if self.sumbtn.isChecked():
            self._cursor.set_active(False)
            self.ax[1, 0].set_ylim(0, self.traces[0]['config']['z'].sum(axis=0).max() * 1.05)
            self.ax[0, 1].set_xlim(0, self.traces[0]['config']['z'].sum(axis=1).max() * 1.05)
            self._update_label(self.ax[1, 0], 'x', self.traces[0]['config']['xlabel'])
            self._update_label(self.ax[1, 0], 'y', self.traces[0]['config']['zlabel'], extra='sum of ')

            self._update_label(self.ax[0, 1], 'x', self.traces[0]['config']['zlabel'], extra='sum of ')
            self._update_label(self.ax[0, 1], 'y', self.traces[0]['config']['ylabel'])

            self._lines.append(self.ax[0, 1].plot(self.traces[0]['config']['z'].sum(axis=1),
                                                  self.traces[0]['config']['yaxis'],
                                                  color='C0',
                                                  marker='.')[0])
            self.ax[0, 1].set_title("")
            self._lines.append(self.ax[1, 0].plot(self.traces[0]['config']['xaxis'],
                                                  self.traces[0]['config']['z'].sum(axis=0),
                                                  color='C0',
                                                  marker='.')[0])
            self.ax[1, 0].set_title("")
            self._datacursor = mplcursors.cursor(self._lines, multiple=False)
        else:
            self._cursor.set_active(True)
            self._update_label(self.ax[1, 0], 'x', self.traces[0]['config']['xlabel'])
            self._update_label(self.ax[1, 0], 'y', self.traces[0]['config']['zlabel'])

            self._update_label(self.ax[0, 1], 'x', self.traces[0]['config']['zlabel'])
            self._update_label(self.ax[0, 1], 'y', self.traces[0]['config']['ylabel'])

            self.ax[1, 0].set_ylim(0, self.traces[0]['config']['z'].max() * 1.05)
            self.ax[0, 1].set_xlim(0, self.traces[0]['config']['z'].max() * 1.05)
        self.fig.canvas.draw_idle()

    def remove_plots(self):
        for line in self._lines:
            line.remove()
        self._lines = []
        if self._datacursor:
            self._datacursor.remove()

    def _click(self, event):

        if event.inaxes == self.ax[0, 0] and not self.sumbtn.isChecked():
            xpos = (abs(self.traces[0]['config']['xaxis'] - event.xdata)).argmin()
            ypos = (abs(self.traces[0]['config']['yaxis'] - event.ydata)).argmin()
            self.remove_plots()

            self._lines.append(self.ax[0, 1].plot(self.traces[0]['config']['z'][:, xpos],
                                                  self.traces[0]['config']['yaxis'],
                                                  color='C0',
                                                  marker='.')[0])
            xlabel, xunit = self._get_label_and_unit(self.traces[0]['config']['xlabel'])
            self.ax[0, 1].set_title("{} = {} {} ".format(xlabel, self.traces[0]['config']['xaxis'][xpos], xunit),
                                    fontsize='small')
            self.traces[0]['config']['xpos'] = self.traces[0]['config']['xaxis'][xpos]
            self._lines.append(self.ax[1, 0].plot(self.traces[0]['config']['xaxis'],
                                                  self.traces[0]['config']['z'][ypos, :],
                                                  color='C0',
                                                  marker='.')[0])
            ylabel, yunit = self._get_label_and_unit(self.traces[0]['config']['ylabel'])
            self.ax[1, 0].set_title("{} = {} {}".format(ylabel, self.traces[0]['config']['yaxis'][ypos], yunit),
                                    fontsize='small')
            self.traces[0]['config']['ypos'] = self.traces[0]['config']['yaxis'][ypos]
            self._datacursor = mplcursors.cursor(self._lines, multiple=False)
            self.fig.canvas.draw_idle()
