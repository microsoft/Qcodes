"""
Live plotting using pyqtgraph
"""

import numpy as np
import itertools

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QWidget, QShortcut, QHBoxLayout
from qtpy.QtCore import QBuffer, QIODevice, QByteArray

import pyqtgraph as pg
from pyqtgraph import dockarea

import warnings
from collections import namedtuple

from .base import BasePlot
from .colors import color_cycle, colorscales

pg.mkQApp()


# Subclass of pyqtgraph Dock to change style and to add png representation for
# interactive ipython environments
class Dock(dockarea.Dock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def updateStyle():
            r = '2px'
            if self.label.dim:
                # This is the background-tab
                fg = '#888'
                bg = '#ddd'
                border = '#ccc'
                border_px = '1px'
            else:
                fg = '#333'
                bg = '#ccc'
                border = '#888'
                border_px = '1px'

            if self.label.orientation == 'vertical':
                self.label.vStyle = """DockLabel {
                    background-color : %s;
                    color : %s;
                    border-top-right-radius: 0px;
                    border-top-left-radius: %s;
                    border-bottom-right-radius: 0px;
                    border-bottom-left-radius: %s;
                    border-width: 0px;
                    border-right: %s solid %s;
                    padding-top: 3px;
                    padding-bottom: 3px;
                }""" % (bg, fg, r, r, border_px, border)
                self.label.setStyleSheet(self.label.vStyle)
            else:
                self.label.hStyle = """DockLabel {
                    background-color : %s;
                    color : %s;
                    border-top-right-radius: %s;
                    border-top-left-radius: %s;
                    border-bottom-right-radius: 0px;
                    border-bottom-left-radius: 0px;
                    border-width: 0px;
                    border-bottom: %s solid %s;
                    padding-left: 3px;
                    padding-right: 3px;
                }""" % (bg, fg, r, r, border_px, border)
                self.label.setStyleSheet(self.label.hStyle)
        self.label.updateStyle = updateStyle
        self.label.closeButton.setStyleSheet('border: none')

    def _repr_png_(self):
        """
        Create a png representation of the current Dock.
        """

        QtWidgets.QApplication.processEvents()

        image = self.grab()
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.ReadWrite)
        image.save(buffer, 'PNG')
        buffer.close()

        return bytes(byte_array)

TransformState = namedtuple('TransformState', 'translate scale revisit')


class QtPlot(QWidget, BasePlot):

    """
    Plot x/y lines or x/y/z heatmap data. The first trace may be included
    in the constructor, other traces can be added with QtPlot.add().

    For information on how x/y/z *args are handled see add() in the base
    plotting class.

    Args:
        *args: shortcut to provide the x/y/z data. See BasePlot.add

        figsize: (width, height) tuple in pixels to pass to GraphicsWindow
            default (1000, 600)

        figposition (dx, dy) tuple in pixels to pass to GraphicsWindow,
            distance from the upper left corner. Default None

        interval: period in seconds between update checks
            default 0.25

        theme: tuple of (foreground_color, background_color), where each is
            a valid Qt color. default (dark gray, white), opposite the
            pyqtgraph default of (white, black)

        **kwargs: passed along to QtPlot.add() to add the first data trace
    """

    def __init__(self, *args, figsize=(1000, 600), figposition=None,
                 interval=0.25, window_title=None, theme=((60, 60, 60), 'w'),
                 show_window=True, parent=None, **kwargs):
        QWidget.__init__(self, parent=parent)
        # Set base interval to None to disable that JS update-widget thingy
        BasePlot.__init__(self, interval=None)

        QShortcut(QtGui.QKeySequence("Ctrl+W"), self, self.close)
        QShortcut(QtGui.QKeySequence("Ctrl+Q"), self, self.close)

        self.traces = []
        self.subplots = []
        self.interval = interval
        self.auto_updating = False

        self.setWindowTitle(window_title or 'Plotwindow')
        if figposition:
            geometry_settings = itertools.chain(figposition,figsize)
            self.setGeometry(*geometry_settings)
        else:
            self.resize(*figsize)

        self.theme = theme
        self.area = dockarea.DockArea()

        layout = QHBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.area)
        self.setLayout(layout)

        if show_window:
            self.show()
        else:
            self.hide()

        self.add_subplot()

        if args or kwargs:
            self.add(*args, **kwargs)

        QtWidgets.QApplication.processEvents()

        if self.interval:
            self.auto_update()

    def closeEvent(self, event):
        """
        Make sure all dock-widgets are deleted upon closing or during garbage-
        collection. Otherwise references keep plots alive forever.
        """
        self.area.deleteLater()
        self.deleteLater()
        event.accept()

    def auto_update(self, interval=None):
        """
        Update the plot in a given interval.

        Args:
            interval (float): interval in seconds to wait before syncing the
            data, and updating the plot.
        """
        if (interval is not self.interval) and interval is not None:
            self.interval = interval

        if self.interval is None:
            self.auto_updating = False
            return

        self.auto_updating = True

        # update_data also calls self.update_plot()
        self.update_data()

        QtWidgets.QApplication.processEvents()

        if self.auto_updating:
            # We use the singleShot to avoid update queues in case the plotting
            # is slow
            QtCore.QTimer.singleShot(self.interval * 1000, self.auto_update)

    def halt(self):
        """
        Stop automatic updates to this plot, by disabling its update timer
        """
        self.auto_updating = False

    def clear(self):
        """
        Clear the plot window and remove all subplots and traces
        so that the window can be reused.
        """
        self.area.clear()
        self.traces = []
        self.subplots = []

    def add_subplot(self, title=None, position='right',
                    relativeto=None, **kwargs):
        """
        Add a new dock to the current window.

        Args:
            title (str):
                Title of the dock

            position (str):
                'bottom', 'top', 'left', 'right', 'above', or 'below'

            relativeto (DockWidget, int):
                If relativeto is None, then the new Dock is added to fill an
                entire edge of the window. If relativeto is another Dock, then
                the new Dock is placed adjacent to it (or in a tabbed
                configuration for 'above' and 'below').
        """

        title = '#{} - {}'.format(len(self.subplots) + 1, title or 'Plot')
        subplot_dock = Dock(name=title, autoOrientation=False, closable=True)

        if type(relativeto) is int:
            relativeto = self.subplots[relativeto - 1]
        self.area.addDock(subplot_dock, position, relativeto)

        subplot_widget = pg.GraphicsLayoutWidget()
        subplot_widget.setBackground(self.theme[1])

        hist_item = pg.HistogramLUTWidget()
        hist_item.item.vb.setMinimumWidth(10)
        hist_item.setMinimumWidth(120)
        hist_item.setBackground(self.theme[1])
        hist_item.axis.setPen(self.theme[0])
        hist_item.hide()

        subplot_dock.addWidget(subplot_widget, 0, 0)
        subplot_dock.addWidget(hist_item, 0, 1)

        plot_item = subplot_widget.addPlot()
        for _, ax in plot_item.axes.items():
            ax['item'].setPen(self.theme[0])

        subplot_dock.subplot_widget = subplot_widget
        subplot_dock.hist_item = hist_item
        subplot_dock.plot_item = plot_item

        self.subplots.append(subplot_dock)

        return subplot_dock

    def add_to_plot(self, subplot=1, **kwargs):
        """
        Add a dataset to a subplot. Create a new subplot if it does not exist.

        Args:
            subplot (int): The subplot the dataset is added to, indexing starts
            with 1. -1 represents the last existing index.
        """
        if subplot == -1:
            subplot = len(self.subplots) + 1
        if subplot > len(self.subplots):
            for i in range(subplot - len(self.subplots)):
                subplot_dock = self.add_subplot(**kwargs)

        subplot_object = self.subplots[subplot - 1]

        if 'title' in kwargs:
            title = kwargs['title']
        else:
            if 'z' in kwargs:
                title = self.get_default_array_title(kwargs['z'])
            elif 'y' in kwargs:
                title = self.get_default_array_title(kwargs['y'])
            elif 'x' in kwargs:
                title = self.get_default_array_title(kwargs['x'])

        subplot_object.setTitle('#{} - {}'.format(subplot, title or 'Plot'))

        if kwargs.get('clear', False):
            subplot_object.plot_item.clear()
            subplot_object.hist_item.hide()

        if 'z' in kwargs:
            plot_object, subplot_item = self._draw_image(
                subplot_object, **kwargs)

            self._update_cmap(plot_object)

        else:
            plot_object, subplot_item = self._draw_plot(
                subplot_object, **kwargs)

        transpose = kwargs.get('transpose', False)

        x = kwargs.get('x', None)
        y = kwargs.get('y', None)
        z = kwargs.get('z', None)
        if transpose:
            x, y = y, x

        kwargs['x'] = x
        kwargs['y'] = y

        transpose_z = False
        if hasattr(z, 'set_arrays'):
            if (x in z.set_arrays) and (y in z.set_arrays):
                if (z.set_arrays.index(x) == 1) and (z.set_arrays.index(y) == 0):
                    transpose_z = not transpose_z
            else:
                kwargs['interpolate'] = True
        kwargs['transpose_z'] = transpose_z

        self.traces.append({
            'config': kwargs,
            'plot_object': plot_object
        })

        self.update_plot()
        self.auto_update()

        self._update_labels(subplot_item, kwargs)
        return subplot_object

    def _draw_plot(self, subplot_object, y, x=None, color=None, width=None,
                   antialias=None, **kwargs):

        subplot_widget = subplot_object.subplot_widget
        plot_item = subplot_object.plot_item

        for side in ('left', 'bottom'):
            ax = plot_item.getAxis(side)
            ax.setPen(self.theme[0])
            # ax._qcodes_label = ''

        if 'pen' not in kwargs:
            if color is None:
                cycle = color_cycle
                color = cycle[len(self.traces) % len(cycle)]
            if width is None:
                width = 2
            kwargs['pen'] = pg.mkPen(color, width=width)

        if antialias is None:
            # looks a lot better antialiased, but slows down with many points
            # TODO: dynamically update this based on total # of points
            antialias = (len(y) < 1000)

        # If a marker symbol is desired use the same color as the line
        if any([('symbol' in key) for key in kwargs]):
            if 'symbolPen' not in kwargs:
                symbol_pen_width = 0.5 if antialias else 1.0
                kwargs['symbolPen'] = pg.mkPen('444', width=symbol_pen_width)
            if 'symbolBrush' not in kwargs:
                kwargs['symbolBrush'] = color

        pl = plot_item.plot(*self._line_data(x, y),
                            antialias=antialias, **kwargs)

        return pl, plot_item

    def _line_data(self, x, y):
        arrs = [arg for arg in [x, y] if arg is not None]
        for arr in arrs:
            finite = np.isfinite(arr)
            if not np.any(finite):
                return []
        if len(arrs) == 2:
            mask = np.logical_and(*[np.isfinite(arr) for arr in arrs])
        elif len(arrs) == 1:
            mask = np.isfinite(arrs[0])
        else:
            return None
        return [arr[mask] for arr in arrs]

    def _draw_image(self, subplot_object, z, cmap='hot', **kwargs):

        hist_item = subplot_object.hist_item
        plot_item = subplot_object.plot_item

        hist_item.show()

        # Item for displaying image data
        img = pg.ImageItem()
        hist_item.setImageItem(img)

        plot_item.addItem(img)
        if 'zlabel' in kwargs:  # used to specify a custom zlabel
            hist_item.axis.setLabel(kwargs['zlabel'])
        else:  # otherwise extracts the label from the dataarray
            hist_item.axis.setLabel(self.get_label(z),
                                    self.get_units(z))

        for side in ('left', 'bottom'):
            ax = plot_item.getAxis(side)
            ax.setPen(self.theme[0])

        plot_object = {
            'image': img,
            'hist': hist_item,
            'histlevels': hist_item.getLevels(),
            'cmap': cmap,
            'scales': {
                'x': TransformState(0, 1, True),
                'y': TransformState(0, 1, True)
            }
        }

        return plot_object, plot_item

    def _update_image(self, plot_object, config):
        z = config['z']

        img = plot_object['image']
        hist = plot_object['hist']
        scales = plot_object['scales']

        # make sure z is a *new* numpy float array (pyqtgraph barfs on ints),
        # and replace nan with minimum val bcs I can't figure out how to make
        # pyqtgraph handle nans - though the source does hint at a way:
        # http://www.pyqtgraph.org/documentation/_modules/pyqtgraph/widgets/ColorMapWidget.html
        # see class RangeColorMapItem
        #
        # For me (Merlin) nans work fine, its only the histogram that gets
        # messed up.

        z = np.asfarray(z)
        if config['transpose_z']:
            z = z.T

        finite = np.isfinite(z)
        if not np.any(finite):
            return

        maskX = np.any(finite, axis=1)
        maskY = np.any(finite, axis=0)

        minX, maxX = np.amin(np.where(maskX)), np.amax(np.where(maskX))
        minY, maxY = np.amin(np.where(maskY)), np.amax(np.where(maskY))

        z_range = (np.nanmin(z), np.nanmax(z))

        z = z[minX:maxX + 1, minY:maxY + 1]

        hist_range = hist.getLevels()
        if hist_range == plot_object['histlevels']:
            plot_object['histlevels'] = z_range
            hist.setLevels(*z_range)
            hist_range = z_range

        mask = {'y': [minY, maxY + 1],
                'x': [minX, maxX + 1]}

        # This is only needed for the histogram! otherwise there is no problem
        # with nans!
        z[~finite[minX:maxX + 1, minY:maxY + 1]] = z_range[0]

        img.setImage(z, levels=hist_range)
        scales_changed = False

        for axletter, axscale in scales.items():
            if axscale.revisit:
                axdata = config.get(axletter, None)
                newscale = self._get_transform(axdata)
                if (newscale.translate != axscale.translate or
                        newscale.scale != axscale.scale):
                    scales_changed = True
                scales[axletter] = newscale

        if scales_changed:
            img.resetTransform()
            img.translate(scales['x'].translate, scales['y'].translate)
            img.scale(scales['x'].scale, scales['y'].scale)

    def _update_cmap(self, plot_object):
        gradient = plot_object['hist'].gradient
        gradient.setColorMap(self._cmap(plot_object['cmap']))

    def set_cmap(self, cmap, traces=None):
        if isinstance(traces, int):
            traces = (traces,)
        elif traces is None:
            traces = range(len(self.traces))

        for i in traces:
            plot_object = self.traces[i]['plot_object']
            if not isinstance(plot_object, dict) or 'hist' not in plot_object:
                continue

            plot_object['cmap'] = cmap
            self._update_cmap(plot_object)

    def _get_transform(self, array):
        """
        pyqtgraph seems to only support uniform pixels in image plots.

        for a given setpoint array, extract the linear transform it implies
        if the setpoint data is *not* linear (or close to it), or if it's not
        uniform in any nested dimensions, issue a warning and return the
        default transform of 0, 1

        returns namedtuple TransformState(translate, scale, revisit)

        in pyqtgraph:
        translate means how many pixels to shift the image, away
            from the bottom or left edge being at zero on the axis
        scale means the data delta

        revisit is True if we just don't have enough info to scale yet,
        but we might later.
        """

        if array is None:
            return TransformState(0, 1, True)

        # do we have enough confidence in the setpoint data we've seen
        # so far that we don't have to repeat this as more data comes in?
        revisit = False

        # somewhat arbitrary - if the first 20% of the data or at least
        # 10 rows is uniform, assume it's uniform thereafter
        MINROWS = 10
        MINFRAC = 0.2

        # maximum setpoint deviation from linear to accept is 10% of a pixel
        MAXPX = 0.1

        if hasattr(array[0], '__len__'):
            # 2D array: check that all (non-empty) elements are congruent
            inner_len = max(len(row) for row in array)
            collapsed = np.array([np.nan] * inner_len)
            rows_before_trusted = max(MINROWS, len(array) * MINFRAC)
            for i, row in enumerate(array):
                for j, val in enumerate(row):
                    if np.isnan(val):
                        if i < rows_before_trusted:
                            revisit = True
                        continue
                    if np.isnan(collapsed[j]):
                        collapsed[j] = val
                    elif val != collapsed[j]:
                        warnings.warn(
                            'nonuniform nested setpoint array passed to '
                            'pyqtgraph. ignoring, using default scaling.')
                        return TransformState(0, 1, False)
        else:
            collapsed = array

        if np.isnan(collapsed).any():
            revisit = True

        indices_setpoints = list(zip(*((i, s) for i, s in enumerate(collapsed)
                                       if not np.isnan(s))))
        if not indices_setpoints:
            return TransformState(0, 1, revisit)

        indices, setpoints = indices_setpoints
        npts = len(indices)
        if npts == 1:
            indices = indices + (indices[0] + 1,)
            setpoints = setpoints + (setpoints[0] + 1,)

        i0 = indices[0]
        s0 = setpoints[0]
        total_di = indices[-1] - i0
        total_ds = setpoints[-1] - s0

        if total_ds == 0:
            warnings.warn('zero setpoint range passed to pyqtgraph. '
                          'ignoring, using default scaling.')
            return TransformState(0, 1, False)

        for i, s in zip(indices[1:-1], setpoints[1:-1]):
            icalc = i0 + (s - s0) * total_di / total_ds
            if np.abs(i - icalc) > MAXPX:
                warnings.warn('nonlinear setpoint array passed to pyqtgraph. '
                              'ignoring, using default scaling.')
                return TransformState(0, 1, False)

        scale = total_ds / total_di
        # extra 0.5 translation to get the first setpoint at the center of
        # the first pixel
        translate = s0 - (i0 + 0.5) * scale

        return TransformState(translate, scale, revisit)

    def _update_labels(self, subplot_object, config):
        """
        Updates x and y labels, by default tries to extract label from
        the DataArray objects located in the trace config. Custom labels
        can be specified the **kwargs "xlabel" and "ylabel"
        """

        for axletter, side in (('x', 'bottom'), ('y', 'left')):
            ax = subplot_object.getAxis(side)
            # pyqtgraph doesn't seem able to get labels, only set
            # so we'll store it in the axis object and hope the user
            # doesn't set it separately before adding all traces
            if axletter + 'label' in config and not ax.labelText:
                label = config[axletter + 'label']
                ax.setLabel(label)
            if axletter in config and not ax.labelText:
                label = self.get_label(config[axletter])
                units = self.get_units(config[axletter])
                ax.setLabel(label, units)

    def update(self):
        """
        Update the data in this plot, using the updaters given with
        MatPlot.add() or in the included DataSets, then include this in
        the plot.
        This is a wrapper routine that the update widget calls,
        inside this we call self.update() which should be subclassed
        """
        BasePlot.update(self)
        QWidget.update(self)

    def update_plot(self):
        for trace in self.traces:
            config = trace['config']
            plot_object = trace['plot_object']

            # TODO
            # Only update when data has changed ?!
            # update plot is called when any updater returned true, but there
            # is no reason to update all the other subplots, how can I
            # determine if the data from a trace has changed? I see no easy
            # link between the updater (i.e. the datset.sync()) and the trace
            # anymore.

            if 'z' in config:
                self._update_image(plot_object, config)
            elif 'x' in config and 'y' in config:
                plot_object.setData(*self._line_data(config['x'], config['y']))
            elif 'y' in config:
                plot_object.setData(*self._line_data(config['y']))
            elif 'x' in config:
                plot_object.setData(*self._line_data(config['x']))

    def _cmap(self, scale):
        if isinstance(scale, str):
            if scale in colorscales:
                values, colors = zip(*colorscales[scale])
            else:
                raise ValueError(scale + ' not found in colorscales')
        elif len(scale) == 2:
            values, colors = scale

        return pg.ColorMap(values, colors)

    def copy_to_clipboard(self):
        """
        Copy the current image to a the system clipboard
        """

        app = pg.mkQApp()
        clipboard = app.clipboard()
        clipboard.setPixmap(self.grab())

    def _repr_png_(self):
        """
        Create a png representation of the current window.
        """

        QtWidgets.QApplication.processEvents()

        image = self.grab(self.area.contentsRect())

        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.ReadWrite)
        image.save(buffer, 'PNG')
        buffer.close()
        return bytes(byte_array)

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
        image = self.win.grab()
        image.save(filename, "PNG", 0)

