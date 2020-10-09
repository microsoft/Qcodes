"""
Live plotting using pyqtgraph
"""
from typing import Optional, Dict, Union, Deque, List, cast
import numpy as np
import pyqtgraph as pg
import pyqtgraph.multiprocess as pgmp

from pyqtgraph.multiprocess.remoteproxy import ClosedError, ObjectProxy
from pyqtgraph.graphicsItems.PlotItem.PlotItem import PlotItem
from pyqtgraph import QtGui

import qcodes.utils.helpers

import warnings
import logging
from collections import namedtuple, deque

from .base import BasePlot
from .colors import color_cycle, colorscales
import qcodes

TransformState = namedtuple('TransformState', 'translate scale revisit')

log = logging.getLogger(__name__)

class QtPlot(BasePlot):
    """
    Plot x/y lines or x/y/z heatmap data. The first trace may be included
    in the constructor, other traces can be added with QtPlot.add().

    For information on how ``x/y/z *args`` are handled see ``add()`` in the
     base plotting class.

    Args:
        *args: shortcut to provide the x/y/z data. See BasePlot.add

        figsize: (width, height) tuple in pixels to pass to GraphicsWindow
            default (1000, 600)
        interval: period in seconds between update checks
            default 0.25
        theme: tuple of (foreground_color, background_color), where each is
            a valid Qt color. default (dark gray, white), opposite the
            pyqtgraph default of (white, black)
        fig_x_pos: fraction of screen width to place the figure at
            0 is all the way to the left and
            1 is all the way to the right.
            default None let qt decide.
        fig_y_pos: fraction of screen width to place the figure at
            0 is all the way to the top and
            1 is all the way to the bottom.
            default None let qt decide.
        **kwargs: passed along to QtPlot.add() to add the first data trace
    """
    proc = None
    rpg = None
    # we store references to plots to keep the garbage collections from
    # destroying the windows. To keep memory consumption within bounds we
    # limit this to an arbitrary number of plots here using a deque
    # The issue is that even when closing a window it's difficult to
    # remove it from the list. This could potentially be done with a
    # close event on win but this is difficult with remote proxy process
    # as the list of plots lives in the main process and the plot locally
    # in a remote process
    max_len = qcodes.config['gui']['pyqtmaxplots']
    max_len = cast(int, max_len)
    plots: Deque['QtPlot'] = deque(maxlen=max_len)

    def __init__(self, *args, figsize=(1000, 600), interval=0.25,
                 window_title='', theme=((60, 60, 60), 'w'), show_window=True,
                 remote=True, fig_x_position=None, fig_y_position=None,
                 **kwargs):
        super().__init__(interval)

        if 'windowTitle' in kwargs.keys():
            warnings.warn("windowTitle argument has been changed to "
                          "window_title. Please update your call to QtPlot")
            temp_wt = kwargs.pop('windowTitle')
            if not window_title:
                window_title = temp_wt
        self.theme = theme

        if remote:
            if not self.__class__.proc:
                self._init_qt()
        else:
            # overrule the remote pyqtgraph class
            self.rpg = pg
            self.qc_helpers = qcodes.utils.helpers
        try:
            self.win = self.rpg.GraphicsWindow(title=window_title)
        except (ClosedError, ConnectionResetError) as err:
            # the remote process may have crashed. In that case try restarting
            # it
            if remote:
                log.warning("Remote plot responded with {} \n"
                            "Restarting remote plot".format(err))
                self._init_qt()
                self.win = self.rpg.GraphicsWindow(title=window_title)
            else:
                raise err
        self.win.setBackground(theme[1])
        self.win.resize(*figsize)
        self._orig_fig_size = figsize

        self.set_relative_window_position(fig_x_position, fig_y_position)
        self.subplots = [self.add_subplot()] # type: List[Union[PlotItem, ObjectProxy]]

        if args or kwargs:
            self.add(*args, **kwargs)

        if not show_window:
            self.win.hide()

        self.plots.append(self)

    def set_relative_window_position(self, fig_x_position, fig_y_position):
        if fig_x_position is not None or fig_y_position is not None:
            _, _, width, height = QtGui.QDesktopWidget().screenGeometry().getCoords()
            if fig_y_position is not None:
                y_pos = height * fig_y_position
            else:
                y_pos = self.win.y()
            if fig_x_position is not None:
                x_pos = width * fig_x_position
            else:
                x_pos = self.win.x()
            self.win.move(x_pos, y_pos)

    @classmethod
    def _init_qt(cls):
        # starting the process for the pyqtgraph plotting
        # You do not want a new process to be created every time you start a
        # run, so this only starts once and stores the process in the class
        pg.mkQApp()
        cls.proc = pgmp.QtProcess()  # pyqtgraph multiprocessing
        cls.rpg = cls.proc._import('pyqtgraph')
        cls.qc_helpers = cls.proc._import('qcodes.utils.helpers')

    def clear(self):
        """
        Clears the plot window and removes all subplots and traces
        so that the window can be reused.
        """
        self.win.clear()
        self.traces = []
        self.subplots: List[Union[PlotItem, ObjectProxy]] = []

    def add_subplot(self):
        subplot_object = self.win.addPlot()

        for side in ('left', 'bottom'):
            ax = subplot_object.getAxis(side)
            ax.setPen(self.theme[0])
            ax._qcodes_label = ''

        return subplot_object

    def add_to_plot(self, subplot=1, **kwargs):
        if subplot > len(self.subplots):
            for i in range(subplot - len(self.subplots)):
                self.subplots.append(self.add_subplot())
        subplot_object = self.subplots[subplot - 1]

        if 'name' in kwargs:
            if subplot_object.legend is None:
                subplot_object.addLegend(offset=(-30,30))

        if 'z' in kwargs:
            plot_object = self._draw_image(subplot_object, **kwargs)
        else:
            plot_object = self._draw_plot(subplot_object, **kwargs)

        self._update_labels(subplot_object, kwargs)
        prev_default_title = self.get_default_title()

        self.traces.append({
            'config': kwargs,
            'plot_object': plot_object
        })

        if prev_default_title == self.win.windowTitle():
            self.win.setWindowTitle(self.get_default_title())
        self.fixUnitScaling()

        return plot_object

    def _draw_plot(self, subplot_object, y, x=None, color=None, width=None,
                   antialias=None, **kwargs):
        if 'pen' not in kwargs:
            if color is None:
                cycle = color_cycle
                color = cycle[len(self.traces) % len(cycle)]
            if width is None:
                # there are currently very significant performance issues
                # with a penwidth larger than one
                width = 1
            kwargs['pen'] = self.rpg.mkPen(color, width=width)

        if antialias is None:
            # looks a lot better antialiased, but slows down with many points
            # TODO: dynamically update this based on total # of points
            antialias = (len(y) < 1000)

        # If a marker symbol is desired use the same color as the line
        if any([('symbol' in key) for key in kwargs]):
            if 'symbolPen' not in kwargs:
                symbol_pen_width = 0.5 if antialias else 1.0
                kwargs['symbolPen'] = self.rpg.mkPen('444',
                                                     width=symbol_pen_width)
            if 'symbolBrush' not in kwargs:
                kwargs['symbolBrush'] = color

        # suppress warnings when there are only NaN to plot
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN axis encountered')
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            pl = subplot_object.plot(*self._line_data(x, y),
                                     antialias=antialias, **kwargs)
        return pl

    def _line_data(self, x, y):
        return [self._clean_array(arg) for arg in [x, y] if arg is not None]

    def _draw_image(self, subplot_object, z, x=None, y=None, cmap=None,
                    zlabel=None,
                    zunit=None,
                    **kwargs):
        if cmap is None:
            cmap = qcodes.config['gui']['defaultcolormap']
        img = self.rpg.ImageItem()
        subplot_object.addItem(img)

        hist = self.rpg.HistogramLUTItem()
        hist.setImageItem(img)
        hist.axis.setPen(self.theme[0])

        if zunit is None:
            _, zunit = self.get_label(z)
        if zlabel is None:
            zlabel, _ = self.get_label(z)

        hist.axis.setLabel(zlabel, zunit)

        # TODO - ensure this goes next to the correct subplot?
        self.win.addItem(hist)

        plot_object = {
            'image': img,
            'hist': hist,
            'histlevels': hist.getLevels(),
            'cmap': cmap,
            'scales': {
                'x': TransformState(0, 1, True),
                'y': TransformState(0, 1, True)
            }
        }

        self._update_image(plot_object, {'x': x, 'y': y, 'z': z})
        self._update_cmap(plot_object)

        return plot_object

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
        z = np.asfarray(z).T
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            try:
                z_range = (np.nanmin(z), np.nanmax(z))
            except:
                # we get a warning here when z is entirely NaN
                # nothing to plot, so give up.
                return
        z[np.where(np.isnan(z))] = z_range[0]

        hist_range = hist.getLevels()
        if hist_range == plot_object['histlevels']:
            plot_object['histlevels'] = z_range
            hist.setLevels(*z_range)
            hist_range = z_range

        img.setImage(self._clean_array(z), levels=hist_range)

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
        can be specified the **kwargs "xlabel" and "ylabel". Custom units
        can be specified using the kwargs xunit, ylabel
        """
        for axletter, side in (('x', 'bottom'), ('y', 'left')):
            ax = subplot_object.getAxis(side)
            # danger: 🍝
            # find if any kwarg from plot.add in the base class
            # matches xlabel or ylabel, signaling a custom label
            if axletter+'label' in config and not ax._qcodes_label:
                label = config[axletter+'label']
            else:
                label = None

            # find if any kwarg from plot.add in the base class
            # matches xunit or yunit, signaling a custom unit
            if axletter+'unit' in config and not ax._qcodes_label:
                unit = config[axletter+'unit']
            else:
                unit = None

            #  find ( more hope to) unit and label from
            # the data array inside the config
            if axletter in config and not ax._qcodes_label:
                # now if we did not have any kwark gor label or unit
                # fallback to the data_array
                if unit is  None:
                    _, unit = self.get_label(config[axletter])
                if label is None:
                    label, _ = self.get_label(config[axletter])

            # pyqtgraph doesn't seem able to get labels, only set
            # so we'll store it in the axis object and hope the user
            # doesn't set it separately before adding all traces
            ax._qcodes_label = label
            ax._qcodes_unit = unit
            ax.setLabel(label, unit)

    def update_plot(self):
        for trace in self.traces:
            config = trace['config']
            plot_object = trace['plot_object']
            if 'z' in config:
                self._update_image(plot_object, config)
            else:
                plot_object.setData(*self._line_data(config['x'], config['y']))

    def _clean_array(self, array):
        """
        we can't send a DataArray to remote pyqtgraph for some reason,
        so send the plain numpy array
        """
        if hasattr(array, 'ndarray') and isinstance(array.ndarray, np.ndarray):
            return array.ndarray
        return array

    def _cmap(self, scale):
        if isinstance(scale, str):
            if scale in colorscales:
                values, colors = zip(*colorscales[scale])
            else:
                raise ValueError(scale + ' not found in colorscales')
        elif len(scale) == 2:
            values, colors = scale

        return self.rpg.ColorMap(values, colors)

    def _repr_png_(self):
        """
        Create a png representation of the current window.
        """
        image = self.win.grab()
        byte_array = self.rpg.QtCore.QByteArray()
        buffer = self.rpg.QtCore.QBuffer(byte_array)
        buffer.open(self.rpg.QtCore.QIODevice.ReadWrite)
        image.save(buffer, 'PNG')
        buffer.close()

        if hasattr(byte_array, '_getValue'):
            return bytes(byte_array._getValue())
        else:
            return bytes(byte_array)

    def save(self, filename=None):
        """
        Save current plot to filename, by default
        to the location corresponding to the default
        title.

        Args:
            filename (Optional[str]): Location of the file
        """
        default = f"{self.get_default_title()}.png"
        filename = filename or default
        image = self.win.grab()
        image.save(filename, "PNG", 0)

    def setGeometry(self, x, y, w, h):
        """ Set geometry of the plotting window """
        self.win.setGeometry(x, y, w, h)

    def autorange(self, reset_colorbar: bool=False) -> None:
        """
        Auto range all limits in case they were changed during interactive
        plot. Reset colormap if changed and resize window to original size.

        Args:
            reset_colorbar: Should the limits and colorscale of the colorbar
                be reset. Off by default
        """
        # seem to be a bug in mypy but the type of self.subplots cannot be
        # deducted even when typed above so ignore it and cast for now
        subplots = self.subplots
        for subplot in subplots:
            vBox = subplot.getViewBox()
            vBox.enableAutoRange(vBox.XYAxes)
        cmap = None
        # resize histogram
        for trace in self.traces:
            if 'plot_object' in trace.keys():
                if (isinstance(trace['plot_object'], dict) and
                            'hist' in trace['plot_object'].keys() and
                            reset_colorbar):
                    cmap = trace['plot_object']['cmap']
                    maxval = trace['config']['z'].max()
                    minval = trace['config']['z'].min()
                    trace['plot_object']['hist'].setLevels(minval, maxval)
                    trace['plot_object']['hist'].vb.autoRange()
        if cmap:
            self.set_cmap(cmap)
        # set window back to original size
        self.win.resize(*self._orig_fig_size)

    def fixUnitScaling(self, startranges: Optional[Dict[str, Dict[str, Union[float,int]]]]=None):
        """
        Disable SI rescaling if units are not standard units and limit
        ranges to data if known.

        Args:

            startranges: The plot can automatically infer the full ranges
                         array parameters. However it has no knowledge of the
                         ranges or regular parameters. You can explicitly pass
                         in the values here as a dict of the form
                         {'paramtername': {max: value, min:value}}
        """
        axismapping = {'x': 'bottom',
                       'y': 'left'}
        standardunits = self.standardunits
        # seem to be a bug in mypy but the type of self.subplots cannot be
        # deducted even when typed above so ignore it and cast for now
        subplots = self.subplots
        for i, plot in enumerate(subplots):
            # make a dict mapping axis labels to axis positions
            for axis in ('x', 'y', 'z'):
                if self.traces[i]['config'].get(axis) is not None:
                    unit = getattr(self.traces[i]['config'][axis], 'unit', None)
                    if unit is not None and unit not in standardunits:
                        if axis in ('x', 'y'):
                            ax = plot.getAxis(axismapping[axis])
                        else:
                            # 2D measurement
                            # Then we should fetch the colorbar
                            ax = self.traces[i]['plot_object']['hist'].axis
                        ax.enableAutoSIPrefix(False)
                        # because updateAutoSIPrefix called from
                        # enableAutoSIPrefix doesnt actually take the
                        # value of the argument into account we have
                        # to manually replicate the update here
                        ax.autoSIPrefixScale = 1.0
                        ax.setLabel(unitPrefix='')
                        ax.picture = None
                        ax.update()

                    # set limits either from dataset or
                    setarr = getattr(self.traces[i]['config'][axis], 'ndarray', None)
                    arrmin = None
                    arrmax = None
                    if setarr is not None and not np.all(np.isnan(setarr)):
                        arrmax = np.nanmax(setarr)
                        arrmin = np.nanmin(setarr)
                    elif startranges is not None:
                        try:
                            paramname = self.traces[i]['config'][axis].full_name
                            arrmax = startranges[paramname]['max']
                            arrmin = startranges[paramname]['min']
                        except (IndexError, KeyError, AttributeError):
                            continue

                    if axis == 'x':
                        rangesetter = getattr(plot.getViewBox(), 'setXRange')
                    elif axis == 'y':
                        rangesetter = getattr(plot.getViewBox(), 'setYRange')
                    else:
                        rangesetter = None

                    if (rangesetter is not None
                        and arrmin is not None
                        and arrmax is not None):
                        rangesetter(arrmin, arrmax)
