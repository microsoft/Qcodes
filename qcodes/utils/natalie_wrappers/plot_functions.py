import numpy as np
from os.path import sep
from copy import deepcopy
import functools
from matplotlib import ticker
import matplotlib.pyplot as plt

from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.utils.natalie_wrappers.file_setup import CURRENT_EXPERIMENT


def _rescale_mpl_axes(plot):
    def scale_formatter(i, pos, scale):
        return "{0:g}".format(i * scale)

    for i, subplot in enumerate(plot.subplots):
        for axis in 'x', 'y', 'z':
            if plot.traces[i]['config'].get(axis):
                unit = plot.traces[i]['config'][axis].unit
                label = plot.traces[i]['config'][axis].label
                maxval = abs(plot.traces[0]['config'][axis].ndarray).max()
                units_to_scale = ('V')
                if unit in units_to_scale:
                    if maxval < 1e-6:
                        scale = 1e9
                        new_unit = "n" + unit
                    elif maxval < 1e-3:
                        scale = 1e6
                        new_unit = "μ" + unit
                    elif maxval < 1:
                        scale = 1e3
                        new_unit = "m" + unit
                    else:
                        continue
                    tx = ticker.FuncFormatter(
                        functools.partial(scale_formatter, scale=scale))
                    new_label = "{} ({})".format(label, new_unit)
                    if axis in ('x', 'y'):
                        getattr(subplot, "{}axis".format(
                            axis)).set_major_formatter(tx)
                        getattr(subplot, "set_{}label".format(axis))(new_label)
                    else:
                        subplot.qcodes_colorbar.formatter = tx
                        subplot.qcodes_colorbar.ax.yaxis.set_major_formatter(
                            tx)
                        subplot.qcodes_colorbar.set_label(new_label)
                        subplot.qcodes_colorbar.update_ticks()


def _plot_setup(data, inst_meas, useQT=True, startranges=None):
    title = "{} #{:03d}".format(CURRENT_EXPERIMENT["sample_name"],
                                data.location_provider.counter)
    rasterized_note = " rasterized plot"
    num_subplots = 0
    counter_two = 0
    for j, i in enumerate(inst_meas):
        if getattr(i, "names", False):
            num_subplots += len(i.names)
        else:
            num_subplots += 1
    if useQT:
        plot = QtPlot(fig_x_position=CURRENT_EXPERIMENT['plot_x_position'])
    else:
        plot = MatPlot(subplots=(1, num_subplots))

    def _create_plot(plot, i, name, data, counter_two, j, k):
        """
        Args:
            plot: The plot object, either QtPlot() or MatPlot()
            i: The parameter to measure
            name: -
            data: The DataSet of the current measurement
            counter_two: The sub-measurement counter. Each measurement has a
                number and each sub-measurement has a counter.
            j: The current sub-measurement
            k: -
        """
        color = 'C' + str(counter_two)
        counter_two += 1
        inst_meas_name = "{}_{}".format(i._instrument.name, name)
        inst_meas_data = getattr(data, inst_meas_name)
        inst_meta_data = __get_plot_type(inst_meas_data, plot)
        if useQT:
            plot.add(inst_meas_data, subplot=j + k + 1)
            plot.subplots[j + k].showGrid(True, True)
            if j == 0:
                plot.subplots[0].setTitle(title)
            else:
                plot.subplots[j + k].setTitle("")

            # Avoid SI rescaling if units are not standard units
            standardunits = ['V', 's', 'J', 'W', 'm', 'eV', 'A', 'K', 'g',
                             'Hz', 'rad', 'T', 'H', 'F', 'Pa', 'C', 'Ω', 'Ohm',
                             'S']
            # make a dict mapping axis labels to axis positions
            # TODO: will the labels for two axes ever be identical?
            whatwhere = {}
            for pos in ('bottom', 'left', 'right'):
                whatwhere.update({plot.subplots[j + k].getAxis(pos).labelText:
                                  pos})
            tdict = {'bottom': 'setXRange', 'left': 'setYRange'}
            # now find the data (not setpoint)
            checkstring = '{}_{}'.format(i._instrument.name, name)

            thedata = [data.arrays[d] for d in data.arrays.keys()
                       if d == checkstring][0]

            # Disable autoscale for the measured data
            if thedata.unit not in standardunits:
                subplot = plot.subplots[j + k]
                try:
                    # 1D measurement
                    ax = subplot.getAxis(whatwhere[thedata.label])
                    ax.enableAutoSIPrefix(False)
                except KeyError:
                    # 2D measurement
                    # Then we should fetch the colorbar
                    ax = plot.traces[j + k]['plot_object']['hist'].axis
                    ax.enableAutoSIPrefix(False)
                    ax.setLabel(text=thedata.label, unit=thedata.unit,
                                unitPrefix='')

            # Set up axis scaling
            for setarr in thedata.set_arrays:
                subplot = plot.subplots[j + k]
                ax = subplot.getAxis(whatwhere[setarr.label])
                # check for autoscaling
                if setarr.unit not in standardunits:
                    ax.enableAutoSIPrefix(False)
                    # At this point, it has already been decided that
                    # the scale is milli whatever
                    # (the default empty plot is from -0.5 to 0.5)
                    # so we must undo that
                    ax.setScale(1e-3)
                    ax.setLabel(text=setarr.label, unit=setarr.unit,
                                unitPrefix='')
                # set the axis ranges
                if not(np.all(np.isnan(setarr))):
                    # In this case the setpoints are "baked" into the param
                    rangesetter = getattr(subplot.getViewBox(),
                                          tdict[whatwhere[setarr.label]])
                    rangesetter(setarr.min(), setarr.max())
                else:
                    # in this case someone must tell _create_plot what the
                    # range should be. We get it from startranges
                    rangesetter = getattr(subplot.getViewBox(),
                                          tdict[whatwhere[setarr.label]])
                    (rmin, rmax) = startranges[setarr.label]
                    rangesetter(rmin, rmax)
            QtPlot.qc_helpers.foreground_qt_window(plot.win)

        else:
            if 'z' in inst_meta_data:
                xlen, ylen = inst_meta_data['z'].shape
                rasterized = xlen * ylen > 5000
                plot.add(inst_meas_data, subplot=j + k + 1,
                         rasterized=rasterized)
            else:
                rasterized = False
                plot.add(inst_meas_data, subplot=j + k + 1, color=color)
                plot.subplots[j + k].grid()
            if j == 0:
                if rasterized:
                    fulltitle = title + rasterized_note
                else:
                    fulltitle = title
                plot.subplots[0].set_title(fulltitle)
            else:
                if rasterized:
                    fulltitle = rasterized_note
                else:
                    fulltitle = ""
                plot.subplots[j + k].set_title(fulltitle)

    for j, i in enumerate(inst_meas):
        if getattr(i, "names", False):
            # deal with multidimensional parameter
            for k, name in enumerate(i.names):
                _create_plot(plot, i, name, data, counter_two, j, k)
                counter_two += 1
        else:
            # simple_parameters
            _create_plot(plot, i, i.name, data, counter_two, j, 0)
            counter_two += 1
    return plot, num_subplots


def __get_plot_type(data, plot):
    # this is a hack because expand_trace works
    # in place. Also it should probably * expand its args and kwargs. N
    # Same below
    data_copy = deepcopy(data)
    metadata = {}
    plot.expand_trace((data_copy,), kwargs=metadata)
    return metadata


def _save_individual_plots(data, inst_meas, display_plot=True):

    def _create_plot(i, name, data, counter_two, display_plot=True):
        # Step the color on all subplots no just on plots
        # within the same axis/subplot
        # this is to match the qcodes-pyqtplot behaviour.
        title = "{} #{:03d}".format(CURRENT_EXPERIMENT["sample_name"],
                                    data.location_provider.counter)
        rasterized_note = " rasterized plot full data available in datafile"
        color = 'C' + str(counter_two)
        counter_two += 1
        plot = MatPlot()
        inst_meas_name = "{}_{}".format(i._instrument.name, name)
        inst_meas_data = getattr(data, inst_meas_name)
        inst_meta_data = __get_plot_type(inst_meas_data, plot)
        if 'z' in inst_meta_data:
            xlen, ylen = inst_meta_data['z'].shape
            rasterized = xlen * ylen > 5000
            plot.add(inst_meas_data, rasterized=rasterized)
        else:
            rasterized = False
            plot.add(inst_meas_data, color=color)
            plot.subplots[0].grid()
        if rasterized:
            plot.subplots[0].set_title(title + rasterized_note)
        else:
            plot.subplots[0].set_title(title)
        title_list = plot.get_default_title().split(sep)
        title_list.insert(-1, CURRENT_EXPERIMENT['pdf_subfolder'])
        title = sep.join(title_list)
        _rescale_mpl_axes(plot)
        plot.tight_layout()
        plot.save("{}_{:03d}.pdf".format(title,
                                         counter_two))
        if display_plot:
            plot.fig.canvas.draw()
            plt.show()
        else:
            plt.close(plot.fig)

    counter_two = 0
    for j, i in enumerate(inst_meas):
        if getattr(i, "names", False):
            # deal with multidimensional parameter
            for k, name in enumerate(i.names):
                _create_plot(i, name, data, counter_two, display_plot)
                counter_two += 1
        else:
            _create_plot(i, i.name, data, counter_two, display_plot)
            counter_two += 1
