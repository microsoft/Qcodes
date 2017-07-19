from os.path import abspath
from os.path import sep
from os import makedirs
import os
import logging
from copy import deepcopy
import numpy as np
from typing import Optional, Tuple

import qcodes as qc
from qcodes.loops import Loop
from qcodes.data.data_set import DataSet
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils.qcodes_device_annotator import DeviceImage

from IPython import get_ipython

log = logging.getLogger(__name__)
CURRENT_EXPERIMENT = {}
CURRENT_EXPERIMENT["logging_enabled"] = False


def init(mainfolder: str, sample_name: str, station, plot_x_position=0.66,
         annotate_image=True):
    """

    Args:
        mainfolder:  base location for the data
        sample_name:  name of the sample
        plot_x_position: fractional of screen position to put QT plots.
                         0 is all the way to the left and
                         1 is all the way to the right.

    """
    if sep in sample_name:
        raise TypeError("Use Relative names. That is without {}".format(sep))
    # always remove trailing sep in the main folder
    if mainfolder[-1] == sep:
        mainfolder = mainfolder[:-1]

    mainfolder = abspath(mainfolder)

    CURRENT_EXPERIMENT["mainfolder"] = mainfolder
    CURRENT_EXPERIMENT["sample_name"] = sample_name
    CURRENT_EXPERIMENT['init'] = True

    CURRENT_EXPERIMENT['plot_x_position'] = plot_x_position

    path_to_experiment_folder = sep.join([mainfolder, sample_name, ""])
    CURRENT_EXPERIMENT["exp_folder"] = path_to_experiment_folder

    try:
        makedirs(path_to_experiment_folder)
    except FileExistsError:
        pass

    log.info("experiment started at {}".format(path_to_experiment_folder))

    loc_provider = qc.FormatLocation(
        fmt=path_to_experiment_folder + '{counter}')
    qc.data.data_set.DataSet.location_provider = loc_provider
    CURRENT_EXPERIMENT["provider"] = loc_provider

    CURRENT_EXPERIMENT['station'] = station

    ipython = get_ipython()
    # turn on logging only if in ipython
    # else crash and burn
    if ipython is None:
        raise RuntimeWarning("History can't be saved. "
                             "-Refusing to proceed (use IPython/jupyter)")
    else:
        logfile = "{}{}".format(path_to_experiment_folder, "commands.log")
        CURRENT_EXPERIMENT['logfile'] = logfile
        if not CURRENT_EXPERIMENT["logging_enabled"]:
            log.debug("Logging commands to: t{}".format(logfile))
            ipython.magic("%logstart -t -o {} {}".format(logfile, "append"))
            CURRENT_EXPERIMENT["logging_enabled"] = True
        else:
            log.debug("Logging already started at {}".format(logfile))

    # Annotate image if wanted and necessary
    if annotate_image:
        _init_device_image(station)


def _init_device_image(station):

    di = DeviceImage(CURRENT_EXPERIMENT["exp_folder"], station)

    success = di.loadAnnotations()
    if not success:
        di.annotateImage()
    CURRENT_EXPERIMENT['device_image'] = di


def _select_plottables(tasks):
    """
    Helper function to select plottable tasks. Used inside the doNd functions.

    A task is here understood to be anything that the qc.Loop 'each' can eat.
    """
    # allow passing a single task
    if not hasattr(tasks, '__iter__'):
        tasks = (tasks,)

    # is the following check necessary AND sufficient?
    plottables = [task for task in tasks if hasattr(task, '_instrument')]

    return tuple(plottables)


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
                whatwhere.update({plot.subplots[j+k].getAxis(pos).labelText:
                                  pos})
            tdict = {'bottom': 'setXRange', 'left': 'setYRange'}
            # now find the data (not setpoint)
            checkstring = '{}_{}'.format(i._instrument.name, i.name)

            thedata = [data.arrays[d] for d in data.arrays.keys()
                       if d == checkstring][0]

            # Disable autoscale for the measured data
            if thedata.unit not in standardunits:
                subplot = plot.subplots[j+k]
                try:
                    # 1D measurement
                    ax = subplot.getAxis(whatwhere[thedata.label])
                    ax.enableAutoSIPrefix(False)
                except KeyError:
                    # 2D measurement
                    # Then we should fetch the colorbar
                    ax = plot.traces[j]['plot_object']['hist'].axis
                    ax.enableAutoSIPrefix(False)
                    ax.setLabel(text=thedata.label, unit=thedata.unit,
                                unitPrefix='')

            # Set up axis scaling
            for setarr in thedata.set_arrays:
                subplot = plot.subplots[j+k]
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

        else:
            if 'z' in inst_meta_data:
                xlen, ylen = inst_meta_data['z'].shape
                rasterized = xlen*ylen > 5000
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


def _save_individual_plots(data, inst_meas):

    def _create_plot(i, name, data, counter_two):
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
        plot.save("{}_{:03d}.pdf".format(plot.get_default_title(),
                                         counter_two))
        plot.fig.canvas.draw()

    counter_two = 0
    for j, i in enumerate(inst_meas):
        if getattr(i, "names", False):
            # deal with multidimensional parameter
            for k, name in enumerate(i.names):
                _create_plot(i, name, data, counter_two)
                counter_two += 1
        else:
            _create_plot(i, i.name, data, counter_two)
            counter_two += 1


def _flush_buffers(*params):
    """
    If possible, flush the VISA buffer of the instrument of the
    provided parameters. The params can be instruments as well.

    Supposed to be called inside doNd like so:
    _flush_buffers(inst_set, *inst_meas)
    """

    for param in params:
        if hasattr(param, '_instrument'):
            inst = param._instrument
            if hasattr(inst, 'visa_handle'):
                status_code = inst.visa_handle.clear()
                if status_code is not None:
                    log.warning("Cleared visa buffer on "
                                "{} with status code {}".format(inst.name,
                                                                status_code))
        elif isinstance(param, VisaInstrument):
            inst = param
            status_code = inst.visa_handle.clear()
            if status_code is not None:
                log.warning("Cleared visa buffer on "
                            "{} with status code {}".format(inst.name,
                                                            status_code))


def save_device_image(sweeptparameters):
    counter = CURRENT_EXPERIMENT['provider'].counter
    title = "{} #{:03d}".format(CURRENT_EXPERIMENT["sample_name"], counter)
    di = CURRENT_EXPERIMENT['device_image']
    di.updateValues(CURRENT_EXPERIMENT['station'], sweeptparameters)

    log.debug(os.path.join(CURRENT_EXPERIMENT["exp_folder"],
                           '{:03d}'.format(counter)))

    di.makePNG(CURRENT_EXPERIMENT["provider"].counter,
               os.path.join(CURRENT_EXPERIMENT["exp_folder"],
                            '{:03d}'.format(counter)), title)


def _do_measurement(loop: Loop, set_params: tuple, meas_params: tuple,
                    do_plots: Optional[bool]=True) -> Tuple[QtPlot, DataSet]:
    """
    The function to handle all the auxiliary magic of the T10 users, e.g.
    their plotting specifications, the device image annotation etc.
    The input loop should just be a loop ready to run and perform the desired
    measurement. The two params tuple are used for plotting.

    Args:
        loop: The QCoDeS loop object describing the actual measurement
        set_params: tuple of tuples. Each tuple is of the form
            (param, start, stop)
        meas_params: tuple of parameters to measure

    Returns:
        (plot, data)
    """
    parameters = [sp[0] for sp in set_params] + list(meas_params)
    _flush_buffers(*parameters)

    # startranges for _plot_setup
    startranges = dict(zip((sp[0].label for sp in set_params),
                           ((sp[1], sp[2]) for sp in set_params)))

    interrupted = False

    data = loop.get_data_set()

    if do_plots:
        plot, _ = _plot_setup(data, meas_params, startranges=startranges)
    else:
        plot = None
    try:
        if do_plots:
            _ = loop.with_bg_task(plot.update).run()
        else:
            _ = loop.run()
    except KeyboardInterrupt:
        interrupted = True
        print("Measurement Interrupted")
    if do_plots:
        # Ensure the correct scaling before saving
        for subplot in plot.subplots:
            vBox = subplot.getViewBox()
            vBox.enableAutoRange(vBox.XYAxes)
        cmap = None
        # resize histogram
        for trace in plot.traces:
            if 'plot_object' in trace.keys():
                print(type(trace['plot_object']))
                if (isinstance(trace['plot_object'], dict) and
                            'hist' in trace['plot_object'].keys()):
                    cmap = trace['plot_object']['cmap']
                    max = trace['config']['z'].max()
                    min = trace['config']['z'].min()
                    trace['plot_object']['hist'].setLevels(min, max)
                    trace['plot_object']['hist'].vb.autoRange()
        if cmap:
            plot.set_cmap(cmap)
        # set window back to original size
        plot.win.resize(1000, 600)
        plot.save()
        pdfplot, num_subplots = _plot_setup(data, meas_params, useQT=False)
        # pad a bit more to prevent overlap between
        # suptitle and title
        pdfplot.fig.tight_layout(pad=3)
        pdfplot.save("{}.pdf".format(plot.get_default_title()))
        pdfplot.fig.canvas.draw()
        if num_subplots > 1:
            _save_individual_plots(data, meas_params)
    if CURRENT_EXPERIMENT.get('device_image'):
        log.debug('Saving device image')
        save_device_image(tuple(sp[0] for sp in set_params))

    # add the measurement ID to the logfile
    with open(CURRENT_EXPERIMENT['logfile'], 'a') as fid:
        print("#[QCoDeS]# Saved dataset to: {}".format(data.location),
              file=fid)
    if interrupted:
        raise KeyboardInterrupt
    return plot, data


def do1d(inst_set, start, stop, num_points, delay, *inst_meas, do_plots=True):
    """

    Args:
        inst_set:  Instrument to sweep over
        start:  Start of sweep
        stop:  End of sweep
        num_points:  Number of steps to perform
        delay:  Delay at every step
        *inst_meas:  any number of instrument to measure and/or tasks to
          perform at each step of the sweep
        do_plots: Default True: If False no plots are produced.
            Data is still saved
             and can be displayed with show_num.

    Returns:
        plot, data : returns the plot and the dataset

    """

    loop = qc.Loop(inst_set.sweep(start,
                                  stop,
                                  num=num_points), delay).each(*inst_meas)

    set_params = (inst_set, start, stop),
    meas_params = _select_plottables(inst_meas)

    plot, data = _do_measurement(loop, set_params, meas_params,
                                 do_plots=do_plots)

    return plot, data


def do1dDiagonal(inst_set, inst2_set, start, stop, num_points,
                 delay, start2, slope, *inst_meas, do_plots=True):
    """
    Perform diagonal sweep in 1 dimension, given two instruments

    Args:
        inst_set:  Instrument to sweep over
        inst2_set: Second instrument to sweep over
        start:  Start of sweep
        stop:  End of sweep
        num_points:  Number of steps to perform
        delay:  Delay at every step
        start2:  Second start point
        slope:  slope of the diagonal cut
        *inst_meas:  any number of instrument to measure
        do_plots: Default True: If False no plots are produced.
            Data is still saved and can be displayed with show_num.

    Returns:
        plot, data : returns the plot and the dataset

    """

    # (WilliamHPNielsen) If I understand do1dDiagonal correctly, the inst2_set
    # is supposed to be varied secretly in the background
    set_params = ((inst_set, start, stop),)
    meas_params = _select_plottables(inst_meas)

    slope_task = qc.Task(inst2_set, (inst_set)*slope+(slope*start-start2))

    loop = qc.Loop(inst_set.sweep(start, stop, num=num_points),
                   delay).each(slope_task, *inst_meas, inst2_set)

    plot, data = _do_measurement(loop, set_params, meas_params,
                                 do_plots=do_plots)

    return plot, data


def do2d(inst_set, start, stop, num_points, delay,
         inst_set2, start2, stop2, num_points2, delay2,
         *inst_meas, do_plots=True):
    """

    Args:
        inst_set:  Instrument to sweep over
        start:  Start of sweep
        stop:  End of sweep
        num_points:  Number of steps to perform
        delay:  Delay at every step
        inst_set2:  Second instrument to sweep over
        start2:  Start of sweep for second instrument
        stop2:  End of sweep for second instrument
        num_points2:  Number of steps to perform
        delay2:  Delay at every step for second instrument
        *inst_meas:
        do_plots: Default True: If False no plots are produced.
            Data is still saved and can be displayed with show_num.

    Returns:
        plot, data : returns the plot and the dataset

    """

    for inst in inst_meas:
        if getattr(inst, "setpoints", False):
            raise ValueError("3d plotting is not supported")

    innerloop = qc.Loop(inst_set2.sweep(start2,
                                        stop2,
                                        num=num_points2),
                        delay2).each(*inst_meas)
    outerloop = qc.Loop(inst_set.sweep(start,
                                       stop,
                                       num=num_points),
                        delay).each(innerloop)

    set_params = ((inst_set, start, stop),
                  (inst_set2, start2, stop2))
    meas_params = _select_plottables(inst_meas)

    plot, data = _do_measurement(outerloop, set_params, meas_params,
                                 do_plots=do_plots)

    return plot, data


def show_num(id, useQT=False):
    """
    Show  and return plot and data for id in current instrument.
    Args:
        id(number): id of instrument

    Returns:
        plot, data : returns the plot and the dataset

    """
    if not getattr(CURRENT_EXPERIMENT, "init", True):
        raise RuntimeError("Experiment not initalized. "
                           "use qc.Init(mainfolder, samplename)")

    str_id = '{0:03d}'.format(id)

    t = qc.DataSet.location_provider.fmt.format(counter=str_id)
    data = qc.load_data(t)

    plots = []
    for value in data.arrays.keys():
        if "set" not in value:
            if useQT:
                plot = QtPlot(getattr(data, value),
                              fig_x_position=CURRENT_EXPERIMENT['plot_x_position'])
                title = "{} #{}".format(CURRENT_EXPERIMENT["sample_name"],
                                        str_id)
                plot.subplots[0].setTitle(title)
                plot.subplots[0].showGrid(True, True)
            else:
                plot = MatPlot(getattr(data, value))
                title = "{} #{}".format(CURRENT_EXPERIMENT["sample_name"],
                                        str_id)
                plot.subplots[0].set_title(title)
                plot.subplots[0].grid()
            plots.append(plot)
    return data, plots
