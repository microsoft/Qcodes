from matplotlib import ticker
from os.path import sep

import qcodes as qc
from qcodes.utils.natalie_wrappers.file_setup import CURRENT_EXPERIMENT
from qcodes.utils.natalie_wrappers.plot_functions import _plot_setup, \
    _rescale_mpl_axes, _save_individual_plots
from qcodes.utils.natalie_wrappers.device_image_attempt import save_device_image


import logging
log = logging.getLogger(__name__)


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
        if CURRENT_EXPERIMENT.get('pdf_subfolder'):
            plt.ioff()
            pdfplot, num_subplots = _plot_setup(data, meas_params, useQT=False)
            # pad a bit more to prevent overlap between
            # suptitle and title
            _rescale_mpl_axes(pdfplot)
            pdfplot.fig.tight_layout(pad=3)
            title_list = plot.get_default_title().split(sep)
            title_list.insert(-1, CURRENT_EXPERIMENT['pdf_subfolder'])
            title = sep.join(title_list)

            pdfplot.save("{}.pdf".format(title))
            if (pdfdisplay['combined'] or
                    (num_subplots == 1 and pdfdisplay['individual'])):
                pdfplot.fig.canvas.draw()
                plt.show()
            else:
                plt.close(pdfplot.fig)
            if num_subplots > 1:
                _save_individual_plots(
                    data, meas_params, pdfdisplay['individual'])
            plt.ion()
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


def do1d(inst_set, start, stop, num_points, delay, *meas_params,
         do_plots=True):
    """

    Args:
        inst_set:  Instrument to sweep over
        start:  Start of sweep
        stop:  End of sweep
        num_points:  Number of steps to perform
        delay:  Delay at every step
        *meas_params:  any number of parameters to measure and/or tasks to
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
    meas_params = _select_plottables(meas_params)

    plot, data = _do_measurement(loop, set_params, meas_params,
                                 do_plots=do_plots)

    return plot, data


def do2d(inst_set, start, stop, num_points, delay,
         inst_set2, start2, stop2, num_points2, delay2,
         *meas_params, do_plots=True):
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
        *meas_params:
        do_plots: Default True: If False no plots are produced.
            Data is still saved and can be displayed with show_num.

    Returns:
        plot, data : returns the plot and the dataset

    """

    for inst in meas_params:
        if getattr(inst, "setpoints", False):
            raise ValueError("3d plotting is not supported")

    innerloop = qc.Loop(inst_set2.sweep(start2,
                                        stop2,
                                        num=num_points2),
                        delay2).each(*meas_params)
    outerloop = qc.Loop(inst_set.sweep(start,
                                       stop,
                                       num=num_points),
                        delay).each(innerloop)

    set_params = ((inst_set, start, stop),
                  (inst_set2, start2, stop2))
    meas_params = _select_plottables(meas_params)

    plot, data = _do_measurement(outerloop, set_params, meas_params,
                                 do_plots=do_plots)
