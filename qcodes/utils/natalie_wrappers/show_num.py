import qcodes as qc
from qcodes.utils.natalie_wrappers.file_setup import CURRENT_EXPERIMENT
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot


def show_num(id, useQT=False, do_plot=True):
    """
    Show  and return plot and data for id in current instrument.
    Args:
        id(number): id of instrument
    Returns:
        plot, data : returns the plot and the dataset
    """
    if not CURRENT_EXPERIMENT["init"]:
        raise RuntimeError("Experiment not initalized. "
                           "use qc.Init(mainfolder, samplename)")

    str_id = '{0:03d}'.format(id)

    t = qc.DataSet.location_provider.fmt.format(counter=str_id)
    data = qc.load_data(t)

    if do_plot:
        plots = []
        for value in data.arrays.keys():
            if "set" not in value:
                if useQT:
                    plot = QtPlot(
                        getattr(data, value),
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
    else:
        plots = None
    return data, plots
