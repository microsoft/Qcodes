from IPython import get_ipython
from os.path import sep
from os.path import abspath
from os import makedirs
import logging
import qcodes as qc

log = logging.getLogger(__name__)
CURRENT_EXPERIMENT = {}
CURRENT_EXPERIMENT["logging_enabled"] = False
CURRENT_EXPERIMENT["init"] = False
pdfdisplay = {}


def _set_up_exp_file(sample_name: str, mainfolder: str= None):
    """
    Makes
    Args:
        mainfolder:  base location for the data
        sample_name:  name of the sample
    """
    if sep in sample_name:
        raise TypeError("Use Relative names. That is without {}".format(sep))

    if mainfolder is None:
        try:
            mainfolder = qc.config.user.mainfolder
        except KeyError:
            raise KeyError('mainfolder not set in qc.config, see '
                           '"https://github.com/QCoDeS/Qcodes/blob/master'
                           '/docs/examples/Configuring_QCoDeS.ipynb"')

    # always remove trailing sep in the main folder
    if mainfolder[-1] == sep:
        mainfolder = mainfolder[:-1]
    mainfolder = abspath(mainfolder)

    CURRENT_EXPERIMENT["mainfolder"] = mainfolder
    CURRENT_EXPERIMENT["sample_name"] = sample_name
    CURRENT_EXPERIMENT['init'] = True
    path_to_experiment_folder = sep.join([mainfolder, sample_name, ""])
    CURRENT_EXPERIMENT["exp_folder"] = path_to_experiment_folder
    try:
        makedirs(path_to_experiment_folder)
    except FileExistsError:
        pass

    loc_provider = qc.FormatLocation(
        fmt=path_to_experiment_folder + '{counter}')
    qc.data.data_set.DataSet.location_provider = loc_provider
    CURRENT_EXPERIMENT["provider"] = loc_provider

    log.info("experiment started at {}".format(path_to_experiment_folder))


def _set_up_station(station):
    CURRENT_EXPERIMENT['station'] = station


def _set_up_subfolder(subfolder_name: str):
    mainfolder = CURRENT_EXPERIMENT["mainfolder"]
    sample_name = CURRENT_EXPERIMENT["sample_name"]
    CURRENT_EXPERIMENT[subfolder_name + '_subfolder'] = subfolder_name
    try:
        makedirs(sep.join([mainfolder, sample_name, subfolder_name]))
    except FileExistsError:
        pass
    log.info("{} subfolder set up".format(subfolder_name))


def _init_device_image(station):

    di = DeviceImage(CURRENT_EXPERIMENT["exp_folder"], station)

    success = di.loadAnnotations()
    if not success:
        di.annotateImage()
    CURRENT_EXPERIMENT['device_image'] = di


def _set_up_ipython_logging():
    ipython = get_ipython()
    # turn on logging only if in ipython
    # else crash and burn
    if ipython is None:
        raise RuntimeWarning("History can't be saved. "
                             "-Refusing to proceed (use IPython/jupyter)")
    else:
        exp_folder = CURRENT_EXPERIMENT["exp_folder"]
        logfile = "{}{}".format(exp_folder, "commands.log")
        CURRENT_EXPERIMENT['logfile'] = logfile
        if not CURRENT_EXPERIMENT["logging_enabled"]:
            log.debug("Logging commands to: t{}".format(logfile))
            ipython.magic("%logstart -t -o {} {}".format(logfile, "append"))
            CURRENT_EXPERIMENT["logging_enabled"] = True
        else:
            log.debug("Logging already started at {}".format(logfile))


def _set_up_pdf_preferences(subfolder_name: str = 'pdf', display_pdf=True,
                            display_individual_pdf=False):
    _set_up_subfolder(subfolder_name)
    pdfdisplay['individual'] = display_individual_pdf
    pdfdisplay['combined'] = display_pdf


########################################################################
# Actual init functions
########################################################################

def basic_init(sample_name: str, station, mainfolder: str= None):
    _set_up_exp_file(sample_name, mainfolder)
    _set_up_station(station)
    _set_up_ipython_logging()


def your_init(mainfolder: str, sample_name: str, station, plot_x_position=0.66,
              annotate_image=True, display_pdf=True,
              display_individual_pdf=False):
    basic_init(sample_name, station, mainfolder)
    _set_up_pdf_preferences(display_pdf=display_pdf,
                            display_individual_pdf=display_individual_pdf)
    CURRENT_EXPERIMENT['plot_x_position'] = plot_x_position
    if annotate_image:
        _init_device_image(station)


def my_init(sample_name: str, station, pdf_folder=True, analysis_folder=True,
            temp_dict_folder=True, waveforms_folder=True,
            annotate_image=False, mainfolder: str= None, display_pdf=True,
            display_individual_pdf=False, qubit_count=None):
    basic_init(sample_name, station, mainfolder)
    if pdf_folder:
        _set_up_pdf_preferences(display_pdf=display_pdf,
                                display_individual_pdf=display_individual_pdf)
    if analysis_folder:
        _set_up_subfolder('analysis')
    if temp_dict_folder:
        _set_up_subfolder('temp_dict')
    if waveforms_folder:
        _set_up_subfolder('waveforms')
    if annotate_image:
        _init_device_image(station)
    if qubit_count is not None:
        CURRENT_EXPERIMENT["qubit_count"] = qubit_count
