import logging
import logging.handlers

import os
from pathlib import Path

from typing import Union

# TODO: this import here is critical:
# when creating a new config obect this imported refrence will remain pointing
# at the old config object, while imports via 'import qcodes` paired with
# `qcodes.config....` will yield the new config values.
# Also in combination with the config context manger this will not work.
# importing all of qcodes here is not a good solution either as already the
# loading process shall be logged.
from qcodes import config

log = logging.getLogger(__name__)

logging_dir = "logs"
logging_delimiter = ' ¦ '
history_log_name = "command_history.log"
python_log_name = 'qcodes.log'
QCODES_USER_PATH_ENV='QCODES_USER_PATH'


def _get_qcodes_user_path() -> Union[str, None]:
    path = os.environ(QCODES_USER_PATH_ENV,
                      os.path.join(Path.home(), '.qcodes'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def start_logger() -> None:
    """
    Logging of messages passed throug the python logging module
    This sets up logging to a time based logging.
    This means that all logging messages on or above
    filelogginglevel will be written to pythonlog.log
    All logging messages on or above consolelogginglevel
    will be written to stderr.
    """
    format_string_items = ['%(asctime)s', '%(name)s', '%(levelname)s',
                           '%(funcName)s', '%(lineno)d', '%(message)s']
    format_string = logging_delimiter.join(format_string_items)
    formatter = logging.Formatter(format_string)
    try:
        filelogginglevel = config.core.file_loglevel
    except KeyError:
        filelogginglevel = "INFO"
    consolelogginglevel = config.core.loglevel
    ch = logging.StreamHandler()
    ch.setLevel(consolelogginglevel)
    ch.setFormatter(formatter)
    filename = os.path.join(_get_qcodes_user_path(),
                            logging_dir,
                            python_log_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh1 = logging.handlers.TimedRotatingFileHandler(filename,
                                                    when='midnight')
    fh1.setLevel(filelogginglevel)
    fh1.setFormatter(formatter)
    logging.basicConfig(handlers=[ch, fh1], level=logging.DEBUG)
    # capture any warnings from the warnings module
    logging.captureWarnings(capture=True)
    log.info("QCoDes python logger setup")


def start_command_history_logger() -> None:
    """
    logging of the history of the interactive command shell
    works only with ipython
    """
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is None:
        log.warn("Command history can't be saved outside of IPython/jupyter")
        return
    ipython.magic("%logstop")
    filename = os.path.join(_get_qcodes_user_path(),
                            logging_dir,
                            history_log_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ipython.magic("%logstart -t -o {} {}".format(filename, "append"))
    log.info("Started logging IPython history")


def start_all_logging() -> None:
    start_logger()
    start_command_history_logger()
