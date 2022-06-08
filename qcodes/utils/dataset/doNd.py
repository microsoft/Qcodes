import warnings

from qcodes.dataset.doNd import (
    AbstractSweep,
    ArraySweep,
    LinSweep,
    LogSweep,
    do0d,
    do1d,
    do2d,
    dond,
)
from qcodes.dataset.plotting import plot_and_save_image as plot

warnings.warn(
    "qcodes.utils.dataset.doNd module is deprecated. "
    "Please update to import from qcodes.dataset"
)

__all__ = [
    "do0d",
    "do1d",
    "do2d",
    "dond",
    "AbstractSweep",
    "ArraySweep",
    "LinSweep",
    "LogSweep",
    "plot",
]
