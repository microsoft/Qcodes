import warnings

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

warnings.warn("The Keysight SD_common module in qcodes is deprecated "
              "Please use the module from qcodes-contrib-drivers",
              category=QCoDeSDeprecationWarning)
