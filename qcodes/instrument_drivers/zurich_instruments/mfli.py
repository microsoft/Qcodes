# Zurich Instrument MFLI stub
#
# For the real implementation, please look into the package zhinst-qcodes
# at https://github.com/zhinst/zhinst-qcodes


try:
    from zhinst.qcodes.mfli import *
except ImportError:
    raise ImportError('''Could not find Zurich Instruments QCodes drivers.
                         Please install package zhinst-qcodes.
                      ''')
