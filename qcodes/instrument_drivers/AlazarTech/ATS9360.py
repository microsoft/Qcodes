import numpy as np
from distutils.version import LooseVersion
import warnings

from .ATS import AlazarTech_ATS
from .utils import TraceParameter

from qcodes.utils import validators

class AlazarTech_ATS9360(AlazarTech_ATS):
    """
    This class is the driver for the ATS9360 board
    it inherits from the ATS base class

    TODO(nataliejpg):
        -  add clock source options and sample rate options
           (problem being that byte_to_value_dict of
           sample_rate relies on value of clock_source)

    """
    samples_divisor = 128
    _trigger_holdoff_min_fw_version = '21.07'

    def __init__(self, name, **kwargs):
        dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll'
        super().__init__(name, dll_path=dll_path, **kwargs)

        # add parameters

        # ----- Parameters for the configuration of the board -----
        self.add_parameter(name='clock_source',
                           parameter_class=TraceParameter,
                           get_cmd=None,
                           label='Clock Source',
                           unit=None,
                           initial_value='INTERNAL_CLOCK',
                           val_mapping={'INTERNAL_CLOCK': 1,
                                       'FAST_EXTERNAL_CLOCK': 2,
                                       'EXTERNAL_CLOCK_10MHz_REF': 7})
        self.add_parameter(name='external_sample_rate',
                           get_cmd=None,
                           parameter_class=TraceParameter,
                           label='External Sample Rate',
                           unit='S/s',
                           vals=validators.MultiType(validators.Ints(300000000, 1800000000),
                                                     validators.Enum('UNDEFINED')),
                           initial_value='UNDEFINED')
        self.add_parameter(name='sample_rate',
                           get_cmd=None,
                           parameter_class=TraceParameter,
                           label='Internal Sample Rate',
                           unit='S/s',
                           initial_value='UNDEFINED',
                           val_mapping={1_000: 1,
                                        2_000: 2,
                                        5_000: 4,
                                       10_000: 8,
                                       20_000: 10,