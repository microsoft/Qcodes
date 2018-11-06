# PCI-6221.py
from .daq import NI_DAQ, DAQReadAnalog, DAQWriteAnalog
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import ManualParameter
from functools import partial

class NI_PCI6221(NI_DAQ):
    '''
    This class is the driver for the National Instruments PCI 6221 data
    acquisition card.
    '''
    def __init__(self, name, dev_id, **kwargs):
        super().__init__(name, dev_id, **kwargs)

        for ai in self.ai_channels:
            self.add_parameter(name=ai + '_sample_rate',
                               parameter_class=ManualParameter,
                               label='Sample rate for ' + ai,
                               unit='S/s',
                               initial_value=250e3,
                               vals=Numbers(
                                max_value=self.get_maximum_input_channel_rate()),
                                )
            self.add_parameter(name=ai + '_samples',
                               parameter_class=ManualParameter,
                               label='Number of samples for ' + ai,
                               unit='S',
                               initial_value=1,
                                )
            self.add_parameter(name=ai + '_timeout',
                               parameter_class=ManualParameter,
                               label='trigger time out ' + ai,
                               unit='s',
                               initial_value=10.0,
                                )
            self.add_parameter(name=ai + '_voltage_range',
                               parameter_class=ManualParameter,
                               label='Voltage range for ' + ai,
                               unit='V',
                               initial_value=[-10.0, 10.0],
                                )
            self.add_parameter(name=ai,
                               parameter_class=DAQReadAnalog,
                               label='Analog Input ' + ai,
                               unit='V',
                               )


        for ao in self.ao_channels:
            self.add_parameter(name=ao,
                               parameter_class=DAQWriteAnalog,
                               label='Analog Output ' + ao,
                               unit='V',
                               )
