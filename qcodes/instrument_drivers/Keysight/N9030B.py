from qcodes import VisaInstrument, Instrument
from qcodes.instrument.parameter import ArrayParameter
import numpy as np
from typing import Any


class N9030B(VisaInstrument):
    """
    Driver for Keysight N9030B PXA signal analyzer.
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter(name='gain',
                           get_cmd='READ:CORR:GAIN?',
                           parameter_class=Trace,
                           unit='dB',
                           label='Gain'
                           )

        self.add_parameter(name='phot',
                           get_cmd='READ:CORR:PHOT?',
                           parameter_class=Trace,
                           unit='dB',
                           label='P-HOT'
                           )

        self.add_parameter(name='pcold',
                           get_cmd='READ:CORR:PCOL?',
                           parameter_class=Trace,
                           unit='dB',
                           label='P-COLD'
                           )

        self.connect_message()


class Trace(ArrayParameter):

    def __init__(self, name: str, instrument: Instrument, get_cmd, unit: str,
                 label: str) -> None:
        self._cmd = get_cmd
        npts = instrument.npts()
        super().__init__(name, (npts,))
        self._instrument = instrument
        self.shape = (npts,)
        self.unit = unit
        self.label = label
        self.setpoint_units = ('Hz',)
        self.setpoint_names = ('Frequency',)

        self.set_sweep(self._instrument.fstart(),
                       self._instrument.fstop(), npts)

    def set_sweep(self, start: float, stop: float, npts: int) -> None:
        f = tuple(np.linspace(float(start), float(stop), num=npts))
        self.setpoints = (f,)
        self.shape = (npts,)

    def get(self) -> np.ndarray:
        array_data = np.array(self._instrument.ask(
            self._cmd).split(','), dtype='f')
        return array_data
