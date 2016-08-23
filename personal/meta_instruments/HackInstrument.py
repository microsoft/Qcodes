from functools import partial
from time import sleep

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils import validators as vals

class HackInstrument(Instrument):
    shared_kwargs = ['instruments']

    def __init__(self, name, instruments=[], **kwargs):
        super().__init__(name, **kwargs)
        self.add_parameter(name='set_T1_wait',
                           set_cmd=self.program_T1,
                           vals=vals.Numbers())

        self.add_parameter(name='measure_analyse',
                           # parameter_class=ManualParameter,
                           get_cmd=self._measure_analyse)

        self.instruments = instruments
        self.PulseBlaster = instruments['PulseBlaster']
        self.Arbstudio = instruments['Arbstudio']
        self.elr_analysis= instruments['analysis']

    def set_T1_wait(self, wait):
        marker_cycles = 100
        sampling_rate = 500
        durations = {'empty': 5, 'load': wait, 'read': 20, 'final_delay': 1}
        # Factor of 2 needed because apparently the core clock is not the same as the sampling rate
        ms = 2 * sampling_rate * 1e3

        self.PulseBlaster.detect_boards()
        self.PulseBlaster.select_board(0)
        self.PulseBlaster.core_clock(sampling_rate)

        self.PulseBlaster.start_programming()

        pulse = 'empty'
        self.PulseBlaster.send_instruction(0, 'continue', 0, marker_cycles)
        start = self.PulseBlaster.send_instruction(0, 'continue', 0, durations[pulse] * ms - marker_cycles)

        pulse = 'load'
        self.PulseBlaster.send_instruction(1, 'continue', 0, marker_cycles)
        self.PulseBlaster.send_instruction(0, 'continue', start, durations[pulse] * ms - marker_cycles)

        pulse = 'read'
        self.PulseBlaster.send_instruction(1, 'continue', 0, marker_cycles)
        self.PulseBlaster.send_instruction(0, 'continue', 0, durations[pulse] * ms - marker_cycles)

        pulse = 'final_delay'
        if durations[pulse] > 0:
            self.PulseBlaster.send_instruction(0, 'continue', 0, durations[pulse] * ms)
        self.PulseBlaster.send_instruction(1, 'branch', start, marker_cycles)

        self.PulseBlaster.stop_programming()

        self.ArbStudio.run([1,2,3])
        sleep(.1)
        self.PulseBlaster.start()

    def _measure_analyse(self):
        return self.analysis.measure_up_proportion()


class CustomParameter(Parameter):
    def __init__(self, name, instrument=None, **kwargs):
        print(kwargs)
        super().__init__(name=name, **kwargs)
        self._instrument = instrument
        # self._set_function = set_function
        self._meta_attrs.extend(['instrument'])

    # def set_function(self, function):
    #     self._set_function = function

    def set(self, val):
        return self._instrument.set_function(val)
        # if self._set_function is not None:
        #     self._value = val
        #     print(self._set_function(val))
        # else:
        #     raise NameError('No function defined')

    def get(self):
        return self._value