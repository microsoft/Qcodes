from time import sleep

import qcodes as qc
from qcodes.instrument.parameter import Parameter, ManualParameter

class PulseParameter(Parameter):
    def __init__(self, name, PulseBlaster, ArbStudio, instrument=None, **kwargs):
        super().__init__(name, **kwargs)
        self._instrument = instrument

        self.PulseBlaster = PulseBlaster
        self.ArbStudio = ArbStudio

        self.sweep_parameter = None # Parameter in durations to sweep
        self.durations = None # Dictionary of durations
        self.marker_cycles = 100
        self.channels = [1,2,3]

    def set(self, val):
        assert self.sweep_parameter is not None, "Must set sweep_parameter"
        assert self.durations is not None, "Must set durations"
        assert self.sweep_parameter in self.durations.keys(), "Sweep parameter must be in durations"

        durations = self.durations.copy()
        durations[self.sweep_parameter] = val

        self.ArbStudio.stop()
        self.PulseBlaster.stop()

        sampling_rate = 500
        # Factor of 2 needed because apparently the core clock is not the same as the sampling rate
        ms = 2 * sampling_rate * 1e3

        self.PulseBlaster.detect_boards()
        self.PulseBlaster.select_board(0)
        self.PulseBlaster.core_clock(sampling_rate)

        self.PulseBlaster.start_programming()

        pulse = 'empty'
        self.PulseBlaster.send_instruction(0, 'continue', 0, self.marker_cycles)
        start = self.PulseBlaster.send_instruction(0, 'continue', 0, durations[pulse] * ms - self.marker_cycles)

        pulse = 'load'
        self.PulseBlaster.send_instruction(1, 'continue', 0, self.marker_cycles)
        self.PulseBlaster.send_instruction(0, 'continue', start, durations[pulse] * ms - self.marker_cycles)

        pulse = 'read'
        self.PulseBlaster.send_instruction(1, 'continue', 0, self.marker_cycles)
        self.PulseBlaster.send_instruction(0, 'continue', 0, durations[pulse] * ms - self.marker_cycles)

        pulse = 'final_delay'
        if durations[pulse] > 0:
            self.PulseBlaster.send_instruction(0, 'continue', 0, durations[pulse] * ms)
        self.PulseBlaster.send_instruction(1, 'branch', start, self.marker_cycles)

        self.PulseBlaster.stop_programming()

        self.ArbStudio.run(channels=self.channels)
        sleep(0.1)
        self.PulseBlaster.start()