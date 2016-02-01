# code for example notebook

import math

from qcodes import MockInstrument, Parameter, Loop, DataArray
from qcodes.utils.validators import Numbers


# here's a toy model - models mimic a communication channel,
# accepting and returning string data, so that they can
# mimic real instruments as closely as possible
#
# This could certainly be built in for simple subclassing
# if we use it a lot
class ModelError(Exception):
    pass


class AModel(object):
    def __init__(self):
        self._gates = [0.0, 0.0, 0.0]
        self._excitation = 0.1

    def _output(self):
        # my super exciting model!
        # make a nice pattern that looks sort of double-dotty
        # with the first two gates controlling the two dots,
        # and the third looking like Vsd
        delta_i = 10
        delta_j = 10
        di = (self._gates[0] + delta_i / 2) % delta_i - delta_i / 2
        dj = (self._gates[1] + delta_j / 2) % delta_j - delta_j / 2
        vsd = math.sqrt(self._gates[2]**2 + self._excitation**2)
        dij = math.sqrt(di**2 + dj**2) - vsd
        g = (vsd**2 + 1) * (1 / (dij**2 + 1) +
                            0.1 * (math.atan(-dij) + math.pi / 2))
        return g

    def write(self, instrument, parameter, value):
        if instrument == 'gates' and parameter[0] == 'c':
            self._gates[int(parameter[1:])] = float(value)
        elif instrument == 'gates' and parameter == 'rst':
            self._gates = [0.0, 0.0, 0.0]
        elif instrument == 'source' and parameter == 'ampl':
            self._excitation = float(value)
        else:
            raise ModelError('unrecognized write {}, {}, {}'.format(
                instrument, parameter, value))

    def ask(self, instrument, parameter):
        gates = self._gates

        if instrument == 'gates' and parameter[0] == 'c':
            v = gates[int(parameter[1:])]
        elif instrument == 'source' and parameter == 'ampl':
            v = self._excitation
        elif instrument == 'meter' and parameter == 'ampl':
            # here's my super complex model output!
            v = self._output() * self._excitation
        else:
            raise ModelError('unrecognized ask {}, {}'.format(
                instrument, parameter))

        return '{:.3f}'.format(v)


# make our mock instruments
# real instruments would subclass IPInstrument or VisaInstrument
# instead of MockInstrument,
# and be instantiated with an address rather than a model
class MockGates(MockInstrument):
    def __init__(self, name, model):
        super().__init__(name, model=model)

        for i in range(3):
            cmdbase = 'c{}'.format(i)
            self.add_parameter('chan{}'.format(i),
                               label='Gate Channel {} (mV)'.format(i),
                               get_cmd=cmdbase + '?',
                               set_cmd=cmdbase + ' {:.4f}',
                               parse_function=float,
                               vals=Numbers(-100, 100))

        self.add_function('reset', call_cmd='rst')


class MockSource(MockInstrument):
    def __init__(self, name, model):
        super().__init__(name, model=model)

        # this parameter uses built-in sweeping to change slowly
        self.add_parameter('amplitude',
                           label='Source Amplitude (\u03bcV)',
                           get_cmd='ampl?',
                           set_cmd='ampl {:.4f}',
                           parse_function=float,
                           vals=Numbers(0, 10),
                           sweep_step=0.1,
                           sweep_delay=0.05)


class MockMeter(MockInstrument):
    def __init__(self, name, model):
        super().__init__(name, model=model)

        self.add_parameter('amplitude',
                           label='Current (nA)',
                           get_cmd='ampl?',
                           parse_function=float)


class AverageGetter(Parameter):
    def __init__(self, measured_param, sweep_values, delay):
        super().__init__(name='avg_' + measured_param.name)
        self.measured_param = measured_param
        self.sweep_values = sweep_values
        self.delay = delay
        if hasattr(measured_param, 'label'):
            self.label = 'Average: ' + measured_param.label

    def get(self):
        loop = Loop(self.sweep_values, self.delay).each(self.measured_param)
        data = loop.run(background=False, data_manager=False, location=False,
                        quiet=True)
        return data.arrays[self.measured_param.name].mean()


class AverageAndRaw(Parameter):
    def __init__(self, measured_param, sweep_values, delay):
        name = measured_param.name
        super().__init__(names=(name, 'avg_' + name))
        self.measured_param = measured_param
        self.sweep_values = sweep_values
        self.delay = delay
        self.sizes = (len(sweep_values), None)
        set_array = DataArray(parameter=sweep_values.parameter,
                              preset_data=sweep_values)
        self.setpoints = (set_array, None)
        if hasattr(measured_param, 'label'):
            self.labels = (measured_param.label,
                           'Average: ' + measured_param.label)

    def get(self):
        loop = Loop(self.sweep_values, self.delay).each(self.measured_param)
        data = loop.run(background=False, data_manager=False, location=False,
                        quiet=True)
        array = data.arrays[self.measured_param.name]
        return (array, array.mean())
