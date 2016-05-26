# code for example notebook

import math

from qcodes import MockInstrument, MockModel, Parameter, Loop, DataArray
from qcodes.utils.validators import Numbers


class AModel(MockModel):
    def __init__(self):
        self._gates = [0.0, 0.0, 0.0]
        self._excitation = 0.1
        super().__init__()

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

    def fmt(self, value):
        return '{:.3f}'.format(value)

    def gates_set(self, parameter, value):
        if parameter[0] == 'c':
            self._gates[int(parameter[1:])] = float(value)
        elif parameter == 'rst' and value is None:
            self._gates = [0.0, 0.0, 0.0]
        else:
            raise ValueError

    def gates_get(self, parameter):
        if parameter[0] == 'c':
            return self.fmt(self._gates[int(parameter[1:])])
        else:
            raise ValueError

    def source_set(self, parameter, value):
        if parameter == 'ampl':
            self._excitation = float(value)
        else:
            raise ValueError

    def source_get(self, parameter):
        if parameter == 'ampl':
            return self.fmt(self._excitation)
        else:
            raise ValueError

    def meter_get(self, parameter):
        if parameter == 'ampl':
            return self.fmt(self._output() * self._excitation)
        else:
            raise ValueError


# make our mock instruments
# real instruments would subclass IPInstrument or VisaInstrument
# or just the base Instrument instead of MockInstrument,
# and be instantiated with an address rather than a model
class MockGates(MockInstrument):
    def __init__(self, name, model=None, **kwargs):
        super().__init__(name, model=model, **kwargs)

        for i in range(3):
            cmdbase = 'c{}'.format(i)
            self.add_parameter('chan{}'.format(i),
                               label='Gate Channel {} (mV)'.format(i),
                               get_cmd=cmdbase + '?',
                               set_cmd=cmdbase + ':{:.4f}',
                               get_parser=float,
                               vals=Numbers(-100, 100))

        self.add_function('reset', call_cmd='rst')


class MockSource(MockInstrument):
    def __init__(self, name, model=None, **kwargs):
        super().__init__(name, model=model, **kwargs)

        # this parameter uses built-in sweeping to change slowly
        self.add_parameter('amplitude',
                           label='Source Amplitude (\u03bcV)',
                           get_cmd='ampl?',
                           set_cmd='ampl:{:.4f}',
                           get_parser=float,
                           vals=Numbers(0, 10),
                           step=0.1,
                           delay=0.05)


class MockMeter(MockInstrument):
    def __init__(self, name, model=None, **kwargs):
        super().__init__(name, model=model, **kwargs)

        self.add_parameter('amplitude',
                           label='Current (nA)',
                           get_cmd='ampl?',
                           get_parser=float)


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
        data = loop.run_temp()
        if self.measured_param._instrument.name is not None:
            inst = self.measured_param._instrument.name + '_'
        else:
            inst = ''
        return data.arrays[inst+self.measured_param.name].mean()


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
        data = loop.run_temp()
        if self.measured_param._instrument.name is not None:
            inst = self.measured_param._instrument.name + '_'
        else:
            inst = ''
        array = data.arrays[inst+self.measured_param.name]
        return (array, array.mean())
