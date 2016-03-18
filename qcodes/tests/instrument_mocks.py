from qcodes.instrument.mock import MockInstrument, MockModel
from qcodes.instrument.server import ask_server
from qcodes.utils.validators import Numbers


class AMockModel(MockModel):
    def __init__(self):
        self._gates = [0.0, 0.0, 0.0]
        self._excitation = 0.1
        self._memory = {}
        super().__init__()

    def fmt(self, value):
        return '{:.3f}'.format(value)

    def gates_set(self, parameter, value):
        if parameter[0] == 'c':
            self._gates[int(parameter[1:])] = float(value)
        elif parameter == 'rst' and value is None:
            self._gates = [0.0, 0.0, 0.0]
        elif parameter[:3] == 'mem':
            slot = int(parameter[3:])
            self._memory[slot] = value
        else:
            raise ValueError

    def gates_get(self, parameter):
        if parameter[0] == 'c':
            return self.fmt(self._gates[int(parameter[1:])])
        elif parameter[:3] == 'mem':
            slot = int(parameter[3:])
            return self._memory[slot]
        else:
            raise ValueError

    def source_set(self, parameter, value):
        if parameter == 'ampl':
            try:
                self._excitation = float(value)
            except:
                # "Off" as in the MultiType sweep step test
                self._excitation = None
        else:
            raise ValueError(parameter, value)

    def source_get(self, parameter):
        if parameter == 'ampl':
            return self.fmt(self._excitation)
        # put mem here too, just so we can be 100% sure it's going through
        # the model
        elif parameter[:3] == 'mem':
            slot = int(parameter[3:])
            return self._memory[slot]
        else:
            raise ValueError

    def meter_get(self, parameter):
        if parameter == 'ampl':
            gates = self._gates
            # here's my super complex model output!
            return self.fmt(self._excitation *
                            (gates[0] + gates[1]**2 + gates[2]**3))
        elif parameter[:5] == 'echo ':
            return self.fmt(float(parameter[5:]))


class MockInstTester(MockInstrument):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_connect(self):
        super().on_connect()
        self.attach_adder()

    @ask_server
    def attach_adder(self):
        '''
        this function attaches a closure to the object, so can only be
        executed after creating the server because a closure is not
        picklable
        '''
        a = 5

        def f(b):
            '''
            not the same function as the original method
            '''
            return a + b
        self.add5 = f

    @ask_server
    def add5(self, b):
        '''
        The local copy of this should not get run, because it should
        be overwritten on the server by the closure version.
        This function itself should not be run, but we will see its docstring.
        '''
        raise RuntimeError('dont run this one!')


class MockGates(MockInstTester):
    def __init__(self, model, **kwargs):
        for i in range(3):
            cmdbase = 'c{}'.format(i)
            self.add_parameter('chan{}'.format(i), get_cmd=cmdbase + '?',
                               set_cmd=cmdbase + ':{:.4f}',
                               get_parser=float,
                               vals=Numbers(-10, 10))
            self.add_parameter('chan{}step'.format(i),
                               get_cmd=cmdbase + '?',
                               set_cmd=cmdbase + ':{:.4f}',
                               get_parser=float,
                               vals=Numbers(-10, 10),
                               sweep_step=0.1, sweep_delay=0.005)
        self.add_function('reset', call_cmd='rst')

        super().__init__('gates', model=model, delay=0.001,
                         use_async=True, **kwargs)


class MockSource(MockInstTester):
    def __init__(self, model, **kwargs):
        self.add_parameter('amplitude', get_cmd='ampl?',
                           set_cmd='ampl:{:.4f}', get_parser=float,
                           vals=Numbers(0, 1),
                           sweep_step=0.2, sweep_delay=0.005)

        super().__init__('source', model=model, delay=0.001, **kwargs)


class MockMeter(MockInstTester):
    def __init__(self, model, **kwargs):
        self.add_parameter('amplitude', get_cmd='ampl?', get_parser=float)
        self.add_function('echo', call_cmd='echo {:.2f}?',
                          args=[Numbers(0, 1000)], return_parser=float)

        super().__init__('meter', model=model, delay=0.001, **kwargs)
