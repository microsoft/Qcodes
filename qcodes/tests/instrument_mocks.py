from qcodes.instrument.mock import MockInstrument, MockModel
from qcodes.instrument.server import ask_server


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
        the local copy of this should not get run, because it should
        be overwritten on the server by the closure version
        '''
        raise RuntimeError('dont run this one!')
