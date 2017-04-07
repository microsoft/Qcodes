# A minimal implementation of the signal chain idea.
# It is up to the user to ensure that the signal chain is applied to
# parameters for which it makes sense to apply it.

from qcodes.instrument.parameter import StandardParameter


def addToSignalChain(param, mapping, value):
    """
    Add a mapping to the signalchain of a parameter.

    Args:
        param (StandardParameter): A QCoDeS StandardParameter. The signal
            chain will probably not make sense for other things than voltages.
        mapping (str): A string describing the mapping. Must be either
            'offset' or 'multiplier'.
        value (Union[float, int]): The numerical parameter of the mapping.
    """

    if not isinstance(param, StandardParameter):
        raise NotImplementedError('Signal Chain only supports '
                                  'StandardParameters. Recieved an instance '
                                  'of {}.'.format(param.__class__))

    maps = {'offset': {'get': lambda x, a: x + a,
                       'set': lambda x, a: x - a},
            'multiplier': {'get': lambda x, a: x * a,
                           'set': lambda x, a: x / a}}

    # set up a signal chain "inside" the parameter
    if not hasattr(param, 'signalchain'):
        param._meta_attrs.append('signalchain')
        param.signalchain = [(mapping, value)]
        # TODO: save original/raw set/get
    else:
        param.signalchain.append((mapping, value))

    # Daisy-chain gets and sets

    old_get = param._get
    old_set = param._set

    def new_get():
        inst_val = old_get()
        new_val = maps[mapping]['get'](inst_val, value)
        param._save_val(new_val)  # will overwrite previous links in the chain
        return new_val

    def new_set(set_val):
        new_val = maps[mapping]['set'](set_val, value)
        param._save_val(new_val)  # will overwrite previous links in the chain
        old_set(new_val)  # pipes value back through chain

    param._get = new_get
    param._set = new_set
