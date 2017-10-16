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

    Raises:
        NotImplementedError: If param is not a StandardParameter.
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

        # The original (raw/instrument) set/get functions
        param.raw_get = param._get
        param.raw_set = param._set
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


def removeFromSignalChain(param, element_number):
    """
    Remove an element from the signal chain of a parameter.

    Args:
        param (StandardParameter): A QCoDeS StandardParameter with a signal
            chain.
        element_number (int): The (1-indexed) number of the element to remove,
            counting from the instrument.
    """

    if not hasattr(param, 'signalchain'):
        raise ValueError('{} has no assigned signalchain'.format(param))

    if element_number > len(param.signalchain):
        raise ValueError('Can not remove element {} '.format(element_number) +
                         'from signal chain. Only contains '
                         '{} elements'.format(len(param.signalchain)))

    # remove the element by removing everything and then reapplying everything
    # BUT the specific element

    to_apply = param.signalchain.copy()
    to_apply.remove(to_apply[element_number-1])

    # reset the param to be "virgin"
    del param.signalchain
    param._meta_attrs.remove('signalchain')
    param._get = param.raw_get
    param._set = param.raw_set

    # and reapply the chain
    for (mapping, value) in to_apply:
        addToSignalChain(param, mapping, value)
