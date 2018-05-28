import warnings

from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators

class TraceParameter(Parameter):
    """
    A parameter that keeps track of if its value has been synced to
    the Instrument by setting a flag when changed.
    """
    def __init__(self, *args, **kwargs):
        self._synced_to_card = False
        super().__init__(*args, **kwargs)

    def _set_updated(self):
        self._synced_to_card = True

    @property
    def synced_to_card(self) -> bool:
        return self._synced_to_card

    def set_raw(self, value):
        self._instrument._parameters_synced = False
        self._synced_to_card = False
        self._save_val(value, validate=False)

class TrivialDictionary:
    """
    This class looks like a dictionary to the outside world
    every key maps to this key as a value (lambda x: x)
    """
    def __init__(self):
        warnings.warn("TrivialDictionary is deprecated and will be removed")
        pass

    def __getitem__(self, item):
        return item

    def __contains__(self, item):
        # this makes sure that this dictionary contains everything
        return True


class AlazarParameter(Parameter):
    """
    This class it deprected because it provides little to no value and over
    regular parameters and is implemented in a non standard compliant way

    This class represents of many parameters that are relevant for the Alazar
    driver. This parameters only have a private set method, because the values
    are set by the Alazar driver. They do have a get function which return a
    human readable value. Internally the value is stored as an Alazar readable
    value.

    These parameters also keep track the up-to-dateness of the value of this
    parameter. If the private set_function is called incorrectly, this
    parameter raises an error when the get_function is called to warn the user
    that the value is out-of-date

    Args:
        name: see Parameter class
        label: see Parameter class
        unit: see Parameter class
        instrument: see Parameter class
        value: default value
        byte_to_value_dict: dictionary that maps byte values (readable to the
            alazar) to values that are readable to humans
        vals: see Parameter class, should not be set if byte_to_value_dict is
            provided
    """
    def __init__(self, name=None, label=None, unit=None, instrument=None,
                 value=None, byte_to_value_dict=None, vals=None):
        warnings.warn("AlazarParamater is deprecated. Please replace with "
                      "Regular parameter or TraceParameter")
        if vals is None:
            if byte_to_value_dict is None:
                vals = validators.Anything()
            else:
                # TODO(damazter) (S) test this validator
                vals = validators.Enum(*byte_to_value_dict.values())

        super().__init__(name=name, label=label, unit=unit, vals=vals,
                         instrument=instrument)
        self.instrument = instrument
        self._byte = None
        self._uptodate_flag = False

        # TODO(damazter) (M) check this block
        if byte_to_value_dict is None:
            self._byte_to_value_dict = TrivialDictionary()
            self._value_to_byte_dict = TrivialDictionary()
        else:
            self._byte_to_value_dict = byte_to_value_dict
            self._value_to_byte_dict = {
                v: k for k, v in self._byte_to_value_dict.items()}

        self._set(value)

    def get_raw(self):
        """
        This method returns the name of the value set for this parameter

        Returns:
            value

        """
        # TODO(damazter) (S) test this exception
        if self._uptodate_flag is False:
            raise Exception('The value of this parameter (' + self.name +
                            ') is not up to date with the actual value in '
                            'the instrument.\n'
                            'Most probable cause is illegal usage of ._set() '
                            'method of this parameter.\n'
                            'Don\'t use private methods if you do not know '
                            'what you are doing!')
        return self._byte_to_value_dict[self._byte]

    def _get_byte(self):
        """
        this method gets the byte representation of the value of the parameter

        Returns:
            byte representation

        """
        return self._byte

    def _set(self, value):
        """
        This method sets the value of this parameter
        This method is private to ensure that all values in the instruments
        are up to date always

        Args:
            value: the new value (e.g. 'NPT', 0.5, ...)
        Returns:
            None

        """

        # TODO(damazter) (S) test this validation
        self.validate(value)
        self._byte = self._value_to_byte_dict[value]
        self._uptodate_flag = False
        self._save_val(value)
        return None

    def _set_updated(self):
        """
        This method is used to keep track of which parameters are updated in the
        instrument. If the end-user starts messing with this function, things
        can go wrong.

        Do not use this function if you do not know what you are doing

        Returns:
            None
        """
        self._uptodate_flag = True
