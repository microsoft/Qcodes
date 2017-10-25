from qcodes import Parameter, MultiParameter, ArrayParameter
import numpy as np
import logging

class AcqVariablesParam(Parameter):
    """
    Parameter of an AcquisitionController which has a _check_and_update_instr
    function used for validation and to update instrument attributes and a
    _get_default function which it uses to set the AcqVariablesParam to an
    instrument calculated default.

    Args:
        name: name for this parameter
        instrument: acquisition controller instrument this parameter belongs to
        check_and_update_fn: instrument function to be used for value
            validation and updating instrument values
        default_fn (optional): instrument function to be used to calculate
            a default value to set parameter to
        initial_value (optional): initial value for parameter
    """

    def __init__(self, name, instrument, check_and_update_fn,
                 default_fn=None, initial_value=None):
        super().__init__(name)
        self._instrument = instrument
        self._save_val(initial_value)
        self._check_and_update_instr = check_and_update_fn
        if default_fn is not None:
            self._get_default = default_fn

    def set_raw(self, value):
        """
        Function which checks value using validation function and then sets
        the Parameter value to this value.

        Args:
            value: value to set the parameter to
        """
        self._check_and_update_instr(value, param_name=self.name)
        self._save_val(value)

    def get_raw(self):
        return self._latest['value']

    def to_default(self):
        """
        Function which executes the default_fn specified to calculate the
        default value based on instrument values and then calls the set
        function with this value
        """
        try:
            default = self._get_default()
        except AttributeError as e:
            raise AttributeError('no default function for {} Parameter '
                                 '{}'.format(self.name, e))
        self.set(default)

    def check(self):
        """
        Function which checks the current Parameter value using the specified
        check_and_update_fn which can also serve to update instrument values.

        Return:
            True (if no errors raised when check_and_update_fn executed)
        """
        val = self._latest['value']
        self._check_and_update_instr(val, param_name=self.name)
        return True


class NonSettableDerivedParameter(Parameter):
    """
    Parameter of an AcquisitionController which cannot be updated directly
    as it's value is derived from other parameters. This is intended to be
    used in high level APIs where Alazar parameters such as 'samples_per_record'
    are not set directly but are parameters of the actual instrument anyway.

    This assumes that the parameter is stored via a call to '_save_val' by
    any set of parameter that this parameter depends on.

    Args:
        name: name for this parameter
        instrument: acquisition controller instrument this parameter belongs to
        alternative (str): name of parameter(s) that controls the value of this
            parameter and can be set directly.
    """

    def __init__(self, name, instrument, alternative: str, **kwargs):
        self._alternative = alternative
        super().__init__(name, instrument=instrument, **kwargs)

    def set_raw(self, value):
        """
        It's not possible to directly set this parameter as it's derived from other
        parameters.
        """
        raise NotImplementedError("Cannot directly set {}. To control this parameter"
                                  "set {}".format(self.name, self._alternative))

    def get_raw(self):
        return self.get_latest()


class EffectiveSampleRateParameter(NonSettableDerivedParameter):


    def get_raw(self):
        """
        Obtain the effective sampling rate of the acquisition
        based on clock type, clock speed and decimation

        Returns:
            the number of samples (per channel) per second
        """
        if self._instrument.clock_source.get() == 'EXTERNAL_CLOCK_10MHz_REF':
            rate = self._instrument.external_sample_rate.get()
        elif self._instrument.clock_source.get() == 'INTERNAL_CLOCK':
            rate = self._instrument.sample_rate.get()
        else:
            raise Exception("Don't know how to get sample rate with"
                            " {}".format(self._instrument.clock_source.get()))

        if rate == '1GHz_REFERENCE_CLOCK':
            rate = 1e9

        decimation = self._instrument.decimation.get()
        if decimation > 0:
            rate = rate / decimation

        self._save_val(rate)
        return rate
