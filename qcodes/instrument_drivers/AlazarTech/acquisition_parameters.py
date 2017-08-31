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

    def set(self, value):
        """
        Function which checks value using validation function and then sets
        the Parameter value to this value.

        Args:
            value: value to set the parameter to
        """
        self._check_and_update_instr(value, param_name=self.name)
        self._save_val(value)

    def get(self):
        return self._latest()['value']

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
        val = self._latest()['value']
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

    def set(self, value):
        """
        It's not possible to directly set this parameter as it's derived from other
        parameters.
        """
        raise NotImplementedError("Cannot directly set {}. To control this parameter"
                                  "set {}".format(self.name, self._alternative))

    def get(self):
        return self.get_latest()


class EffectiveSampleRateParameter(NonSettableDerivedParameter):


    def get(self):
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
            raise Exception("Don't know how to get sample rate with {}".format(self._instrument.clock_source.get()))

        if rate == '1GHz_REFERENCE_CLOCK':
            rate = 1e9

        decimation = self._instrument.decimation.get()
        if decimation > 0:
            rate = rate / decimation

        self._save_val(rate)
        return rate


class DemodFreqParameter(ArrayParameter):

    #
    def __init__(self, name, shape, **kwargs):
        self._demod_freqs = []
        super().__init__(name, shape, **kwargs)


    def add_demodulator(self, demod_freq):
        ndemod_freqs = len(self._demod_freqs)
        if demod_freq not in self._demod_freqs:
            self._verify_demod_freq(demod_freq)
            self._demod_freqs.append(demod_freq)
            self._save_val(self._demod_freqs)
            self.shape = (ndemod_freqs+1,)
            self._instrument.acquisition.set_setpoints_and_labels()

    def remove_demodulator(self, demod_freq):
        ndemod_freqs = len(self._demod_freqs)
        if demod_freq in self._demod_freqs:
            self._demod_freqs.pop(self._demod_freqs.index(demod_freq))
            self.shape = (ndemod_freqs - 1,)
            self._save_val(self._demod_freqs)
            self._instrument.acquisition.set_setpoints_and_labels()

    def get(self):
        return self._demod_freqs

    def get_num_demods(self):
        return len(self._demod_freqs)

    def get_max_demod_freq(self):
        if len(self._demod_freqs):
            return max(self._demod_freqs)
        else:
            return None

    def _verify_demod_freq(self, value):
        """
        Function to validate a demodulation frequency

        Checks:
            - 1e6 <= value <= 500e6
            - number of oscillation measured using current 'int_time' param value
              at this demodulation frequency value
            - oversampling rate for this demodulation frequency

        Args:
            value: proposed demodulation frequency
        Returns:
            bool: Returns True if suitable number of oscillations are measured and
            oversampling is > 1, False otherwise.
        Raises:
            ValueError: If value is not a valid demodulation frequency.
        """
        if (value is None) or not (1e6 <= value <= 500e6):
            raise ValueError('demod_freqs must be 1e6 <= value <= 500e6')
        isValid = True
        alazar = self._instrument._get_alazar()
        sample_rate = alazar.effective_sample_rate.get()
        int_time = self._instrument.int_time.get()
        min_oscillations_measured = int_time * value
        oversampling = sample_rate / (2 * value)
        if min_oscillations_measured < 10:
            isValid = False
            logging.warning('{} oscillation measured for largest '
                            'demod freq, recommend at least 10: '
                            'decrease sampling rate, take '
                            'more samples or increase demodulation '
                            'freq'.format(min_oscillations_measured))
        elif oversampling < 1:
            isValid = False
            logging.warning('oversampling rate is {}, recommend > 1: '
                            'increase sampling rate or decrease '
                            'demodulation frequency'.format(oversampling))

        return isValid