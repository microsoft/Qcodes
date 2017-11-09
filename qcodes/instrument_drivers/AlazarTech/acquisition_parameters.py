from qcodes import Parameter

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
