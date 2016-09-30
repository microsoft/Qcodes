"""
Metaclass to record instances of an Instrument or RemoteInstrument
only if it was created successfully.
"""


class InstrumentMetaclass(type):
    def __call__(cls, *args, **kwargs):
        # create the instrument. This calls the entire __init__ chain and
        # returns the fully initialized object, so if failures happen in
        # a subclass, they will prevent recording the instance
        instrument = super().__call__(*args, **kwargs)

        # for RemoteInstrument, we want to record this instance with the
        # class that it proxies, not with RemoteInstrument itself
        cls = getattr(instrument, '_instrument_class', type(instrument))
        cls.record_instance(instrument)

        return instrument
