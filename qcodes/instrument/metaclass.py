"""Metaclass to choose between Instrument and RemoteInstrument"""

import warnings

from .remote import RemoteInstrument


class InstrumentMetaclass(type):
    def __call__(cls, *args, server_name=None, **kwargs):
        """
        Create either a real Instrument or a RemoteInstrument as requested.

        This (metaclass.__call__) is what is actually executed when you
        instantiate an instrument, and returns the fully initialized object
        (unlike class.__new__ which returns before __init__) so we can use this
        to determine if the object was successfully created and only then
        record its instance.

        Args:
            cls (type): the specific instrument class you invoked

            *args (List[Any]): positional args to the instrument constructor

            server_name (Optional[Union[str, None]]): if ``None`` we construct
                a local instrument (with the class you requested). If a string,
                we construct this instrument on a server with that name, or the
                default from the instrument's classmethod
                ``default_server_name`` if a blank string is used)

            **kwargs (Dict[Any]): the kwargs to the instrument constructor,
                after omitting server_name
        """
        if server_name is None:
            instrument = super().__call__(*args, **kwargs)
        else:
            warnings.warn('Multiprocessing is in beta, use at own risk',
                          UserWarning)
            instrument = RemoteInstrument(*args, instrument_class=cls,
                                          server_name=server_name, **kwargs)

        # for RemoteInstrument, we want to record this instance with the
        # class that it proxies, not with RemoteInstrument itself
        # cls.record_instance(instrument)

        return instrument
