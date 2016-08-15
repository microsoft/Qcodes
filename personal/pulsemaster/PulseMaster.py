from qcodes import Instrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils import validators as vals


class PulseMaster(Instrument):
    shared_kwargs = ['instruments']
    def __init__(self, name, instruments, **kwargs):
        super().__init__(name, **kwargs)

        self.instruments = {instrument.name: instrument for instrument in instruments}
        self.no_instruments = len(instruments)

        # self.add_parameter('instruments',
        #                    parameter_class=ManualParameter,
        #                    initial_value={},
        #                    vals=vals.Anything())
        #
        self.add_parameter('trigger_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self.instruments.keys()))

        self.add_parameter('acquisition_instrument',
                           parameter_class=ManualParameter,
                           initial_value=None,
                           vals=vals.Enum(*self.instruments.keys()))

    # def add_instrument(self, instrument,
    #                    acquisition_instrument=False):
    #     assert isinstance(instrument, Instrument),\
    #            'Can only add instruments that are a subclass of Instrument'
    #     assert instrument.name not in self.instruments.keys(),\
    #            'Instrument with same name is already added'
    #
    #     if acquisition_instrument:
    #         self.acquisition_instrument = instrument.name

class Connection():
    def __init__(self, PulseMaster,
               master_instrument_name, master_instrument_channel,
               slave_instrument_name, slave_instrument_channel,
               delay=0):
        self.PulseMaster = PulseMaster

        self.master_instrument = PulseMaster.instruments[master_instrument_name]
        self.master_instrument_channel = master_instrument_channel

        self.slave_instrument = PulseMaster.instruments[slave_instrument_name]
        self.slave_instrument_channel = slave_instrument_channel

        self.delay = delay