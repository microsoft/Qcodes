from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList


class RohdeSchwarzHMC804xChannel(InstrumentChannel):
    def __init__(self, parent, name, channel):
        super().__init__(parent, name)

        select_cmd = ":INSTrument:NSELect {};".format(channel)

        self.add_parameter("set_voltage",
                           label='Target voltage output',
                           set_cmd="{} :SOURce:VOLTage:LEVel:IMMediate:AMPLitude {}".format(
                               select_cmd, '{}'),
                           get_cmd="{} :SOURce:VOLTage:LEVel:IMMediate:AMPLitude?".format(
                               select_cmd),
                           get_parser=float,
                           unit='V',
                           vals=vals.Numbers(0, 32.050)
                          )
        self.add_parameter("set_current",
                           label='Target current output',
                           set_cmd="{} :SOURce:CURRent:LEVel:IMMediate:AMPLitude {}".format(
                               select_cmd, '{}'),
                           get_cmd="{} :SOURce:CURRent:LEVel:IMMediate:AMPLitude?".format(
                               select_cmd),
                           get_parser=float,
                           unit='A',
                           vals=vals.Numbers(0.5e-3, self._parent.max_current)
                           )
        self.add_parameter('state',
                           label='Output enabled',
                           set_cmd='{} :OUTPut:CHANnel:STATe {}'.format(select_cmd, '{}'),
                           get_cmd='{} :OUTPut:CHANnel:STATe?'.format(select_cmd),
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF')
                           )
        self.add_parameter("voltage",
                           label='Measured voltage',
                           get_cmd="{} :MEASure:SCALar:VOLTage:DC?".format(
                               select_cmd),
                           get_parser=float,
                           unit='V',
                          )
        self.add_parameter("current",
                           label='Measured current',
                           get_cmd="{} :MEASure:SCALar:CURRent:DC?".format(
                               select_cmd),
                           get_parser=float,
                           unit='A',
                           )
        self.add_parameter("power",
                           label='Measured power',
                           get_cmd="{} :MEASure:SCALar:POWer?".format(
                               select_cmd),
                           get_parser=float,
                           unit='W',
                           )


class RohdeSchwarzHMC804x(VisaInstrument):
    """
    This is the qcodes driver for the Rohde & Schwarz HMC804x Power Supply
    """

    _max_currents = {3: 3.0, 2: 5.0, 1: 10.0}

    def __init__(self, name, address, num_channels, **kwargs):
        super().__init__(name, address, **kwargs)

        self.max_current = RohdeSchwarzHMC804x._max_currents[num_channels]

        self.add_parameter('state',
                           label='Output enabled',
                           set_cmd='OUTPut:MASTer:STATe {}',
                           get_cmd='OUTPut:MASTer:STATe?',
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF')
                           )

        # channel-specific parameters
        channels = ChannelList(self, "SupplyChannel", RohdeSchwarzHMC804xChannel)
        for ch_num in range(1, num_channels+1):
            ch_name = "ch{}".format(ch_num)
            channel = RohdeSchwarzHMC804xChannel(self, ch_name, ch_num)
            channels.append(channel)
            self.add_submodule(ch_name, channel)
        channels.lock()
        self.add_submodule("channels", channels)

        self.connect_message()


class RohdeSchwarzHMC8041(RohdeSchwarzHMC804x):
    """
    This is the qcodes driver for the Rohde & Schwarz HMC8041 Power Supply
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, num_channels=1, **kwargs)


class RohdeSchwarzHMC8042(RohdeSchwarzHMC804x):
    """
    This is the qcodes driver for the Rohde & Schwarz HMC8042 Power Supply
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, num_channels=2, **kwargs)


class RohdeSchwarzHMC8043(RohdeSchwarzHMC804x):
    """
    This is the qcodes driver for the Rohde & Schwarz HMC8043 Power Supply
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, num_channels=3, **kwargs)
