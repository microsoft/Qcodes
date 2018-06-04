from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList


class RigolDP8xxChannel(InstrumentChannel):
    def __init__(self, parent, name, channel, ch_range, ovp_range, ocp_range):
        super().__init__(parent, name)

        self.vmax = ch_range[0]
        self.imax = ch_range[1]
        self.ovp_range = ovp_range
        self.ocp_range = ocp_range

        select_cmd = ":INSTrument:NSELect {};".format(channel)
        strstrip = lambda s: str(s).strip()

        self.add_parameter("set_voltage",
                           label='Target voltage output',
                           set_cmd="{} :SOURce:VOLTage:LEVel:IMMediate:AMPLitude {}".format(
                               select_cmd, '{}'),
                           get_cmd="{} :SOURce:VOLTage:LEVel:IMMediate:AMPLitude?".format(
                               select_cmd),
                           get_parser=float,
                           unit='V',
                           vals=vals.Numbers(min(0, self.vmax), max(0, self.vmax))
                          )
        self.add_parameter("set_current",
                           label='Target current output',
                           set_cmd="{} :SOURce:CURRent:LEVel:IMMediate:AMPLitude {}".format(
                               select_cmd, '{}'),
                           get_cmd="{} :SOURce:CURRent:LEVel:IMMediate:AMPLitude?".format(
                               select_cmd),
                           get_parser=float,
                           unit='A',
                           vals=vals.Numbers(0, self.imax)
                           )
        self.add_parameter('state',
                           label='Output enabled',
                           set_cmd='{} :OUTPut:STATe {}'.format(select_cmd, '{}'),
                           get_cmd='{} :OUTPut:STATe?'.format(select_cmd),
                           get_parser=strstrip,
                           vals=vals.OnOff()
                           )
        self.add_parameter('mode',
                           label='Get the output mode',
                           get_cmd='{} :OUTPut:MODE?'.format(select_cmd),
                           get_parser=strstrip,
                           val_mapping={'ConstantVoltage': 'CV',
                                        'ConstantCurrent': 'CC',
                                        'Unregulated': 'UR'}
                          )
        self.add_parameter("voltage",
                           label='Measured voltage',
                           get_cmd="{} :MEASure:VOLTage:DC?".format(
                               select_cmd),
                           get_parser=float,
                           unit='V',
                          )
        self.add_parameter("current",
                           label='Measured current',
                           get_cmd="{} :MEASure:CURRent:DC?".format(
                               select_cmd),
                           get_parser=float,
                           unit='A',
                           )
        self.add_parameter("power",
                           label='Measured power',
                           get_cmd="{} :MEASure:POWer?".format(
                               select_cmd),
                           get_parser=float,
                           unit='W',
                           )
        self.add_parameter("ovp_value",
                           label='Over Voltage Protection value',
                           set_cmd="{} :VOLTage:PROTection:LEVel {}".format(
                               select_cmd, '{}'),
                           get_cmd="{} :VOLTage:PROTection:LEVel?".format(
                               select_cmd),
                           get_parser=float,
                           unit='V',
                           vals=vals.Numbers(self.ovp_range[0], self.ovp_range[1])
                           )
        self.add_parameter('ovp_state',
                           label='Over Voltage Protection status',
                           set_cmd='{} :VOLTage:PROTection:STATe {}'.format(select_cmd, '{}'),
                           get_cmd='{} :VOLTage:PROTection:STATe?'.format(select_cmd),
                           get_parser=strstrip,
                           vals=vals.OnOff()
                           )
        self.add_parameter("ocp_value",
                           label='Over Current Protection value',
                           set_cmd="{} :CURRent:PROTection:LEVel {}".format(
                               select_cmd, '{}'),
                           get_cmd="{} :CURRent:PROTection:LEVel?".format(
                               select_cmd),
                           get_parser=float,
                           unit='A',
                           vals=vals.Numbers(self.ocp_range[0], self.ocp_range[1])
                           )
        self.add_parameter('ocp_state',
                           label='Over Current Protection status',
                           set_cmd='{} :CURRent:PROTection:STATe {}'.format(select_cmd, '{}'),
                           get_cmd='{} :CURRent:PROTection:STATe?'.format(select_cmd),
                           get_parser=strstrip,
                           vals=vals.OnOff()
                           )

class _RigolDP8xx(VisaInstrument):
    """
    This is the general DP8xx Power Supply driver class that implements shared parameters and functionality
    among all similar power supply from Rigole.

    This driver was written to be inherited from by a specific driver (e.g. DP832).
    """

    def __init__(self, name, address, channels_ranges, ovp_ranges, ocp_ranges, **kwargs):
        super().__init__(name, address, **kwargs)

        #Check if precision extension has been installed
        opt = self.installed_options()
        if 'DP8-ACCURACY' in opt:
            ovp_ranges = ovp_ranges[1]
            ocp_ranges = ocp_ranges[1]
        else:
            ovp_ranges = ovp_ranges[0]
            ocp_ranges = ocp_ranges[0]

        # channel-specific parameters
        channels = ChannelList(self, "SupplyChannel", RigolDP8xxChannel, snapshotable=False)
        for ch_num, channel_range in enumerate(channels_ranges):
            ch_name = "ch{}".format(ch_num + 1)
            channel = RigolDP8xxChannel(self, ch_name, ch_num + 1, channel_range, ovp_ranges[ch_num], ocp_ranges[ch_num])
            channels.append(channel)
            self.add_submodule(ch_name, channel)
        channels.lock()
        self.add_submodule("channels", channels)

        self.connect_message()

    def installed_options(self):
        """Return the installed options"""

        opt = self.ask("*OPT?")
        optl = opt.strip().split(',')
        optl_clean = [x for x in optl if x != '0']
        return optl_clean
