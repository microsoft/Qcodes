from typing import Any, List, Sequence, Tuple

from qcodes import ChannelList, InstrumentChannel, VisaInstrument
from qcodes import validators as vals


class RigolDP8xxChannel(InstrumentChannel):
    def __init__(
            self,
            parent: "_RigolDP8xx",
            name: str,
            channel: int,
            ch_range: Tuple[float, float],
            ovp_range: Tuple[float, float],
            ocp_range: Tuple[float, float]
    ):
        super().__init__(parent, name)

        self.vmax = ch_range[0]
        self.imax = ch_range[1]
        self.ovp_range = ovp_range
        self.ocp_range = ocp_range

        select_cmd = f":INSTrument:NSELect {channel};"
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
                           get_cmd=f'{select_cmd} :OUTPut:STATe?',
                           get_parser=strstrip,
                           vals=vals.OnOff()
                           )
        self.add_parameter('mode',
                           label='Get the output mode',
                           get_cmd=f'{select_cmd} :OUTPut:MODE?',
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
                           get_cmd=f'{select_cmd} :VOLTage:PROTection:STATe?',
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
                           get_cmd=f'{select_cmd} :CURRent:PROTection:STATe?',
                           get_parser=strstrip,
                           vals=vals.OnOff()
                           )


class _RigolDP8xx(VisaInstrument):
    """
    This is the general DP8xx Power Supply driver class that implements shared parameters and functionality
    among all similar power supply from Rigole.

    This driver was written to be inherited from by a specific driver (e.g. DP832).
    """

    def __init__(
            self,
            name: str,
            address: str,
            channels_ranges: Sequence[Tuple[float, float]],
            ovp_ranges: Tuple[Sequence[Tuple[float, float]],
                              Sequence[Tuple[float, float]]],
            ocp_ranges: Tuple[Sequence[Tuple[float, float]],
                              Sequence[Tuple[float, float]]],
            **kwargs: Any
    ):
        super().__init__(name, address, **kwargs)

        # Check if precision extension has been installed
        opt = self.installed_options()
        if 'DP8-ACCURACY' in opt:
            ovp_ranges_selected = ovp_ranges[1]
            ocp_ranges_selected = ocp_ranges[1]
        else:
            ovp_ranges_selected = ovp_ranges[0]
            ocp_ranges_selected = ocp_ranges[0]

        # channel-specific parameters
        channels = ChannelList(self, "SupplyChannel", RigolDP8xxChannel, snapshotable=False)
        for ch_num, channel_range in enumerate(channels_ranges):
            ch_name = f"ch{ch_num + 1}"
            channel = RigolDP8xxChannel(
                self,
                ch_name,
                ch_num + 1,
                channel_range,
                ovp_ranges_selected[ch_num],
                ocp_ranges_selected[ch_num]
            )
            channels.append(channel)
            self.add_submodule(ch_name, channel)
        self.add_submodule("channels", channels.to_channel_tuple())

        self.connect_message()

    def installed_options(self) -> List[str]:
        """Return the installed options"""

        opt = self.ask("*OPT?")
        optl = opt.strip().split(',')
        optl_clean = [x for x in optl if x != '0']
        return optl_clean
