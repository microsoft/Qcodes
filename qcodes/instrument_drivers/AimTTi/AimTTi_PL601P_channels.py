from typing import List, Dict, Optional

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList
from qcodes.instrument.base import Instrument, Parameter
from qcodes.utils.helpers import create_on_off_val_mapping


class NotKnownModel(Exception):
    pass

class AimTTiChannel(InstrumentChannel):
    """
    This is the class that holds the output channels of AimTTi power
    supply.
    """
    def __init__(self, parent: Instrument, name: str,
                 channel: str, **kwargs) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The 'colloquial' name of the channel.
            channel: The name used by the AimTTi.
        """
        super().__init__(parent, name, **kwargs)

        self.channel = channel
        # The instrument can store up to ten configurations
        # internally.
        self.set_up_store_slots = [i for i in range(0, 10)]

        self.add_parameter('volt',
                           get_cmd=self._get_voltage_value,
                           get_parse=float,
                           set_cmd=f'V{channel} {{}}',
                           label='Voltage',
                           unit='V')

        self.add_parameter('volt_step_size',
                           get_cmd=self._get_voltage_step_size,
                           get_parser=float,
                           set_cmd=f'DELTAV{channel} {{}}',
                           label='Voltage Step Size',
                           unit='V')

        self.add_parameter('curr',
                           get_cmd=self._get_current_value,
                           get_parser=float,
                           set_cmd=f'I{channel} {{}}',
                           label='Current',
                           unit='A')

        self.add_parameter('curr_range',
                           get_cmd=f'IRANGE{channel}?',
                           get_parser=int,
                           set_cmd=self._set_current_range,
                           label='Current Range',
                           unit='A',
                           vals=vals.Numbers(1, 2),
                           docstring='Set the current range of the output.'
                           'Here, the integer 1 is for the Low range, '
                           'and integer 2 is for the High range.')

        self.add_parameter('curr_step_size',
                           get_cmd=self._get_current_step_size,
                           get_parser=float,
                           set_cmd=f'DELTAI{channel} {{}}',
                           label='Current Step Size',
                           unit='A')

        self.add_parameter('output',
                           get_cmd=f'OP{channel}?',
                           get_parser=float,
                           set_cmd=f'OP{channel} {{}}',
                           val_mapping=create_on_off_val_mapping(on_val=1,
                                                                 off_val=0))

    def _get_voltage_value(self) -> float:
        channel_id = self.channel
        _voltage = self.ask_raw(f'V{channel_id}?')
        _voltage_split = _voltage.split()
        return float(_voltage_split[1])

    def _get_current_value(self) -> float:
        channel_id = self.channel
        _current = self.ask_raw(f'I{channel_id}?')
        _current_split = _current.split()
        return float(_current_split[1])

    def _get_voltage_step_size(self) -> float:
        channel_id = self.channel
        _voltage_step_size = self.ask_raw(f'DELTAV{channel_id}?')
        _v_step_size_split = _voltage_step_size.split()
        return float(_v_step_size_split[1])

    def _get_current_step_size(self) -> float:
        channel_id = self.channel
        _current_step_size = self.ask_raw(f'DELTAI{channel_id}?')
        _c_step_size_split = _current_step_size.split()
        return float(_c_step_size_split[1])

    def _set_current_range(self, val: int) -> None:
        """
        This is the private function that ensures that the output is switched
        off before changing the current range, as pointed out by the instrument
        manual.
        """
        channel_id = self.channel
        self.output(False)
        self.write(f'IRANGE{channel_id} {val}')

    def increment_volt_by_step_size(self) -> None:
        """
        A bound method that increases the voltage output of the corresponding
        channel by an amount the step size set by the user via ``volt_step_size``
        parameter.
        """
        channel_id = self.channel
        self.write(f'INCV{channel_id}')

    def decrement_volt_by_step_size(self) -> None:
        """
        A bound method that decreases the voltage output of the corresponding
        channel by an amount the step size set by the user via ``volt_step_size``
        parameter.
        """
        channel_id = self.channel
        self.write(f'DECV{channel_id}')

    def increment_curr_by_step_size(self) -> None:
        """
        A bound method that increases the current output of the corresponding
        channel by an amount the step size set by the user via ``curr_step_size``
        parameter.
        """
        channel_id = self.channel
        self.write(f'INCI{channel_id}')

    def decrement_curr_by_step_size(self) -> None:
        """
        A bound method that decreases the current output of the corresponding
        channel by an amount the step size set by the user via ``curr_step_size``
        parameter.
        """
        channel_id = self.channel
        self.write(f'DECI{channel_id}')

    def save_setup(self, slote: int) -> None:
        """
        A bound function that saves the output setup to the internal
        store specified by the numbers 0-9.
        """
        if not slote in self.set_up_store_slots:
            raise RuntimeError("Slote number should be an integer between"
                               "0 adn 9.")

        channel_id = self.channel
        self.write(f'SAV{channel_id} {slote}')

    def load_setup(self, slote: int) -> None:
        """
        A bound function that loadss the output setup from the internal
        store specified by the numbers 0-9.
        """
        if not slote in self.set_up_store_slots:
            raise RuntimeError("Slote number should be an integer between"
                               "0 adn 9.")

        channel_id = self.channel
        self.write(f'RCL{channel_id} {slote}')


class AimTTi(VisaInstrument):
    """
    This is the QCoDeS driver for the Aim TTi PL-P series power supply.
    Tested with Aim TTi PL601-P equipped with a single output channel.
    """
    def __init__(self, name: str, address: str) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS.
            address: VISA resource address
        """
        super().__init__(name, address, terminator='\n')

        channels = ChannelList(self, "Channels", AimTTiChannel,
                               snapshotable=False)

        _model = self.get_idn()['model']

        _numOutputChannels = {'PL068-P': 1, 'PL155-P': 1, 'PL303-P': 1,
                             'PL601-P': 1, 'PL303QMD-P': 2, 'PL303QMT': 3}

        if not _model in _numOutputChannels.keys():
            raise NotKnownModel("Unknown model, connection cannot be "
                                "established.")

        self.numOfChannels = _numOutputChannels[_model]
        for i in range(1, self.numOfChannels+1):
            channel = AimTTiChannel(self, f'ch{i}', i)
            channels.append(channel)
            self.add_submodule(f'ch{i}', channel)

        channels.lock()
        self.add_submodule('channels', channels)
        self.connect_message()

    # Interface Management

    def get_idn(self) -> Dict[str, Optional[str]]:
        """
        Returns the instrument identification including vendor, model, serial
        number and the firmware.
        """
        IDNstr = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDNstr.split(','))

        IDN: Dict[str, Optional[str]] = {'vendor': vendor, 'model': model,
                                         'serial': serial, 'firmware': firmware}
        return IDN

    def get_address(self) -> int:
        """
        Returns the bus address.
        """
        busAddressStr = self.ask_raw('ADDRESS?')
        busAddress = busAddressStr.strip()
        return int(busAddress)

    def get_IP(self) -> str:
        """
        Returns the IP address of the LAN interface, if the connection exists.
        If there is a pre-configured static IP and the instrument is not
        connected to a LAN interface, that static IP will be returned.
        Otherwise, the return value is '0.0.0.0'.
        """
        ipAddress = self.ask_raw('IPADDR?')
        return ipAddress.strip()

    def get_netMask(self) -> str:
        """
        Returns the netmask of the LAN interface, if the connection exists.
        """
        netMask = self.ask_raw('NETMASK?')
        return netMask.strip()

    def get_netConfig(self) -> str:
        """
        Returns the means by which an IP address is acquired, i.e.,
        DHCP, AUTO or STATIC.
        """
        netConfig = self.ask_raw('NETCONFIG?')
        return netConfig.strip()
