from typing import TYPE_CHECKING, ClassVar

from qcodes import validators as vals
from qcodes.instrument import (
    ChannelList,
    Instrument,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import Parameter, create_on_off_val_mapping

if TYPE_CHECKING:
    from typing_extensions import Unpack


class NotKnownModel(Exception):
    """
    An Error thrown when connecting to an unknown Aim TTi model
    """

    pass


class AimTTiChannel(InstrumentChannel):
    """
    This is the class that holds the output channels of AimTTi power
    supply.
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        channel: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The 'colloquial' name of the channel.
            channel: The name used by the AimTTi.
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(parent, name, **kwargs)

        self.channel = channel
        # The instrument can store up to ten configurations
        # internally.
        self.set_up_store_slots = [i for i in range(0, 10)]

        self.volt: Parameter = self.add_parameter(
            "volt",
            get_cmd=self._get_voltage_value,
            get_parser=float,
            set_cmd=f"V{channel} {{}}",
            label="Voltage",
            unit="V",
        )
        """Parameter volt"""

        self.volt_step_size: Parameter = self.add_parameter(
            "volt_step_size",
            get_cmd=self._get_voltage_step_size,
            get_parser=float,
            set_cmd=f"DELTAV{channel} {{}}",
            label="Voltage Step Size",
            unit="V",
        )
        """Parameter volt_step_size"""

        self.curr: Parameter = self.add_parameter(
            "curr",
            get_cmd=self._get_current_value,
            get_parser=float,
            set_cmd=f"I{channel} {{}}",
            label="Current",
            unit="A",
        )
        """Parameter curr"""

        self.curr_range: Parameter = self.add_parameter(
            "curr_range",
            get_cmd=f"IRANGE{channel}?",
            get_parser=int,
            set_cmd=self._set_current_range,
            label="Current Range",
            unit="A",
            vals=vals.Numbers(1, 2),
            docstring="Set the current range of the output. "
            "Here, the integer 1 is for the Low range, "
            "and integer 2 is for the High range.",
        )
        """Set the current range of the output. Here, the integer 1 is for the Low range, and integer 2 is for the High range."""

        self.curr_step_size: Parameter = self.add_parameter(
            "curr_step_size",
            get_cmd=self._get_current_step_size,
            get_parser=float,
            set_cmd=f"DELTAI{channel} {{}}",
            label="Current Step Size",
            unit="A",
        )
        """Parameter curr_step_size"""

        self.output: Parameter = self.add_parameter(
            "output",
            get_cmd=f"OP{channel}?",
            get_parser=float,
            set_cmd=f"OP{channel} {{}}",
            val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
        )
        """Parameter output"""

    def _get_voltage_value(self) -> float:
        channel_id = self.channel
        _voltage = self.ask_raw(f"V{channel_id}?")
        _voltage_split = _voltage.split()
        return float(_voltage_split[1])

    def _get_current_value(self) -> float:
        channel_id = self.channel
        _current = self.ask_raw(f"I{channel_id}?")
        _current_split = _current.split()
        return float(_current_split[1])

    def _get_voltage_step_size(self) -> float:
        channel_id = self.channel
        _voltage_step_size = self.ask_raw(f"DELTAV{channel_id}?")
        _v_step_size_split = _voltage_step_size.split()
        return float(_v_step_size_split[1])

    def _get_current_step_size(self) -> float:
        channel_id = self.channel
        _current_step_size = self.ask_raw(f"DELTAI{channel_id}?")
        _c_step_size_split = _current_step_size.split()
        return float(_c_step_size_split[1])

    def _set_current_range(self, val: int) -> None:
        """
        This is the private function that ensures that the output is switched
        off before changing the current range, as pointed out by the instrument
        manual.
        """
        channel_id = self.channel
        with self.output.set_to(False):
            self.write(f"IRANGE{channel_id} {val}")

    def increment_volt_by_step_size(self) -> None:
        """
        A bound method that increases the voltage output of the corresponding
        channel by an amount the step size set by the user via ``volt_step_size``
        parameter.
        """
        channel_id = self.channel
        self.write(f"INCV{channel_id}")
        # Clear the cache.
        _ = self.volt.get()

    def decrement_volt_by_step_size(self) -> None:
        """
        A bound method that decreases the voltage output of the corresponding
        channel by an amount the step size set by the user via ``volt_step_size``
        parameter.
        """
        channel_id = self.channel
        self.write(f"DECV{channel_id}")
        # Clear the cache.
        _ = self.volt.get()

    def increment_curr_by_step_size(self) -> None:
        """
        A bound method that increases the current output of the corresponding
        channel by an amount the step size set by the user via ``curr_step_size``
        parameter.
        """
        channel_id = self.channel
        self.write(f"INCI{channel_id}")
        # Clear the cache.
        _ = self.curr.get()

    def decrement_curr_by_step_size(self) -> None:
        """
        A bound method that decreases the current output of the corresponding
        channel by an amount the step size set by the user via ``curr_step_size``
        parameter.
        """
        channel_id = self.channel
        self.write(f"DECI{channel_id}")
        # Clear the cache.
        _ = self.curr.get()

    def save_setup(self, slot: int) -> None:
        """
        A bound function that saves the output setup to the internal
        store specified by the numbers 0-9.
        """
        if slot not in self.set_up_store_slots:
            raise RuntimeError("Slot number should be an integer between 0 and 9.")

        channel_id = self.channel
        self.write(f"SAV{channel_id} {slot}")

    def load_setup(self, slot: int) -> None:
        """
        A bound function that loads the output setup from the internal
        store specified by the numbers 0-9.
        """
        if slot not in self.set_up_store_slots:
            raise RuntimeError("Slot number should be an integer between 0 and 9.")

        channel_id = self.channel
        self.write(f"RCL{channel_id} {slot}")
        # Update snapshot after load.
        _ = self.snapshot(update=True)

    def set_damping(self, val: int) -> None:
        """
        Sets the current meter measurement averaging on and off.
        """
        if val not in [0, 1]:
            raise RuntimeError(
                "To 'turn on' and 'turn off' the averaging, "
                "use '1' and '0', respectively."
            )
        channel_id = self.channel
        self.write(f"DAMPING{channel_id} {val}")


class AimTTi(VisaInstrument):
    """
    Base class for Aim TTi PL-P series power supply.
    This class should not be instantiated directly, but rather one of the
    subclasses corresponding to the specific model of the power supply should
    be used.
    """

    _numOutputChannels: ClassVar[dict[str, int]] = {
        "PL068-P": 1,
        "PL155-P": 1,
        "PL303-P": 1,
        "PL601-P": 1,
        "PL303QMD-P": 2,
        "PL303QMT-P": 3,
        "QL355TP": 3,
    }
    default_terminator = "\n"

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS.
            address: VISA resource address
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name, address, **kwargs)

        channels = ChannelList(self, "Channels", AimTTiChannel, snapshotable=False)

        _model = self.get_idn()["model"]

        if (_model not in self._numOutputChannels.keys()) or (_model is None):
            raise NotKnownModel("Unknown model, connection cannot be established.")

        self.numOfChannels = self._numOutputChannels[_model]
        for i in range(1, self.numOfChannels + 1):
            channel = AimTTiChannel(self, f"ch{i}", i)
            channels.append(channel)
            self.add_submodule(f"ch{i}", channel)

        self.add_submodule("channels", channels.to_channel_tuple())
        self.connect_message()

    # Interface Management

    def get_idn(self) -> dict[str, str | None]:
        """
        Returns the instrument identification including vendor, model, serial
        number and the firmware.
        """
        IDNstr = self.ask_raw("*IDN?")
        vendor, model, serial, firmware = map(str.strip, IDNstr.split(","))

        IDN: dict[str, str | None] = {
            "vendor": vendor,
            "model": model,
            "serial": serial,
            "firmware": firmware,
        }
        return IDN

    def get_address(self) -> int:
        """
        Returns the bus address.
        """
        busAddressStr = self.ask_raw("ADDRESS?")
        busAddress = busAddressStr.strip()
        return int(busAddress)

    def get_IP(self) -> str:
        """
        Returns the IP address of the LAN interface, if the connection exists.
        If there is a pre-configured static IP and the instrument is not
        connected to a LAN interface, that static IP will be returned.
        Otherwise, the return value is '0.0.0.0'.
        """
        ipAddress = self.ask_raw("IPADDR?")
        return ipAddress.strip()

    def get_netMask(self) -> str:
        """
        Returns the netmask of the LAN interface, if the connection exists.
        """
        netMask = self.ask_raw("NETMASK?")
        return netMask.strip()

    def get_netConfig(self) -> str:
        """
        Returns the means by which an IP address is acquired, i.e.,
        DHCP, AUTO or STATIC.
        """
        netConfig = self.ask_raw("NETCONFIG?")
        return netConfig.strip()

    def local_mode(self) -> None:
        """
        Go to local mode until the next remote command is recieved. This
        function does not release any active interface lock.
        """
        self.write("LOCAL")

    def is_interface_locked(self) -> int:
        """
        Returns '1' if the interface lock is owned by the requesting instance,
        '0' if there is no active lock and '-1' if the lock is unavailable.
        """
        is_lockedSTR = self.ask_raw("IFLOCK?")
        is_locked = is_lockedSTR.strip()
        return int(is_locked)

    def lock_interface(self) -> int:
        """
        Requests instrument interface lock. Returns '1' if successful and
        '-1' if the lock is unavailable.
        """
        lockSTR = self.ask_raw("IFLOCK")
        lock = lockSTR.strip()
        return int(lock)

    def unlock_interface(self) -> int:
        """
        Requests the release of instrument interface lock. Returns '0'
        if successful and '-1' if unsuccessful.
        """
        unlockSTR = self.ask_raw("IFUNLOCK")
        unlock = unlockSTR.strip()
        return int(unlock)
