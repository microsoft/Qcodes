from typing import Dict, Optional, Any, Callable
import re
import numpy as np
import logging

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Numbers


logger = logging.getLogger()


class StahlChannel(InstrumentChannel):
    def __init__(self, parent: VisaInstrument, name: str, channel_number: int):
        super().__init__(parent, name)

        self._channel_string = f"{channel_number:02d}"
        self._channel_number = channel_number
        self._acknowledge_reply = chr(6)

        self.add_parameter(
            "voltage",
            get_cmd=f"{self.parent.identifier} U{self._channel_string}",
            get_parser=self._stahl_get_parser("V"),
            set_cmd=self._set_voltage,
            unit="V",
            vals=Numbers(
                -self.parent.voltage_range,
                self.parent.voltage_range
            )
        )

        self.add_parameter(
            "current",
            get_cmd=f"{self.parent.identifier} I{self._channel_string}",
            get_parser=self._stahl_get_parser("mA"),
            unit="mA",
        )

        self.add_parameter(
            "is_locked",
            get_cmd=self._get_lock_status,
            get_parser={"0": False, "1": True}.get,
        )

    @staticmethod
    def _stahl_get_parser(unit: str) -> Callable:
        regex = f"([\+\-][\d]?\d,\d{{3}} ){unit}$"

        def parser(response):
            result = re.search(regex, response).groups()[0]
            return float(result.replace(",", "."))

        return parser

    def _set_voltage(self, voltage: float) -> None:
        voltage_normalized = np.interp(
            voltage,
            self.parent.voltage_range * np.array([-1, 1]),
            [0, 1]
        )

        send_string = f"{self.parent.identifier} CH{self._channel_string} {voltage_normalized:.5f}"
        response = self.ask(send_string)  # This is not a bug. We are asking in a set

        if response != self._acknowledge_reply:
            logger.warning(f"Command {send_string} did not produce an acknowledge reply")

    def _get_lock_status(self) -> chr:
        send_string = f"{self.parent.identifier} LOCK"

        response = self.parent.visa_handle.query_binary_values(
            send_string,
            datatype='B',
            header_fmt="empty"
        )

        chnr = self._channel_number - 1
        channel_group = chnr // 4
        lock_code_group = response[channel_group]
        return format(lock_code_group, "b")[chnr % 4 + 1]


class Stahl(VisaInstrument):
    def __init__(self, name: str, address: str):
        super().__init__(name, address, terminator="\r")
        self.visa_handle.baud_rate = 115200

        instrument_info = self._parse_idn_string(
            self.ask("IDN")
        )

        for key, value in instrument_info.items():
            setattr(self, key, value)

        channels = ChannelList(
            self, "channel", StahlChannel, snapshotable=False
        )

        for channel_number in range(1, self.n_channels + 1):
            name = f"channel{channel_number}"
            channel = StahlChannel(
                self,
                name,
                channel_number
            )
            self.add_submodule(name, channel)
            channels.append(channel)

        self.add_submodule("channel", channels)

        self.add_parameter(
            "temperature",
            get_cmd=self._get_temperature,
            unit="C"
        )

        self.connect_message()

    def _get_temperature(self) -> float:

        send_string = f"{self.identifier} TEMP"
        response_characters = self.visa_handle.query_binary_values(
            send_string,
            datatype='B',
            header_fmt="empty"
        )

        response_string = "".join(map(chr, response_characters))
        # The '°' is the reason why we cannot simply call 'ask'
        # Stupid instrument :-(
        groups = re.search("TEMP (.*)°C", response_string).groups()
        return float(groups[0])

    @staticmethod
    def _parse_idn_string(ind_string) -> Dict[str, Any]:

        groups = re.search(
            "(HV|BS)(\d{3}) (\d{3}) (\d{2}) [buqsm]",
            ind_string
        ).groups()

        idparsers = {
            "model": str,
            "serial_number": str,
            "voltage_range": float,
            "n_channels": int,
            "output_type": {
                "b": "bipolar",
                "u": "unipolar",
                "q": "quadrupole",
                "s": "steerer",
                "m": "bipolar milivolt"
            }.get
        }

        return {
            name: idparsers[name](value)
            for name, value in zip(idparsers.keys(), groups)
        }

    def get_idn(self) -> Dict[str, Optional[str]]:

        return {
            "vendor": "Stahl",
            "model": self.model,
            "serial": self.serial_number,
            "firmware": None
        }

    @property
    def identifier(self):
        return f"{self.model}{self.serial_number}"
