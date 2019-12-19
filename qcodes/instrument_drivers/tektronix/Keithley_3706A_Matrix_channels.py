import logging
import struct
import numpy as np
import warnings
import time
from typing import List, Dict, Optional

import qcodes as qc
from qcodes import VisaInstrument, DataSet
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.base import Instrument, Parameter
from qcodes.instrument.parameter import ArrayParameter, ParameterWithSetpoints
import qcodes.utils.validators as vals
from qcodes.utils.helpers import create_on_off_val_mapping


class KeithleyMatrixChannel(InstrumentChannel):
    """
    """

    def __init__(self, parent: Instrument, name: str, channel: str) -> None:
        """
        """
        super().__init__(parent, name)


class Keithley_3706A(VisaInstrument):
    """
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA resource address
        """
        super().__init__(name, address, terminator='\n', **kwargs)

        self.connect_message()

        self.add_parameter('gpib_enable',
                           get_cmd=self._get_gpib_status,
                           set_cmd=self._set_gpib_status,
                           vals=vals.Enum('true', 'false'))

        self.add_parameter('lan_enable',
                           get_cmd=self._get_lan_status,
                           set_cmd=self._set_lan_status,
                           vals=vals.Enum('true', 'false'))

    def _get_gpib_status(self) -> str:
        return self.ask('comm.gpib.enable')

    def _set_gpib_status(self, val: str) -> None:
        self.write(f'comm.gpib.enable = {val}')

    def _get_lan_status(self) -> str:
        return self.ask('comm.lan.enable')

    def _set_lan_status(self, val: str) -> None:
        self.write(f'comm.lan.enable = {val}')

    def get_idn(self) -> Dict[str, Optional[str]]:
        idnstr = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, idnstr.split(','))
        model = model[6:]

        idn: Dict[str, Optional[str]] = {'vendor': vendor, 'model': model,
                                         'serial': serial, 'firmware': firmware}
        return idn

    def get_switch_cards(self) -> List[Dict[str, Optional[str]]]:
        switch_cards: List[Dict[str, Optional[str]]] = []
        for i in range(1, 7):
            scard = self.ask(f'slot[{i}].idn')
            if scard != 'Empty Slot':
                model, mtype, firmware, serial = map(str.strip,
                                                     scard.split(','))
                sdict = {'slot_no': i, 'model': model, 'mtype': mtype,
                         'firmware': firmware, 'serial': serial}
                switch_cards.append(sdict)
        return switch_cards

    def get_available_memory(self) -> Dict[str, Optional[str]]:
        memstring = self.ask('memory.available()')
        systemMemory, scriptMemory, \
            patternMemory, configMemory = map(str.strip, memstring.split(','))

        memory_available: Dict[str, Optional[str]] = {
            'System Memory  (%)': systemMemory,
            'Script Memory  (%)': scriptMemory,
            'Pattern Memory (%)': patternMemory,
            'Config Memory  (%)': configMemory
        }
        return memory_available

    def get_ip_address(self) -> str:
        return self.ask('lan.status.ipaddress')

    def reset_local_network(self) -> None:
        self.write('lan.reset()')

    def connect_message(self) -> None:
        """
        """
        idn = self.get_idn()
        cards = self.get_switch_cards()

        con_msg = ('Connected to: {vendor} {model} SYSTEM SWITCH '
                   '(serial:{serial}, firmware:{firmware})'.format(**idn))
        print(con_msg)
        self.log.info(f"Connected to instrument: {idn}")

        for _, item in enumerate(cards):
            card_info = ('Slot {slot_no}- Model:{model}, Matrix Type:{mtype}, '
                         'Firmware:{firmware}, Serial:{serial}'.format(**item))
            print(card_info)
            self.log.info(f'Switch Cards: {item}')

    def ask(self, cmd: str) -> str:
        """
        Override of normal ask. This is important, since queries to the
        instrument must be wrapped in 'print()'
        """
        return super().ask('print({:s})'.format(cmd))
