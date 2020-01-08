import itertools
from typing import List, Dict, Optional, Union, Tuple

import qcodes.utils.validators as vals
from qcodes import VisaInstrument
from qcodes.utils.helpers import create_on_off_val_mapping


class UnknownOrEmptySlot(Exception):
    pass


class InvalidValue(Exception):
    pass


class Keithley_3706A(VisaInstrument):
    """
    This is the QCoDeS instrument driver for the Keithley 3706A-SNFP
    System Switch.
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA resource address
        """
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter('reset_channel',
                           get_cmd=None,
                           set_cmd=self._reset_channel,
                           docstring='Resets the specified channels to '
                                     'factory default settings.')

        self.add_parameter('open_channel',
                           get_cmd=None,
                           set_cmd=self._open_channel,
                           docstring='Opens the specified channels and '
                                     'backplane relays.')

        self.add_parameter('close_channel',
                           get_cmd=None,
                           set_cmd=self._close_channel,
                           docstring='Closes the channels and '
                                     'backplane relays.',
                           vals=vals.Strings())

        self.add_parameter('exclusive_close',
                           get_cmd=None,
                           set_cmd=self._set_exclusive_close,
                           docstring='Closes the specified channels such that '
                                     'any presently closed channels opens if '
                                     'they are not in the specified by the '
                                     'parameter.',
                           vals=vals.Strings())

        self.add_parameter('exclusive_slot_close',
                           get_cmd=None,
                           set_cmd=self._set_exclusive_slot_close,
                           docstring='Closes the specified channels on the '
                                     'associated slots abd opens any other '
                                     'channels if they are not specified by '
                                     'the parameter.',
                           vals=vals.Strings())

        self.add_parameter('channel_connect_rule',
                           get_cmd=self._get_channel_connect_rule,
                           set_cmd=self._set_channel_connect_rule,
                           docstring='Controls the connection rule for closing '
                                     'and opening channels when using '
                                     '`exclusive_close` and '
                                     '`exclusive_slot_close` parameters. '
                                     'If it is set to break before make, '
                                     'it is ensured that all channels open '
                                     'before any channels close. If it is set '
                                     'to make before break, it is ensured that '
                                     'all channels close before any channels '
                                     'open. If it is off, channels open and '
                                     'close simultaneously.',
                           vals=vals.Enum('BREAK_BEFORE_MAKE',
                                          'MAKE_BEFORE_BREAK',
                                          'OFF'))

        self.add_parameter('gpib_enable',
                           get_cmd=self._get_gpib_status,
                           set_cmd=self._set_gpib_status,
                           docstring='Enables or disables GPIB connection.',
                           val_mapping=create_on_off_val_mapping(on_val='true',
                                                                 off_val='false'
                                                                 ))

        self.add_parameter('gpib_address',
                           get_cmd=self._get_gpib_address,
                           get_parser=int,
                           set_cmd=self._set_gpib_address,
                           docstring='Sets and gets the GPIB address.',
                           vals=vals.Ints(1, 30))

        self.add_parameter('lan_enable',
                           get_cmd=self._get_lan_status,
                           set_cmd=self._set_lan_status,
                           docstring='Enables or disables LAN connection.',
                           val_mapping=create_on_off_val_mapping(on_val='true',
                                                                 off_val='false'
                                                                 ))

        self.connect_message()

    def _reset_channel(self, val: str) -> None:
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, backplane relays or "allslots".')
        self.write(f"channel.reset('{val}')")

    def _open_channel(self, val: str) -> None:
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, or "allslots".')
        self.write(f"channel.open('{val}')")

    def _close_channel(self, val: str) -> None:
        slots = ['allslots', *self._get_slot_id()]
        if val in slots:
            raise InvalidValue("Slots cannot be closed all together.")
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels or channel '
                               'ranges and associated backplane relays.')
        self.write(f"channel.close('{val}')")

    def _set_exclusive_close(self, val: str) -> None:
        slots = ['allslots', *self._get_slot_id()]
        if val in slots:
            raise InvalidValue("Slots cannot be exclusively closed.")
        if val == "":
            raise InvalidValue('An empty string may cause all channels and '
                               'associated backplane relays to open. Use '
                               '"open_channel" parameter instead.')
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels or channel '
                               'ranges and associated backplane relays.')
        self.write(f"channel.exclusiveclose('{val}')")

    def _set_exclusive_slot_close(self, val: str) -> None:
        slots = ['allslots', *self._get_slot_id()]
        if val in slots:
            raise InvalidValue("Slots cannot be exclusively closed.")
        if val == "":
            raise InvalidValue('Argument cannot be an empty string.')
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels or channel '
                               'ranges and associated backplane relays.')
        self.write(f"channel.exclusiveslotclose('{val}')")

    def _get_channel_connect_rule(self) -> str:
        connect_rule = {1: 'BREAK_BEFORE_MAKE',
                        2: 'MAKE_BEFORE_BREAK',
                        0: 'OFF'}
        rule = self.ask('channel.connectrule')
        return connect_rule[int(float(rule))]

    def _set_channel_connect_rule(self, val: str) -> None:
        self.write(f'channel.connectrule = channel.{val}')

    def _get_gpib_status(self) -> str:
        return self.ask('comm.gpib.enable')

    def _set_gpib_status(self, val: Union[str, bool]) -> None:
        self.write(f'comm.gpib.enable = {val}')

    def _get_lan_status(self) -> str:
        return self.ask('comm.lan.enable')

    def _set_lan_status(self, val: Union[str, bool]) -> None:
        self.write(f'comm.lan.enable = {val}')

    def _get_gpib_address(self) -> int:
        return int(float(self.ask('gpib.address')))

    def _set_gpib_address(self, val: int) -> None:
        self.write(f'gpib.address = {val}')

    def get_closed_channels(self, val: str) -> str:
        """
        Queries for the closed channels.

        Args:
            val: A string representing the channels,
                backplane relays or channel patterns to be queried.
        """
        if val == "":
            raise InvalidValue('Argument cannot be an empty string.')
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, backplane relays or "allslots".')
        return self.ask(f"channel.getclose('{val}')")

    def set_forbidden_channels(self, val: str) -> None:
        """
        Prevents the closing of specified channels and backplane
        relays.

        Args:
            val: A string representing channels and backplane relays
                to make forbidden to close.
        """
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, backplane relays or "allslots".')
        self.write(f"channel.setforbidden('{val}')")

    def get_forbidden_channels(self, val: str) -> str:
        """
        Returns a string that lists the channels and backplane relays
        that are forbidden to close.

        Args:
            val: A string representing the channels,
                backplane relays or channel patterns to be queried to see
                if they are forbidden to close.
        """
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, backplane relays or "allslots".')
        return self.ask(f"channel.getforbidden('{val}')")

    def clear_forbidden_channels(self, val: str) -> None:
        """
        Clears the list of channels that are forbidden to close.

        Args:
            val: A string representing the channels that will no longer
                be listed as forbidden to close.
        """
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, backplane relays or "allslots".')
        self.write(f"channel.clearforbidden('{val}')")

    def set_delay(self, val: str, delay_time: float) -> None:
        """
        Sets an additional delay time for the specified channels.

        Args:
            val: A string representing the channels for which there will
                be an additional delay time.
            delay_time: Delay time for the specified channels in seconds.
        """
        backplanes = self.get_analog_backplane_specifiers()
        specifiers = val.split(',')
        for element in specifiers:
            if element in backplanes:
                raise InvalidValue("Additional delay times cannot be set for "
                                   "analog backplane relays.")
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, or "allslots".')
        self.write(f"channel.setdelay('{val}', {delay_time})")

    def get_delay(self, val: str) -> float:
        """
        Queries for the delay times.

        Args:
            val: A string representing the channels to query for
                additional delay times.
        """
        backplanes = self.get_analog_backplane_specifiers()
        specifiers = val.split(',')
        for element in specifiers:
            if element in backplanes:
                raise InvalidValue("Additional delay times cannot be set for "
                                   "analog backplane relays.")
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, or "allslots".')
        return float(self.ask(f"channel.getdelay('{val}')"))

    def set_backplane(self, val: str, backplane: str) -> None:
        """
        Sets the analog backplane relays to use with given channels
        when they are used in switching applications.

        Args:
            val: A string representing the list of channels to change.
            backplane: A string representing the list of analog backplane
                relays to set for the channels specified.
        """
        backplanes = self.get_analog_backplane_specifiers()
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, or "allslots".')
        specifiers = backplane.split(',')
        for element in specifiers:
            if element not in backplanes:
                raise InvalidValue(f'{val} is not a valid specifier. '
                                   'The specifier should be analog '
                                   'backplane relay.')
        self.write(f"channel.setbackplane('{val}', '{backplane}')")

    def get_backplane(self, val: str) -> str:
        """
        Lists the backplane relays that are controlled in switching
        applications for specified channels.

        Args:
            val: A string representing the channels being queried.
        """
        backplanes = self.get_analog_backplane_specifiers()
        specifiers = val.split(',')
        for element in specifiers:
            if element in backplanes:
                raise InvalidValue(f'{val} cannot be a analog backplane relay.')
        if not self._validator(val):
            raise InvalidValue(f'{val} is not a valid specifier. '
                               'The specifier should be channels, channel '
                               'ranges, slots, or "allslots".')
        return self.ask(f"channel.getbackplane('{val}')")

    def _get_slot_id(self) -> List[str]:
        """
        Returns the slot ids of the installed cards.
        """
        cards = self.get_switch_cards()
        slot_id = []
        for _, item in enumerate(cards):
            slot_id.append('{slot_no}'.format(**item))
        return slot_id

    def _get_number_of_rows(self) -> List[int]:
        """
        Returns the total number of rows of the installed cards.
        """
        slot_id = self._get_slot_id()
        total_number_of_rows = [int(float(self.ask(
            f'slot[{int(i)}].rows.matrix'))) for i in slot_id]
        return total_number_of_rows

    def _get_number_of_columns(self) -> List[int]:
        """
        Returns the total number of columns of the installed cards.
        """
        slot_id = self._get_slot_id()
        total_number_of_columns = [int(float(self.ask(
            f'slot[{int(i)}].columns.matrix'))) for i in slot_id]
        return total_number_of_columns

    def _get_rows(self) -> List[List[str]]:
        """
        Returns the elements of each row.
        """
        total_number_of_rows = self._get_number_of_rows()
        row_list = []
        for _, item in enumerate(total_number_of_rows):
            rows_in_each_slot = [str(i) for i in range(1, item + 1)]
            row_list.append(rows_in_each_slot)
        return row_list

    def _get_columns(self) -> List[List[str]]:
        """
        Returns the elements of each column.
        """
        total_number_of_columns = self._get_number_of_columns()
        column_list = []
        for _, item in enumerate(total_number_of_columns):
            columns_in_each_slot = []
            for i in range(1, item + 1):
                if i < 10:
                    columns_in_each_slot.append('0' + str(i))
                else:
                    columns_in_each_slot.append(str(i))
            column_list.append(columns_in_each_slot)
        return column_list

    def _get_channel_ranges(self) -> List[str]:
        """
        A helper function that gets two channel names from the available
        channels list and join them via a colon to define a channel range.
        """
        range_list = []
        for i in self._get_slot_id():
            channel = self.get_channels_by_slot(int(i))
            for element in itertools.combinations(channel, 2):
                range_list.append(':'.join(element))
        return range_list

    def get_channels(self) -> List[str]:
        """
        This function returns the name of the matrix channels.
        User can call this function to see the names of the available
        channels, in case he/she is not familiar with the naming convention.
        However, note that, this is a standalone helper function and
        the usage of channel attributes of the instrument driver does
        not depend on the functionality of this method.
        """
        slot_id = self._get_slot_id()
        row_list = self._get_rows()
        column_list = self._get_columns()
        matrix_channels = []
        for i, slot in enumerate(slot_id):
            for element in itertools.product(slot, row_list[i], column_list[i]):
                matrix_channels.append(''.join(element))
        return matrix_channels

    def get_channels_by_slot(self, val: int) -> List[str]:
        """
        Returns the channel names of a given slot.

        Args:
            val: An integer value specifying the slot number.
        """
        slot_id = self._get_slot_id()
        if str(val) not in slot_id:
            raise UnknownOrEmptySlot("Please provide a valid slot identifier. "
                                     f'Available slots are {slot_id}.')
        row_list = self._get_rows()
        column_list = self._get_columns()
        matrix_channels_by_slot = []
        for element in itertools.product(str(val), row_list[0], column_list[0]):
            matrix_channels_by_slot.append(''.join(element))
        return matrix_channels_by_slot

    def get_analog_backplane_specifiers(self) -> List[str]:
        """
        Returns a list of comma separated strings representing available analog
        backplane relays. This function should not be mixed with the
        `get_backplane` method. The latter returns backplane relays which are
        associated with a channel by using `set_backplane` method.
        """
        backplane_common_number = '9'
        backplane_relay_common_numbers = ['11', '12', '13', '14', '15', '16']
        slot_id = self._get_slot_id()
        analog_backplane_relays = []
        for element in itertools.product(slot_id, backplane_common_number,
                                         backplane_relay_common_numbers):
            analog_backplane_relays.append(''.join(element))
        return analog_backplane_relays

    def get_idn(self) -> Dict[str, Optional[str]]:
        """
        Overwrites the generic QCoDeS get IDN method. Returns
        a dictionary including the vendor, model, serial number and
        firmware version of the instrument.
        """
        idnstr = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, idnstr.split(','))
        model = model[6:]

        idn: Dict[str, Optional[str]] = {'vendor': vendor, 'model': model,
                                         'serial': serial, 'firmware': firmware}
        return idn

    def get_switch_cards(self) -> Tuple[Dict[str, str], ...]:
        """
        Returns a list of dictionaries listing the properties of the installed
        switch cards including the slot number tha it is installed, model,
        firmware version and serial number.
        """
        switch_cards: List[Dict[str, str]] = []
        for i in range(1, 7):
            scard = self.ask(f'slot[{i}].idn')
            if scard != 'Empty Slot':
                model, mtype, firmware, serial = map(str.strip,
                                                     scard.split(','))
                sdict = {'slot_no': str(i), 'model': model, 'mtype': mtype,
                         'firmware': firmware, 'serial': serial}
                switch_cards.append(sdict)
        return tuple(switch_cards)

    def get_available_memory(self) -> Dict[str, Optional[str]]:
        """
        Returns the amount of memory that is currently available for
        storing scripts, configurations and channel patterns.
        """
        memstring = self.ask('memory.available()')
        system_memory, script_memory, \
            pattern_memory, config_memory = map(str.strip, memstring.split(','))

        memory_available: Dict[str, Optional[str]] = {
            'System Memory  (%)': system_memory,
            'Script Memory  (%)': script_memory,
            'Pattern Memory (%)': pattern_memory,
            'Config Memory  (%)': config_memory
        }
        return memory_available

    def get_ip_address(self) -> str:
        """
        Returns the current IP address of the instrument.
        """
        return self.ask('lan.status.ipaddress')

    def reset_local_network(self) -> None:
        """
        Resets the local network (LAN).
        """
        self.write('lan.reset()')

    def save_setup(self, val: Optional[str] = None) -> None:
        """
        Saves the present setup.

        Args:
            val: An optional string representing the path and the file name
                to which the setup shall be saved on a USB flash drive. If not
                provided, the setup will be saved to the nonvolatile memory
                of the instrument. Any previous saves shall be overwritten.
        """
        if val is not None:
            self.write(f"setup.save('{val}')")
        else:
            self.write(f'setup.save()')

    def load_setup(self, val: Union[int, str]) -> None:
        """
        Loads the settings from a saved setup.

        Args:
            val: An integer or a string that specifies the location of saved
                setup. If it is `0`, factory defaults load. If it is `1`,
                the saved setup from the nonvolatile memory is recalled.
                Otherwise, a string specifying the relative path to the saved
                setup on a USB drive should be passed in.
        """
        self.write(f"setup.recall('{val}')")

    def _validator(self, val: str) -> bool:
        """
        Instrument specific validator. As the number of validation points
        are around 15k, to avoid QCoDeS parameter validation to print them all,
        we shall raise a custom exception.
        """
        ch = self.get_channels()
        ch_range = self._get_channel_ranges()
        slots = ['allslots', *self._get_slot_id()]
        backplanes = self.get_analog_backplane_specifiers()
        specifier = val.split(',')
        for element in specifier:
            if element not in (*ch, *ch_range, *slots, *backplanes):
                return False
        return True

    def connect_message(self, idn_param: str = 'IDN',
                        begin_time: float = None) -> None:
        """
        Overwrites the generic QCoDeS instrument connect message.
        Here, additionally, we provide information about
        which slots of the system switch is occupied with what
        kind of matrix, as well.
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
