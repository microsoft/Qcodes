import logging
import re
from functools import partial
from time import sleep, time

from qcodes import validators as vals
from qcodes import Instrument, InstrumentChannel, VisaInstrument

log = logging.getLogger(__name__)


class ANC300WrongAxisId(Exception):
    pass


class ANC300WrongAxisType(Exception):
    pass


class ANC300GenericError(Exception):
    pass


class ANMxx0(InstrumentChannel):
    """
    QCoDeS driver for an Attocube ANMxx0 module.

    To be used as channels of an Attocube ANC300 piezo controller.

    TODO(ThibaudRuelle): separate channel and instrument to be able to
        reference ANC300 in ANMxx0 init
    """
    def __init__(self, parent: Instrument, name, aid):
        """
        Args:
            parent: The ANC300 instance to which the channel is to be attached
            name: The name of the channel
            aid: The axis id (slot number) of the module
        """
        super().__init__(parent, name)
        self.aid = aid

        serial_no = self.ask('getser {:d}'.format(aid))
        self.model = serial_no.split('-')[0][:-1]

        if self.model in ('ANM300', 'NULL'):
            filter_mapping = {16: '16', 160: '160', 'off': 'off'}
        elif self.model == 'ANM200':
            filter_mapping = {1.6: "1.6", 16: '16', 160: '160',
                              1600: '1600', 'off': 'off'}
        elif self.model == 'ANM150':
            filter_mapping = None
        else:
            raise ValueError('Module model {!s} '
                             'is not supported.'.format(self.model))

        # TODO(ThibaudRuelle) add soft limits, initial values if adequate

        # TODO(Thibaud Ruelle) ans_parser = self._parent.ans_parser
        # leads to error in add_parameter at instantiation time

        # TODO(Thibaud Ruelle) use set_cmd = False instead of vals= ?

        def ans_parser(name, ans, unit=None, parser=str):
            """
            Parse "{name} = {value} ({unit})" type answers from ANC300.

            Args:
                name: The expected name
                ans: The answer from the instrument
                unit: The expected unit(s). String of list of strings. Defaults to None.
                parser: Function to use to parse the value.

            Returns parser(value).
            """
            ans = ans.strip().replace('=', ' ')
            ansparts = ans.split()
            ans_name, ans_val = ansparts[:2]

            if type(unit) == str or unit is None:
                unit = [unit,]

            if ans_val == '?':
                return None

            try:
                ans_unit = ansparts[2]
            except IndexError:
                ans_unit = None

            if ans_name != name:
                raise ValueError('Expected value name {!r}, '
                                 'received {!r}.'.format(name, ans_name))
            if not ans_unit in unit:
                raise ValueError('Expected value unit {!r}, '
                                 'received {!r}.'.format(unit, ans_unit))

            return parser(ans_val)

        # general parameters

        self.add_parameter('serial_no',
                           get_cmd='getser {}'.format(self.aid),
                           vals=vals.Strings())  # unnecessary when #651 in pip

        self.add_parameter('mode',
                           get_cmd='getm {}'.format(self.aid),
                           get_parser=partial(ans_parser, 'mode'),
                           set_cmd='setm {} {{}}'.format(self.aid),
                           label='Axis mode',
                           vals=vals.Enum('gnd', 'inp', 'cap',
                                          'stp', 'off', 'stp+', 'stp-'))

        self.add_parameter('output_voltage',
                           get_cmd='geto {}'.format(self.aid),
                           get_parser=partial(ans_parser, 'voltage',
                                              unit='V', parser=float),
                           label='Output voltage',
                           unit='V')

        self.add_parameter('capacitance',
                           get_cmd='getc {}'.format(self.aid),
                           get_parser=partial(ans_parser, 'capacitance',
                                              unit='nF', parser=float),
                           label='Capacitance',
                           unit='nF')

        self.add_function('update_capacitance',
                          call_cmd='setm {} cap'.format(self.aid))

        self.add_function('wait_capacitance_updated',
                          call_cmd='stop {}'.format(self.aid))

        # scanning parameters

        if self.model in ('ANM300', 'ANM200', 'NULL'):
            self.add_parameter('offset_voltage',
                            get_cmd='geta {}'.format(self.aid),
                            get_parser=partial(ans_parser, 'voltage',
                                                unit='V', parser=float),
                            set_cmd='seta {} {{:.6f}}'.format(self.aid),
                            label='Offset voltage',
                            unit='V',
                            vals=vals.Numbers(0, 150))

            self.add_parameter('ac_in',
                            get_cmd='getaci {}'.format(self.aid),
                            get_parser=partial(ans_parser, 'acin'),
                            set_cmd='setaci {} {{!s}}'.format(self.aid),
                            label='AC input status',
                            val_mapping={True: 'on', False: 'off'})

            self.add_parameter('dc_in',
                            get_cmd='getdci {}'.format(self.aid),
                            get_parser=partial(ans_parser, 'dcin'),
                            set_cmd='setdci {} {{!s}}'.format(self.aid),
                            label='DC input status',
                            val_mapping={True: 'on', False: 'off'})

            if filter_mapping:
                self.add_parameter('outp_filter',
                                get_cmd='getfil {}'.format(self.aid),
                                get_parser=partial(ans_parser, 'filter'),
                                set_cmd='setfil {} {{!s}}'.format(self.aid),
                                label='Output low-pass filter',
                                unit='Hz',
                                val_mapping=filter_mapping)

        # stepping parameters

        if self.model in ('ANM300', 'ANM150', 'NULL'):
            self.add_parameter('step_frequency',
                            get_cmd='getf {}'.format(self.aid),
                            get_parser=partial(ans_parser, 'frequency',
                                                unit=['Hz','H'], parser=int),
                            set_cmd='setf {} {{:.0f}}'.format(self.aid),
                            label='Stepping frequency',
                            unit='Hz',
                            vals=vals.Ints(1, 10000))

            self.add_parameter('step_amplitude',
                            get_cmd='getv {}'.format(self.aid),
                            get_parser=partial(ans_parser, 'voltage',
                                                unit='V', parser=float),
                            set_cmd='setv {} {{:.6f}}'.format(self.aid),
                            label='Stepping amplitude',
                            unit='V',
                            vals=vals.Numbers(0, 150))

            # TODO(Thibaud Ruelle): define acceptable vals properly for patterns
            self.add_parameter('step_up_pattern',
                            get_cmd='getpu {}'.format(self.aid),
                            get_parser=lambda s: [int(u) for u in
                                                    s.split('\r\n')],
                            set_cmd='setpu {} {{!s}}'.format(self.aid),
                            set_parser=lambda l: " ".join([str(u) for u in l]),
                            vals=vals.Lists())  # unnecessary when #651 in pip

            self.add_parameter('step_down_pattern',
                            get_cmd='getpd {}'.format(self.aid),
                            get_parser=lambda s: [int(u) for u in
                                                    s.split('\r\n')],
                            set_cmd='setpd {} {{!s}}'.format(self.aid),
                            set_parser=lambda l: " ".join([str(u) for u in l]),
                            vals=vals.Lists())  # unnecessary when #651 in pip

            self.add_function('step_up_single',
                            call_cmd='stepu {}'.format(self.aid))

            self.add_function('step_up_cont',
                            call_cmd='stepu {} c'.format(self.aid))

            self.add_function('step_up',
                            call_cmd='stepu {} {{:d}}'.format(self.aid),
                            args=[vals.Ints(min_value=1)])

            self.add_function('step_down_single',
                            call_cmd='stepd {}'.format(self.aid))

            self.add_function('step_down_cont',
                            call_cmd='stepu {} c'.format(self.aid))

            self.add_function('step_down',
                            call_cmd='stepd {} {{:d}}'.format(self.aid),
                            args=[vals.Ints(min_value=1)])

            self.add_function('stop',
                            call_cmd='stop {}'.format(self.aid))

            self.add_function('wait_steps_end',
                            call_cmd='stepw {}'.format(self.aid))

            # TODO(Thibaud Ruelle): test for range before adding param
            self.add_parameter('trigger_up_pin',
                            get_cmd='gettu {}'.format(self.aid),
                            get_parser=partial(ans_parser, 'trigger'),
                            set_cmd='settu {} {{!s}}'.format(self.aid),
                            label='Input trigger up pin',
                            val_mapping={i: str(i) for i in
                                            ['off', *range(1, 15)]})

            self.add_parameter('trigger_down_pin',
                            get_cmd='gettd {}'.format(self.aid),
                            get_parser=partial(ans_parser, 'trigger'),
                            set_cmd='settd {} {{!s}}'.format(self.aid),
                            label='Input trigger down pin',
                            val_mapping={i: str(i) for i in
                                            ['off', *range(1, 15)]})


class ANC300(VisaInstrument):
    """
    QCodeS driver for Attocube ANC300 piezo controller
    """

    def __init__(self, name, address=None, password='123456', axis_names=None,
                 **kwargs):
        """
        Args:
            name: The name of the instrument
            address: The VISA resource name of the instrument
                (e.g. "tcpip0::192.168.1.2::7230::socket")
            password: Password for authenticating into the instrument
                (default: '123456')
            axis_names(optional): List of names to give to the individual
                channels
        """
        super().__init__(name, address, terminator='\r\n', **kwargs)

        self.visa_handle.encoding = 'ascii'

        # Wait for terminal to fire up and clear welcome message
        sleep(0.1)
        self.visa_handle.clear()

        # Authenticate
        self.visa_handle.write(password)

        self.visa_handle.read_raw()  # Flush console echo
        resp = self.visa_handle.read()  # Read password prompt response
        if resp == "Authorization success":
            sleep(0.1)
            self.visa_handle.clear()
        else:
            raise ConnectionRefusedError("Authentication failed")

        # Add available modules as channels to the instrument
        # Maximum aid depends on model
        aid = 1
        while True:
            try:
                # Attempt communication with module
                self.ask('getser {:d}'.format(aid))
            except ANC300WrongAxisType:
                # Slot is empty
                pass
            except ANC300WrongAxisId:
                # Slot does not exist
                break
            else:
                # If module is present, create channel and add to instrument
                axis_name = (axis_names[aid-1] if axis_names
                             else 'axis{}'.format(aid))
                module = ANMxx0(self, axis_name, aid)
                self.add_submodule(axis_name, module)
            finally:
                aid += 1

        # TODO(ThibaudRuelle): add output triggers as channels

        self.add_parameter('serial_no',
                           get_cmd='getcser',
                           vals=vals.Strings())  # unnecessary when #651 in pip

        self.add_parameter('instrument_info',
                           get_cmd='ver',
                           vals=vals.Strings())  # unnecessary when #651 in pip

        self.connect_message()

    def write_raw(self, cmd):
        """
        Override the low-level interface to ``visa_handle.write``.

        Args:
            cmd (str): The command to send to the instrument.
        """
        # the simulation backend does not return anything on
        # write
        log.debug("Writing to instrument %s: %r", self.name, cmd)
        if self.visabackend == 'sim':
            # if we use pyvisa-sim, we must read back the 'OK'
            # response of the setting
            resp = self.visa_handle.ask(cmd)
            if resp != 'OK':
                log.warning('Received non-OK response from instrument '
                            '%s: %r.', self.name, resp)
        else:
            self._ask_raw(cmd)

    def ask_raw(self, cmd):
        """
        Overriding the low-level interface to ``visa_handle.ask``.

        Args:
            cmd (str): The command to send to the instrument.

        Returns:
            str: The instrument's response.
        """
        log.debug("Querying instrument %s: %r", self.name, cmd)

        return self._ask_raw(cmd)

    def _ask_raw(self, cmd):
        """
        Base low-level interface to VISA instrument.

        Args:
            cmd (str): The command to send to the instrument.

        Returns:
            str: The instrument's response.
        """

        _, ret_code = self.visa_handle.write(cmd)
        self.check_error(ret_code)

        # regexp necessary to deal with unreliable message ends
        end_msg_pattern = re.compile(r'(\r\n)?(OK|ERROR)$')

        line = ''
        resp = []
        end_msg_bool = False
        while not end_msg_bool:
            line = self.visa_handle.read().strip()
            resp.append(line)

            match = end_msg_pattern.search(line)
            if match is not None:
                end_msg_bool = True
                err_check = match.group(2)

        log.debug("Reading fron instrument %s: %r",
                  self.name, '\r\n'.join(resp))

        if resp[0].strip("> ").strip() == cmd:  # pop console echo if present
            resp.pop(0)
        answer = '\r\n'.join(resp)
        answer = end_msg_pattern.sub('', answer)  # remove end_msg_pattern

        if err_check == 'ERROR':
            if answer.strip() == 'Wrong axis id':
                raise ANC300WrongAxisId
            elif answer.strip() == 'Wrong axis type':
                raise ANC300WrongAxisType
            else:
                raise ANC300GenericError(answer.strip())

        return answer

    def get_idn(self):
        ver = self.ask("ver").split('\r\n')[0].split(" ")

        vendor = ver[0]
        model = " ".join(ver[1:2])
        serial = self.ask("getcser")
        firmware = " ".join(ver[4:7])
        return {'vendor': vendor, 'model': model,
                'serial': serial, 'firmware': firmware}

    def connect_message(self, idn_param: str = 'IDN',
                        begin_time: float = None) -> None:
        """
        Print a standard message on initial connection to an instrument.

        Args:
            idn_param: name of parameter that returns ID dict.
                Default 'IDN'.
            begin_time: time.time() when init started.
                Default is self._t0, set at start of Instrument.__init__.
        """
        # start with an empty dict, just in case an instrument doesn't
        # heed our request to return all 4 fields.
        idn = {'vendor': None, 'model': None,
               'serial': None, 'firmware': None}
        idn.update(self.get(idn_param))
        con_time = time() - (begin_time or self._t0)

        con_msg = ('Connected to: {vendor} {model} '
                   '(serial:{serial}, firmware:{firmware}) '
                   'in {t:.2f}s'.format(t=con_time, **idn))
        log.info(con_msg)

        axes = []
        for axis in self.submodules.values():
            axes.append('    - {} (slot {}): {} '
                        '(serial: {})'
                        ''.format(axis.short_name, axis.aid,
                                  axis.model, axis.serial_no))

        axes_msg = ('Available axes:\n' +
                    '\n'.join(axes))
        log.info(axes_msg)
