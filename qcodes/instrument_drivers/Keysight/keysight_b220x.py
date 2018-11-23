import re
import warnings
from functools import wraps

from qcodes import VisaInstrument
from qcodes.utils.validators import MultiType, Ints, Enum, Lists


def post_execution_status_poll(func):
    """
    Generates a decorator that clears the instrument's status registers
    before executing the actual call and reads the status register after the
    function call to determine whether an error occured.

    :param func: function to wrap
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.clear_status()
        retval = func(self, *args, **kwargs)

        stb = self.get_status()
        if stb:
            warnings.warn(f"Instrument status byte indicates an error occured "
                          f"(value of STB was: {stb})! Use `get_error` method "
                          f"to poll error message.",
                          stacklevel=2)
        return retval

    return wrapper


class KeysightB220X(VisaInstrument):
    """
    QCodes driver for B2200 / B2201 switch matrix

    Note: The B2200 consists of up to 4 modules and provides two channel
    configuration modes, *Normal* and
    *Auto*. The configuration mode defines whether multiple switch modules
    are treated as one (*Auto* mode), or separately (*Normal* mode). This
    driver only implements the *Auto* mode. Please read the manual section on
    *Channel Configuration Mode* for more info.
    """

    _available_input_ports = Ints(1, 14)
    _available_output_ports = Ints(1, 48)

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self._card = 0

        self.add_parameter(name='get_status',
                           get_cmd='*ESR?',
                           get_parser=int,
                           docstring='Queries status register.')

        self.add_parameter(name='get_error',
                           get_cmd=':SYST:ERR?',
                           docstring='Queries error queue')

        self.add_parameter(name='connections',
                           get_cmd=f':CLOS:CARD? {self._card}',
                           get_parser=KeysightB220X.parse_channel_list,
                           docstring='queries currently active connections '
                                     'and returns a set of tuples {(input, '
                                     'output), ...}'
                           )

        self.add_parameter(name='connection_rule',
                           get_cmd=self._get_connection_rule,
                           set_cmd=self._set_connection_rule,
                           val_mapping={'free': 'FREE',
                                        'single': 'SROU'},
                           docstring=("specifies connection rule. Parameter "
                                      "one of 'free' (default) or 'single'.\n\n"
                                      "In 'free' mode\n"
                                      " - each input port can be connected to "
                                      "multiple output ports\n"
                                      " - and each output port can be "
                                      "connected to multiple input ports.\n"
                                      " - Caution: If the Free connection rule "
                                      "has been specified, ensure multiple "
                                      "input ports are not connected to the "
                                      "same output port. Such configurations "
                                      "can cause damage\n\n"
                                      "In single route mode:\n"
                                      " - each input port can be connected to "
                                      "only one output port\n"
                                      " - and each output port can be "
                                      "connected to only one input port.\n"
                                      " - existing connection to a port will "
                                      "be disconnected when a new connection "
                                      "is made.\n"
                                      )
                           )

        self.add_parameter(name='connection_sequence',
                           get_cmd=f':CONN:SEQ? {self._card}',
                           set_cmd=f':CONN:SEQ {self._card},{{}}',
                           val_mapping={'none': 'NSEQ',
                                        'bbm': 'BBM',
                                        'mbb': 'MBBR'},
                           docstring="One of 'none', 'bbm' (Break before "
                                     "make) or 'mbb' (make before break)"
                           )

        self.add_parameter(name='bias_input_port',
                           get_cmd=f':BIAS:PORT? {self._card}',
                           set_cmd=f':BIAS:PORT {self._card},{{}}',
                           vals=MultiType(KeysightB220X._available_input_ports,
                                          Enum(-1)),
                           get_parser=int,
                           docstring="Selects the input that will be used as "
                                     "bias input port (default 10). The Bias "
                                     "input port cannot be used on subsequent "
                                     "`connect` or `disconnect` commands if "
                                     "Bias mode is ON"
                           )

        self.add_parameter(name='bias_mode',
                           get_cmd=f':BIAS? {self._card}',
                           set_cmd=f':BIAS {self._card},{{}}',
                           val_mapping={True: 1,
                                        False: 0},
                           docstring="Param: True for ON, False for OFF"
                           )

        self.add_parameter(name='gnd_input_port',
                           get_cmd=f':AGND:PORT? {self._card}',
                           set_cmd=f':AGND:PORT {self._card},{{}}',
                           vals=MultiType(KeysightB220X._available_input_ports,
                                          Enum(-1)),
                           get_parser=int,
                           docstring="Selects the input that will be used as "
                                     "GND input port (default 12). The GND "
                                     "input port cannot be used on subsequent "
                                     "`connect` or `disconnect` commands if "
                                     "GND mode is ON"
                           )

        self.add_parameter(name='gnd_mode',
                           get_cmd=f':AGND? {self._card}',
                           set_cmd=f':AGND {self._card},{{}}',
                           val_mapping={True: 1,
                                        False: 0}
                           )

        self.add_parameter(name='unused_inputs',
                           get_cmd=f':AGND:UNUSED? {self._card}',
                           set_cmd=f":AGND:UNUSED {self._card},'{{}}'",
                           get_parser=lambda response: [int(x) for x in
                                                        response.strip(
                                                            "'").split(',') if
                                                        x.strip().isdigit()],
                           set_parser=lambda value: str(value).strip('[]'),
                           vals=Lists(KeysightB220X._available_input_ports)
                           )

        self.add_parameter(name='couple_ports',
                           get_cmd=f':COUP:PORT? {self._card}',
                           set_cmd=f":COUP:PORT {self._card},'{{}}'",
                           set_parser=lambda value: str(value).strip('[]()'),
                           get_parser=lambda response: [int(x) for x in
                                                        response.strip(
                                                            "'").split(',') if
                                                        x.strip().isdigit()],
                           vals=Lists(Enum(1, 3, 5, 7, 9, 11, 13))
                           )

        self.add_parameter(name='couple_mode',
                           get_cmd=f':COUP? {self._card}',
                           set_cmd=f':COUP {self._card},{{}}',
                           val_mapping={True: 1,
                                        False: 0},
                           docstring="Param: True for ON, False for OFF"
                           )

    @post_execution_status_poll
    def connect(self, input_ch, output_ch):
        """Connect given input/output pair.

        :param input_ch: Input channel number 1-14
        :param output_ch: Output channel number 1-48
        """
        KeysightB220X._available_input_ports.validate(input_ch)
        KeysightB220X._available_output_ports.validate(output_ch)

        self.write(f":CLOS (@{self._card:01d}{input_ch:02d}{output_ch:02d})")

    @post_execution_status_poll
    def disconnect(self, input_ch, output_ch):
        """Disconnect given Input/Output pair.

        :param input_ch: Input channel number 1-14
        :param output_ch: Output channel number 1-48
        """
        KeysightB220X._available_input_ports.validate(input_ch)
        KeysightB220X._available_output_ports.validate(output_ch)

        self.write(f":OPEN (@{self._card:01d}{input_ch:02d}{output_ch:02d})")

    @post_execution_status_poll
    def disconnect_all(self):
        """
        opens all connections.

        If ground or bias mode is enabled it will connect all outputs to the
        GND or Bias Port
        """
        self.write(f':OPEN:CARD {self._card}')

    @post_execution_status_poll
    def bias_disable_all_outputs(self):
        """
        Removes all outputs from list of ports that will be connected to GND
        input if port is unused and bias mode is enabled.
        """
        self.write(f':BIAS:CHAN:DIS:CARD {self._card}')

    @post_execution_status_poll
    def bias_enable_all_outputs(self):
        """
        Adds all outputs to list of ports that will be connected to bias input
        if port is unused and bias mode is enabled.
        """
        self.write(f':BIAS:CHAN:ENAB:CARD {self._card}')

    @post_execution_status_poll
    def bias_enable_output(self, output):
        """
        Adds `output` to list of ports that will be connected to bias input
        if port is unused and bias mode is enabled.

        :param output: int 1-48
        """
        KeysightB220X._available_output_ports.validate(output)

        self.write(f':BIAS:CHAN:ENAB (@{self._card}01{output:02d})'
                   )

    @post_execution_status_poll
    def bias_disable_output(self, output):
        """
        Removes `output` from list of ports that will be connected to bias
        input if port is unused and bias mode is enabled.

        :param output: int 1-48
        """
        KeysightB220X._available_output_ports.validate(output)

        self.write(f':BIAS:CHAN:DIS (@{self._card}01{output:02d})')

    @post_execution_status_poll
    def gnd_enable_output(self, output):
        """
        Adds `output` to list of ports that will be connected to GND input
        if port is unused and bias mode is enabled.

        :param output: int 1-48
        """
        KeysightB220X._available_output_ports.validate(output)

        self.write(f':AGND:CHAN:ENAB (@{self._card}01{output:02d})')

    @post_execution_status_poll
    def gnd_disable_output(self, output):
        """
        Removes `output` from list of ports that will be connected to GND
        input if port is unused and bias mode is enabled.

        :param output: int 1-48
        """
        KeysightB220X._available_output_ports.validate(output)

        self.write(f':AGND:CHAN:DIS (@{self._card}01{output:02d})')

    @post_execution_status_poll
    def gnd_enable_all_outputs(self):
        """
        Adds all outputs to list of ports that will be connected to GND input
        if port is unused and bias mode is enabled.
        """
        self.write(f':AGND:CHAN:ENAB:CARD {self._card}')

    @post_execution_status_poll
    def gnd_disable_all_outputs(self):
        """
        Removes all outputs from list of ports that will be connected to GND
        input if port is unused and bias mode is enabled.
        """
        self.write(f':AGND:CHAN:DIS:CARD {self._card}')

    @post_execution_status_poll
    def couple_port_autodetect(self):
        """Autodetect Kelvin connections on Input ports

        This will detect Kelvin connections on the input ports and enable
        couple mode for found kelvin connections. Kelvin connections must use
        input pairs that can be couple-enabled in order to be autodetected.

        `{(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)}`

        Also refer to the manual for more information.
        """
        self.write(':COUP:PORT:DET')

    def clear_status(self):
        """Clears status register and error queue of the instrument."""
        self.write('*CLS')

    def reset(self):
        """Performs an instrument reset.

        Does not reset error queue!
        """
        self.write('*RST')

    @post_execution_status_poll
    def _set_connection_rule(self, mode):
        if 'free' == self.connection_rule() and 'SROU' == mode:
            warnings.warn('When going from *free* to *single* mode existing '
                          'connections are not released.')

        self.write(f':CONN:RULE {self._card},{mode}')

    @post_execution_status_poll
    def _get_connection_rule(self):
        return self.ask(f':CONN:RULE? {self._card}')

    @staticmethod
    def parse_channel_list(channel_list: str):
        """Generate a set of (input, output) tuples from a SCPI channel
        list string.
        """
        pattern = r'(?P<card>\d{0,1}?)(?P<input>\d{1,2})(?P<output>\d{2})(?=(' \
                  r'?:[,\)\r\n]|$))'
        return {(int(match['input']), int(match['output'])) for match in
                re.finditer(pattern, channel_list)}
