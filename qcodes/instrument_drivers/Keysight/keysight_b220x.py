from qcodes import VisaInstrument
from qcodes.utils.validators import MultiType, Ints, Enum, Lists

import re


class KeysightB220X(VisaInstrument):
    """
    QCodes driver for B2200 / B2201 switch matrix
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r', **kwargs)

        self.add_function(name='clear_status',
                          call_cmd='*CLS',
                          docstring='Clears status register and error queue of the instrument.')

        self.add_function(name='reset',
                          call_cmd='*RST',
                          docstring='Performs an instrument reset. Does not reset error queue!')

        self.add_parameter(name='get_status',
                           get_cmd='*ESR?',
                           get_parser=lambda status_str: int(status_str),
                           docstring='Queries status register.')

        self.add_parameter(name='get_error',
                           get_cmd=':SYST:ERR?',
                           docstring='Queries error queue')

        self.add_parameter(name='connections',
                           get_cmd=':CLOS:CARD? 0',
                           get_parser=KeysightB220X.parse_channel_list,
                           docstring='queries currently active connections and returns a set of tuples {(input, output), ...}'
                           )

        self.add_function(name='connect',
                          call_cmd=self._connect,
                          args=[Ints(1, 14), Ints(1, 48)],
                          docstring='connects given pair. Two arguments: input, output')

        self.add_function(name='disconnect',
                          call_cmd=self._disconnect,
                          args=[Ints(1, 14), Ints(1, 48)],
                          docstring='disconnects given pair. Two arguments: input, output'
                          )

        self.add_function(name='disconnect_all',
                          call_cmd=':OPEN:CARD 0',
                          docstring='opens all connections. If ground or bias mode is enabled it will connect all outputs to the GND or Bias Port')

        self.add_parameter(name='connection_rule',
                           get_cmd=':CONN:RULE? 0',
                           set_cmd=':CONN:RULE 0,{}',
                           val_mapping={'free': 'FREE',
                                        'single': 'SROU'},
                           docstring="""specifies connection rule. Parameter one of 'free' (default) or 'single'.
                           In 'free' mode
                            - each input port can be connected to multiple output ports 
                            - and each output port can be connected to multiple input ports.
                            - Caution: If the Free connection rule has been specified, ensure multiple input ports are not connected to the same output port. Such configurations can cause damage
                           
                           in single route mode:
                            - each input port can be connected to only one output port
                            - and each output port can be connected to only one input port.
                            - existing connection to a port will be disconnected when a new connection is made.""")

        self.add_parameter(name='connection_sequence',
                           get_cmd=':CONN:SEQ? 0',
                           set_cmd=':CONN:SEQ 0,{}',
                           val_mapping={'none': 'NSEQ',
                                        'bbm': 'BBM',
                                        'mbb': 'MBBR'},
                           docstring="One of 'none', 'bbm' (Break before make) or 'mbb' (make before break)"
                           )

        self.add_function(name='bias_disable_all_outputs',
                          call_cmd=':BIAS:CHAN:DIS:CARD 0')

        self.add_function(name='bias_enable_all_outputs',
                          call_cmd=':BIAS:CHAN:ENAB:CARD 0')

        self.add_function(name='bias_enable_output',
                          call_cmd=':BIAS:CHAN:ENAB (@001{:02d})',
                          args=[Ints(1, 48), ])

        self.add_function(name='bias_disable_output',
                          call_cmd=':BIAS:CHAN:DIS (@001{:02d})',
                          args=[Ints(1, 48), ])

        self.add_parameter(name='bias_input_port',
                           get_cmd=':BIAS:PORT? 0',
                           set_cmd=':BIAS:PORT 0,{}',
                           vals=MultiType(Ints(1, 14),
                                                     Enum(-1)),
                           get_parser=lambda response: int(response),
                           docstring="Selects the input that will be used as bias input port (default 10). The Bias input port cannot be used on subsequent `connect` or `disconnect` commands if Bias mode is ON"
                           )

        self.add_parameter(name='bias_mode',
                           get_cmd=':BIAS? 0',
                           set_cmd=':BIAS 0,{}',
                           val_mapping={True: 1,
                                        False: 0},
                           docstring="Param: True for ON, False for OFF"
                           )

        self.add_function(name='gnd_disable_all_outputs',
                          call_cmd=':AGND:CHAN:DIS:CARD 0')

        self.add_function(name='gnd_disable_output',
                          call_cmd=':AGND:CHAN:DIS (@001{:02d})',
                          args=[Ints(1, 48), ])

        self.add_function(name='gnd_enable_all_outputs',
                          call_cmd=':AGND:CHAN:ENAB:CARD 0')

        self.add_function(name='gnd_enable_output',
                          call_cmd=':AGND:CHAN:ENAB (@001{:02d})',
                          args=[Ints(1, 48), ])

        self.add_parameter(name='gnd_input_port',
                           get_cmd=':AGND:PORT? 0',
                           set_cmd=':AGND:PORT 0,{}',
                           vals=MultiType(Ints(1, 14),
                                                     Enum(-1)),
                           get_parser=lambda response: int(response),
                           docstring="Selects the input that will be used as GND input port (default 12). The GND input port cannot be used on subsequent `connect` or `disconnect` commands if GND mode is ON"
                           )

        self.add_parameter(name='gnd_mode',
                           get_cmd=':AGND? 0',
                           set_cmd=':AGND 0,{}',
                           val_mapping={True: 1,
                                        False: 0}
                           )

        self.add_parameter(name='unused_inputs',
                           get_cmd=':AGND:UNUSED? 0',
                           set_cmd=":AGND:UNUSED 0,'{}'",
                           get_parser=lambda response: [int(x) for x in response.strip("'").split(',') if x.strip().isdigit()],
                           set_parser=lambda value: str(value).strip('[]'),
                           vals=Lists(Ints(1, 14))
                           )

        self.add_parameter(name='couple_ports',
                           get_cmd=':COUP:PORT? 0',
                           set_cmd=":COUP:PORT 0,'{}'",
                           set_parser=lambda value: str(value).strip('[]()'),
                           get_parser=lambda response: [int(x) for x in response.strip("'").split(',') if x.strip().isdigit()],
                           vals=Lists(Enum(1, 3, 5, 7, 9, 11, 13))
                           )

        self.add_function(name='couple_port_autodetect',
                          call_cmd=':COUP:PORT:DET')

        self.add_parameter(name='couple_mode',
                           get_cmd=':COUP? 0',
                           set_cmd=':COUP 0,{}',
                           val_mapping={True: 1,
                                        False: 0},
                           docstring="Param: True for ON, False for OFF"
                           )


    def _connect(self, channel1, channel2):
        self.write(":CLOS (@{card:01d}{ch1:02d}{ch2:02d})".format(card=0,
                                                                  ch1=channel1,
                                                                  ch2=channel2))

    def _disconnect(self, channel1, channel2):
        self.write(":OPEN (@{card:01d}{ch1:02d}{ch2:02d})".format(card=0,
                                                                  ch1=channel1,
                                                                  ch2=channel2))

    @staticmethod
    def parse_channel_list(channel_list_str):
        """Generates a set of (input, output) tuples from a SCPI channel list."""
        pattern = r'(?P<card>\d{0,1}?)(?P<input>\d{1,2})(?P<output>\d{2})(?=(?:[,\)\r\n]|$))'
        return {(int(match['input']), int(match['output'])) for match in re.finditer(pattern, channel_list_str)}
