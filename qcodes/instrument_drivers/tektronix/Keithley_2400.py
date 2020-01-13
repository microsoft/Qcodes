from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum
from qcodes.utils.helpers import create_on_off_val_mapping


class Keithley_2400(VisaInstrument):
    """
    QCoDeS driver for the Keithley 2400 voltage source.
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter('rangev',
                           get_cmd='SENS:VOLT:RANG?',
                           get_parser=float,
                           set_cmd='SOUR:VOLT:RANG {:f}',
                           label='Voltage range')

        self.add_parameter('rangei',
                           get_cmd='SENS:CURR:RANG?',
                           get_parser=float,
                           set_cmd='SOUR:CURR:RANG {:f}',
                           label='Current range')

        self.add_parameter('compliancev',
                           get_cmd='SENS:VOLT:PROT?',
                           get_parser=float,
                           set_cmd='SENS:VOLT:PROT {:f}',
                           label='Voltage Compliance')

        self.add_parameter('compliancei',
                           get_cmd='SENS:CURR:PROT?',
                           get_parser=float,
                           set_cmd='SENS:CURR:PROT {:f}',
                           label='Current Compliance')

        self.add_parameter('volt',
                           get_cmd=self._get_read_output_protected,
                           get_parser=self._volt_parser,
                           set_cmd=':SOUR:VOLT:LEV {:.8f}',
                           label='Voltage',
                           unit='V',
                           docstring="Sets voltage in 'VOLT' mode. "
                                     "Get returns measured voltage if "
                                     "sensing 'VOLT' otherwise it returns "
                                     "setpoint value. "
                                     "Note that it is an error to read voltage with "
                                     "output off")

        self.add_parameter('curr',
                           get_cmd=self._get_read_output_protected,
                           get_parser=self._curr_parser,
                           set_cmd=':SOUR:CURR:LEV {:.8f}',
                           label='Current',
                           unit='A',
                           docstring = "Sets current in 'CURR' mode. "
                                        "Get returns measured current if "
                                        "sensing 'CURR' otherwise it returns "
                                        "setpoint value. "
                                        "Note that it is an error to read current with "
                                        "output off")


        self.add_parameter('mode',
                           vals=Enum('VOLT', 'CURR'),
                           get_cmd=':SOUR:FUNC?',
                           set_cmd=self._set_mode_and_sense,
                           label='Mode')

        self.add_parameter('sense',
                           vals=Strings(),
                           get_cmd=':SENS:FUNC?',
                           set_cmd=':SENS:FUNC "{:s}"',
                           label='Sense mode')

        self.add_parameter('output',
            set_cmd=':OUTP:STAT {}',
            get_cmd=':OUTP:STAT?',
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"))

        self.add_parameter('nplcv',
                           get_cmd='SENS:VOLT:NPLC?',
                           get_parser=float,
                           set_cmd='SENS:VOLT:NPLC {:f}',
                           label='Voltage integration time')

        self.add_parameter('nplci',
                           get_cmd='SENS:CURR:NPLC?',
                           get_parser=float,
                           set_cmd='SENS:CURR:NPLC {:f}',
                           label='Current integration time')

        self.add_parameter('resistance',
                           get_cmd=self._get_read_output_protected,
                           get_parser=self._resistance_parser,
                           label='Resistance',
                           unit='Ohm',
                           docstring="Measure resistance from current and voltage "
                                     "Note that it is an error to read current "
                                     "and voltage with output off")

        self.write(':TRIG:COUN 1;:FORM:ELEM VOLT,CURR')
        # This line sends 2 commands to the instrument:
        # ":TRIG:COUN 1" sets the trigger count to 1 so that each READ? returns
        # only 1 measurement result.
        # ":FORM:ELEM VOLT,CURR" sets the output string formatting of the the
        # Keithley 2400 to return "{voltage}, {current}".
        # Default value on instrument reset is "VOLT, CURR, RES, TIME, STATUS";
        # however, resistance, status, and time are unused in this driver and
        # so are omitted.
        # These commands do not reset the instrument but do the minimal amount
        # to ensure that voltage and current parameters can be read from the
        # instrument, in the event that output formatting of the instrument was
        # previously changed to some other unknown state.
        self.connect_message()

    def _get_read_output_protected(self) -> str:
        """
        This wrapper function around ":READ?" exists because calling
        ":READ?" on an instrument with output disabled is an error.
        So first we check that output is on and if not we return
        nan for volt, curr etc.
        """
        output = self.output.get_latest()
        if output is None:
            # if get_latest returns None we have
            # to ask the instrument for the status of output
            output = self.output.get()

        if output == 1:
            msg = self.ask(':READ?')
        else:
            raise RuntimeError("Cannot perform read with output off")
        return msg

    def _set_mode_and_sense(self, msg):
        # This helps set the correct read out curr/volt
        if msg == 'VOLT':
            self.sense('CURR')
        elif msg == 'CURR':
            self.sense('VOLT')
        else:
            raise AttributeError('Mode does not exist')
        self.write(':SOUR:FUNC {:s}'.format(msg))

    def reset(self):
        """
        Reset the instrument. When the instrument is reset, it performs the
        following actions.

            Returns the SourceMeter to the GPIB default conditions.

            Cancels all pending commands.

            Cancels all previously send `*OPC` and `*OPC?`
        """
        self.write(':*RST')

    def _volt_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[0]

    def _curr_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[1]

    def _resistance_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        res = fields[0] / fields[1]
        return res
