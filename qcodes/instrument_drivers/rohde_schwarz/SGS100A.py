import numpy as np

from qcodes import Parameter, VisaInstrument
from qcodes.utils import validators as vals
from qcodes.utils.validators import EnumCaseInsensitive


class RohdeSchwarz_SGS100A(VisaInstrument):
    """
    This is the qcodes driver for the Rohde & Schwarz SGS100A signal generator

    Args:
        name: Instrument name
        address: Instrument IP address
        max_frequency: Maximum output frequency. Standard 6 GHz or 12.75 GHz,
            but if combined with an SGU 100A upconverter, this can be increased
            to either 20 GHz or 40 GHz

    .. todo::

        - Add all parameters that are in the manual
        - Add test suite
        - See if there can be a common driver for RS mw sources from which
          different models inherit

    This driver will most likely work for multiple Rohde & Schwarz sources.
    it would be a good idea to group all similar RS drivers together in one
    module.

    Tested working with

    - RS_SGS100A
    - RS_SMB100A

    This driver does not contain all commands available for the RS_SGS100A but
    only the ones most commonly used.
    """

    def __init__(self, name, address, max_frequency=12.75e9, **kwargs):
        super().__init__(name, address, terminator="\n", **kwargs)

        self.add_parameter(
            name="frequency",
            label="Frequency",
            unit="Hz",
            get_cmd="SOUR:FREQ?",
            set_cmd="SOUR:FREQ {:.2f}",
            get_parser=float,
            vals=vals.Numbers(1e6, max_frequency)
        )
        self.add_parameter(
            name="phase",
            label="Phase",
            unit="deg",
            get_cmd="SOUR:PHAS?",
            set_cmd="SOUR:PHAS {:.2f}",
            get_parser=float,
            vals=vals.Numbers(0, 360)
        )
        self.add_parameter(
            name="power",
            label="Power",
            unit="dBm",
            get_cmd="SOUR:POW?",
            set_cmd="SOUR:POW {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-120, 25)
        )
        self.add_parameter(
            name='maximum_power',
            label='Maximum power',
            unit='dBm',
            get_cmd=lambda: self.power.vals._max_value,
            set_cmd=lambda power: setattr(self.power.vals, '_max_value', power),
            initial_value=self.power.vals._max_value,
            # VISA commands do not seem to influence max power
            # get_cmd='SOURce:POWer:LIMit?',
            # set_cmd='SOURce:POWer:LIMit:AMPLitude {:.2f}',
            get_parser=float,
            vals=vals.Numbers(-120, 25),
            docstring='Maximum power limit set in the instrument'
        )

        self.add_parameter(
            "status",
            get_cmd=":OUTP:STAT?",
            set_cmd=":OUTP:STAT {}",
            val_mapping={'on': "1", 'off': "0"}
        )
        self.add_parameter(
            "pulse_modulation_state",
            get_cmd=":SOUR:PULM:STAT?",
            set_cmd=":SOUR:PULM:STAT {}",
            val_mapping={'on': 1, 'off': 0}
        )
        self.add_parameter(
            "pulse_modulation_source",
            get_cmd="SOUR:PULM:SOUR?",
            set_cmd="SOUR:PULM:SOUR {}",
            vals=EnumCaseInsensitive('INTernal', 'EXTernal'),
        )

        # IQ modulation
        self.add_parameter(
            "IQ_modulation",
            get_cmd=":IQ:STAT?",
            set_cmd=":IQ:STAT {}",
            val_mapping={'on': 1, 'off': 0},
            docstring='Switches the I/Q modulation on and off.'
        )
        self.add_parameter(
            "IQ_impairment",
            get_cmd=":IQ:IMPairment:STATe?",
            set_cmd=":IQ:IMPairment:STATe {}",
            val_mapping={'on': 1, 'off': 0},
            docstring='Activates/ deactivates the three impairment or correction '
                      'values leakage, quadrature and IQratio for the baseband '
                      'signal prior to input into the I/Q modulator.'
        )
        self.add_parameter(
            "I_leakage",
            get_cmd=":IQ:IMPairment:LEAKage:I?",
            set_cmd=":IQ:IMPairment:LEAKage:I {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-5, 5),
            docstring='Sets the carrier leakage amplitude for the I-signal component.'
        )
        self.add_parameter(
            "Q_leakage",
            get_cmd=":IQ:IMPairment:LEAKage:Q?",
            set_cmd=":IQ:IMPairment:LEAKage:Q {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-5, 5),
            docstring='Sets the carrier leakage amplitude for the Q-signal component.'
        )
        self.add_parameter(
            "Q_offset",
            get_cmd=":IQ:IMPairment:QUADrature:ANGLe?",
            set_cmd=":IQ:IMPairment:QUADrature:ANGLe {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-8, 8),
            docstring='Sets the quadrature offset for the digital I/Q signal.'
        )
        self.add_parameter(
            "IQ_ratio",
            get_cmd=":IQ:IMPairment:IQRatio:MAGNitude?",
            set_cmd=":IQ:IMPairment:IQRatio:MAGNitude {:.3f}",
            get_parser=float,
            vals=vals.Numbers(-1, 1),
            docstring='Sets the ratio of I modulation to Q modulation '
                      '(amplification “imbalance”). The input may be either in '
                      'dB or %. The resolution is 0.001 dB, an input in percent '
                      'is rounded to the closest valid value in dB. '
                      'A query returns the value in dB.'
        )
        self.add_parameter(
            "IQ_wideband",
            get_cmd=":IQ:WBSTate?",
            set_cmd=":IQ:WBSTate {}",
            val_mapping={'on': 1, 'off': 0},
            docstring='Selects optimized settings for wideband modulation signals.'
        )
        self.add_parameter(
            "IQ_crestfactor",
            get_cmd=":IQ:CREStfactor?",
            set_cmd=":IQ:CREStfactor {:.2f}",
            get_parser=float,
            vals=vals.Numbers(0, 80)
        )


        # Reference oscillator functions
        self.add_parameter(
            "reference_oscillator_source",
            label="Reference oscillator source",
            get_cmd="SOUR:ROSC:SOUR?",
            set_cmd="SOUR:ROSC:SOUR {}",
            vals=EnumCaseInsensitive('INTernal', 'EXTernal'),
        )
        self.add_parameter(
            "reference_oscillator_output_frequency",
            label="Reference oscillator output frequency",
            get_cmd="SOUR:ROSC:OUTP:FREQ?",
            set_cmd="SOUR:ROSC:OUTP:FREQ {}",
            vals=vals.Enum("10MHz", "100MHz", "1000MHz"),
            docstring='Output frequency when used as a reference'
        )
        self.add_parameter(
            "reference_oscillator_external_frequency",
            label="Reference oscillator external frequency",
            get_cmd="SOUR:ROSC:EXT:FREQ?",
            set_cmd="SOUR:ROSC:EXT:FREQ {}",
            vals=vals.Enum("10MHz", "100MHz", "1000MHz"),
            docstring='Frequency of the external reference'
        )

        self.add_parameter(
            'lock_number',
            label='Lock number',
            set_cmd=None,
            initial_value=np.random.randint(10000, 99999999),
            vals=vals.Ints(10000, 99999999),
            docstring='Number used to lock instrument to this controller. '
                      'Should be unique, and a random number is chosen by default.'
                      'Instrument can be locked by .lock()'
        )

        self.add_parameter(
            name='get_errors',
            label='Get errors',
            get_cmd=':SYSTem:ERRor:ALL?',
            docstring='Get list of errors, removing them from the queue'
        )

        self.add_parameter(
            name='operation_mode',
            label='Operation mode',
            get_cmd=':SOUR:OPM?',
            set_cmd=':SOUR:OPM {}',
            vals=EnumCaseInsensitive('NORMal', 'BBBYpass'),
            docstring='Operation mode, bbbypass is baseband-bypass, which directly '
                      'routes the I and Q port to the RF out.'
        )

        self.add_function("reset", call_cmd="*RST; *CLS")
        self.add_function("run_self_tests", call_cmd="*TST?")
        self.add_function("restart", call_cmd=":RESTART")

        self.connect_message()

        # Query current values for all parameters
        for param_name, param in self.parameters.items():
            param()


    def on(self):
        self.status('on')

    def off(self):
        self.status('off')

    def lock(self, lock_number=None):
        if lock_number is not None:
            self.lock_number(lock_number)

        result = self.ask(f":LOCK? {self.lock_number()}")
        if result != "1":
            raise RuntimeError('Instrument has already been locked to another '
                               'number, please first use .unlock()')

    def unlock(self, lock_number=None):
        if lock_number is None:
            lock_number = self.lock_number()

        self.write(f":UNLOCK {lock_number}")

    def add_parameter(
        self,
        name,
        parameter_class=Parameter,
        parent=None,
        vals=None,
        val_mapping=None,
        **kwargs,
    ):
        if isinstance(vals, EnumCaseInsensitive) and val_mapping is None:
            val_mapping = vals.val_mapping

        return super().add_parameter(
            name,
            parameter_class=parameter_class,
            parent=parent,
            vals=vals,
            val_mapping=val_mapping,
            **kwargs
        )