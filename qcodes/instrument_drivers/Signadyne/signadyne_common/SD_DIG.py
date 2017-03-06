from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from functools import partial
try:
    import Signadyne.signadyne.SD_AIN as SD_AIN_lib
except ImportError:
    raise ImportError('To use a Signadyne Digitizer, install the Signadyne module')

class SD_DIG(Instrument):
    """
    This is the qcodes driver for a generic Signadyne Digitizer of the M32/33XX series.

    Status: pre-alpha

    This driver is written with the M3300A in mind.

    Args:
        name (str)      : the name of the digitizer card
        n_channels (int): the number of digitizer channels for the card 

    """
    def __init__(self, **kwargs):
        super().__init__(name, **kwargs)
        self.SD_AIN = SD_AIN_lib()
        self.n_channels = kwargs['n_channels']

        # Create distinct parameters for each of the digitizer channels
        for n in range(n_channels):
            self.add_parameter(
                'prescaler_{}'.format(n),
                label='Prescaler for channel {}'.format(n),
                initial_value=0,
                vals=range(0,4096),
                # Creates a partial function to allow for single-argument set_cmd to change parameter
                set_cmd=partial(SD_AIN.channelPrescalerConfig,  nChannel=n),
                get_cmd=None,
                docstring='The sampling frequency prescaler for channel {}'.format(n_channels))

            self.add_parameter(
                'trigger_mode_{}'.format(n),
                label='Trigger Mode for channel {}'.format(n),
                initial_value=SD_AIN.AIN_RISING_EDGE,
                vals=[SD_AIN.AIN_RISING_EDGE, SD_AIN.AIN_FALLING_EDGE, SD_AIN.AIN_BOTH_EDGES],
                # Configure this to make senes, needs channel and PR number
                # TODO: Figure out how to access a specific parameter
                set_cmd=partial(SD_AIN.channelTriggerConfig, threshold=self.p),
                get_cmd=None,
                docstring='The trigger mode for channel {}'.format(n_channels))

            self.add_parameter(
                'trigger_threshold_{}'.format(n),
                label='Trigger threshold for channel {}'.format(n),
                initial_value=0,
                unit='volts',
                #vals=, # TODO: Create a validator to set min,max to be -3,+3 V (non-integer)
                # Configure this to make sense, needs channel and PR number
                set_cmd=partial(SD_AIN.channelTriggerConfig, 
                                analogTriggerMode=current_trigger_mode) ,
                get_cmd=None,
                docstring='The trigger mode for channel {}'.format(n_channels))

            self.add_parameter(
                'prescaler_{}'.format(n),
                label='Prescaler for channel {}'.format(n),
                initial_value=0,
                vals=range(0,4096),
                # Creates a partial function to allow for single-argument set_cmd to change parameter
                set_cmd=partial(SD_AIN.channelPrescalerConfig,  nChannel=n),
                get_cmd=None,
                docstring='The sampling frequency prescaler for channel {}'.format(n_channels))
        
    # Wrapper for the DAQconfig function within SD_AIN
    def DAQ_config(channel, pointsPerCycle, nCycles, triggerDelay, triggerMode):
        SD_AIN.DAQconfig(channel, pointsPerCycle, nCycles, triggerDelay, triggerMode)
        
