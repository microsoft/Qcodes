from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from functools import partial
try:
    import Signadyne.signadyne.SD_AIN as SD_AIN
    import Signadyne.signadyne.SD_AIN_TriggerMode as SD_TriggerMode
    # TODO: Import all Signadyne classes as themselves
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
        """ Initialises a generic Signadyne digitizer and its parameters

            Args:
                name (str)          : the name of the digitizer card
                [n_channels] (int)  : the number of input channels the specified card has
        """
        super().__init__(name, **kwargs)
        self.SD_AIN = SD_AIN()
        self.n_channels = kwargs['n_channels']


        # Create distinct parameters for each of the digitizer channels
        for n in range(n_channels):

            ########################################################################
            ### Create a set of internal variables to aid set/get cmds in params ###
            ########################################################################

            # For channelInputConfig
            self.__full_scale[self, n]        =  1 # By default, full scale = 1V
            self.__impedance[self, n]         =  0 # By default, Hi-z
            self.__coupling[self, n]          =  0 # By default, DC coupling
            # For channelPrescalerConfig 
            self.__prescaler[self, n]         =  0 # By default, no prescaling
            # For channelTriggerConfig
            self.__trigger_mode[self, n]      =  SD_TriggerMode.RISING_EDGE
            self.__trigger_threshold[self, n] =  0 # By default, threshold at 0V
            # For DAQconfig
            self.__points_per_cycle[self, n]  =  0
            self.__n_cycles                   =  0
            self.__trigger_delay              =  0
            self.__trigger_mode               =  SD_TriggerMode.RISING_EDGE
            # For DAQtriggerExternalConfig
            self.__digital_trigger_mode       =  0 
            self.__digital_trigger_source     =  0
            self.__analog_trigger_mask        =  0
            # For DAQread
            self.__n_points                   =  0
            self.__timeout                    = -1
            # for triggerIOconfig
            self.__direction                  =  0
            # for clockSetFrequency
            self.__frequency                  =  100e6
            # for clockResetPhase
            self.__trigger_behaviour          =  0 
            self.__PXItrigger                 =  0
            self.__skew                       =  0

            self.add_parameter(
                'prescaler_{}'.format(n),
                label='Prescaler for channel {}'.format(n),
                initial_value=0,
                vals=range(0,4096),
                # Creates a partial function to allow for single-argument set_cmd to change parameter
                set_cmd=partial(SD_AIN.channelPrescalerConfig,  nChannel=n),
                get_cmd=None,
                docstring='The sampling frequency prescaler for channel {}'.format(n_channels))

    def set_trigger_mode(channel, mode=None):
        """ Sets the current trigger mode from those defined in SD_TriggerMode

        Args:
            channel (int)       : the input channel you are modifying
            mode (int)          : the trigger mode drawn from the class SD_TriggerMode
        """
        if (channel > self.n_channels):
            raise ValueError("The specified channel {ch} exceeds the number of channels ({n})".format(ch=channel, n=self.n_channels)
        if (mode not in SD_trigger_modes):
            raise ValueError("The specified mode {mode} does not exist.".format(mode=mode))
        self.__trigger_mode[self, channel] = mode
        # TODO: Call the SD library to set the current mode


    def get_trigger_mode(channel):
        """ Returns the current trigger mode

        Args:
            channel (int)       : the input channel you are observing
        """
        return self.__trigger_mode[self, channel]


    def set_trigger_threshold(channel, threshold=0):
        """ Sets the current trigger threshold, in the range of -3V and 3V

        Args:
            channel (int)       : the input channel you are modifying
            threshold (float)   : the value in volts for the trigger threshold
        """
        if (channel > self.n_channels):
            raise ValueError("The specified channel {ch} exceeds the number of channels ({n})".format(ch=channel, n=self.n_channels)
        if (threshold > 3 or threshold < -3):
            raise ValueError("The specified threshold {thresh} V does not exist.".format(thresh=threshold))
        self.__trigger_threshold[self, channel] = threshold
        # TODO: Call the SD library to set the current threshold


    def get_trigger_threshold(channel):
        """ Returns the current trigger threshold

        Args:
            channel (int)       : the input channel you are observing
        """
        return self.__trigger_threshold[self, channel]

