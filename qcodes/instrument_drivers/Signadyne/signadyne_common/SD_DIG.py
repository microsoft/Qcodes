from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators    import Numbers, Enum, Ints, Strings, Anything
from functools import partial
try:
    import signadyne.SD_AIN as SD_AIN
    import signadyne.SD_AIN_TriggerMode as SD_AIN_TriggerMode # for channel edge sensitivities
    import signadyne.SD_TriggerModes  as SD_TriggerModes      # for channel trigger source
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
        self.n_channels = kwargs.pop('n_channels')
        super().__init__(**kwargs)
        self.name       = kwargs['name']
        self.SD_AIN = SD_AIN()

        ########################################################################
        ### Create a set of internal variables to aid set/get cmds in params ###
        ########################################################################

        # for triggerIOconfig
        self.__direction                  =  0
        # for clockSetFrequency
        self.__frequency                  =  100e6
        # for clockResetPhase
        self.__trigger_behaviour          =  0 
        self.__PXItrigger                 =  0
        self.__skew                       =  0

        # Create distinct parameters for each of the digitizer channels

        # For channelInputConfig
        self.__full_scale               = [ 1]*self.n_channels # By default, full scale = 1V
        self.__impedance                = [ 0]*self.n_channels # By default, Hi-z
        self.__coupling                 = [ 0]*self.n_channels # By default, DC coupling
        # For channelPrescalerConfig         
        self.__prescaler                = [ 0]*self.n_channels # By default, no prescaling
        # For channelTriggerConfig           
        self.__trigger_mode             = [ SD_AIN_TriggerMode.RISING_EDGE]*self.n_channels
        self.__trigger_threshold        = [ 0]*self.n_channels # By default, threshold at 0V
        # For DAQconfig                      
        self.__points_per_cycle         = [ 0]*self.n_channels
        self.__n_cycles                 = [ 0]*self.n_channels
        self.__trigger_delay            = [ 0]*self.n_channels
        self.__trigger_mode             = [ SD_AIN_TriggerMode.RISING_EDGE]*self.n_channels
        # For DAQtriggerExternalConfig       
        self.__digital_trigger_mode     = [ 0] *self.n_channels
        self.__digital_trigger_source   = [ 0]*self.n_channels
        self.__analog_trigger_mask      = [ 0]*self.n_channels
        # For DAQread                        
        self.__n_points                 = [ 0]*self.n_channels
        self.__timeout                  = [-1]*self.n_channels
        
        ###############################################
        ###         Create internal parameters      ###
        ###############################################

        # for triggerIOconfig
        self.add_parameter(
            'trigger_direction',
            label='Trigger direction for trigger port',
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The trigger direction for digitizer trigger port'
        )

        # for clockSetFrequency
        self.add_parameter(
            'frequency',
            label='CLKsys frequency',
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The frequency of internal CLKsys in Hz'
        )

        # for clockResetPhase
        self.add_parameter(
            'trigger_behaviour',
            label='Trigger behaviour for resetting CLKsys phase',
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The trigger behaviour for resetting CLKsys phase'
        )

        self.add_parameter(
            'PXI_trigger',
            label='PXI trigger for clockResetPhase',
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The PXI trigger which resets CLKsys'
        )

        self.add_parameter(
            'skew',
            label='Skew between PXI_CLK10 and CLKsync',
            vals=Ints(),
            set_cmd=None,
            get_cmd=None,
            docstring='The skew between PXI_CLK10 and CLKsync in multiples of 10 ns'
        )

        for n in range(n_channels):


            # For channelInputConfig
            self.add_parameter(
                'full_scale_{}'.format(n),
                label='Full scale range for channel {}'.format(n),
                vals=Numbers(SD_AIN.channelMinFullScale(), SD_AIN.channelMaxFullScale())
                # Creates a partial function to allow for single-argument set_cmd to change parameter
                set_cmd=partial(set_full_scale, channel=n),
                get_cmd=partial(SD_AIN.channelFullScale, channel=n),
                docstring='The full scale voltage for channel {}'.format(n)
            )

            # For channelTriggerConfig
            self.add_parameter(
                'impedance_{}'.format(n),
                label='Impedance for channel {}'.format(n),
                vals=Enum([0,1]),
                set_cmd=partial(set_impedance, channel=n),
                get_cmd=partial(SD_AIN.channelImpedance, channel=n),
                docstring='The input impedance of channel {}'.format(n)
            )

            self.add_parameter(
                'coupling_{}'.format(n),
                label='Coupling for channel {}'.format(n),
                vals=Enum([0,1]),
                set_cmd=partial(set_coupling, channel=n),
                get_cmd=partial(SD_AIN.channelCoupling, channel=n),
                docstring='The coupling of channel {}'.format(n)
            )

            # For channelPrescalerConfig 
            self.add_parameter(
                'prescaler_{}'.format(n),
                label='Prescaler for channel {}'.format(n),
                vals=Ints(0,4096),
                set_cmd=partial(SD_AIN.channelPrescalerConfig,  channel=n),
                get_cmd=partial(SD_AIN.channelPrescaler, channel=n),
                docstring='The sampling frequency prescaler for channel {}'.format(n)
            )

            # For channelTriggerConfig
            self.add_parameter(
                'trigger_mode_{}'.format(n), label='Trigger mode for channel {}'.format(n), 
                vals=Enum([1,2,3]),
                set_cmd=partial(set_trigger_mode, channel=n),
                docstring='The trigger mode for channel {}'.format(n)
            )

            self.add_parameter(
                'trigger_threshold_{}'.format(n),
                label='Trigger threshold for channel {}'.format(n),
                vals=Numbers(-3,3),
                set_cmd=None,
                docstring='The trigger threshold for channel {}'.format(n)
            )

            # For DAQconfig
            self.add_parameter(
                'points_per_cycle_{}'.format(n),
                label='Points per cycle for channel {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The number of points per cycle for DAQ {}'.format(n)
            )

            self.add_parameter(
                'n_cycles_{}'.format(n),
                label='n cycles for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The number of cycles to collect on DAQ {}'.format(n)
            )

            self.add_parameter(
                'ext_trigger_delay_{}'.format(n),
                label='Trigger delay for for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The trigger delay for DAQ {}'.format(n)
            )

            # For DAQtriggerExternalConfig
            self.add_parameter(
                'ext_trigger_mode_{}'.format(n),
                label='External trigger mode for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The trigger mode for DAQ {}'.format(n)
            )

            self.add_parameter(
                'digital_trigger_mode_{}'.format(n),
                label='Digital trigger mode for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The digital trigger mode for DAQ {}'.format(n)
            )

            self.add_parameter(
                'digital_trigger_source_{}'.format(n),
                label='Digital trigger source for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The digital trigger source for DAQ {}'.format(n)
            )

            self.add_parameter(
                'analog_trigger_mask_{}'.format(n),
                label='Analog trigger mask for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The analog trigger mask for DAQ {}'.format(n)
            )

            # For DAQread
            self.add_parameter(
                'n_points_{}'.format(n),
                label='n points for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The number of points to be read using DAQread on DAQ {}'.format(n)
            )

            self.add_parameter(
                'timeout_{}'.format(n),
                label='timeout for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=None,
                docstring='The read timeout for DAQ {}'.format(n)
            )


    def set_IO_trigger_direction(direction):
        """ Sets the external port trigger direction

        Args:
            direction (int)     : the port direction (0 = output, 1 = input)

        """
        pass

    def set_CLKsys_frequency(frequency):
        """ Sets the CLKsys frequency

        Args:

        frequency (int)         : frequency of CLKsys in Hz

        """
        pass

    def set_trigger_behaviour(behaviour):
        """ Sets the trigger behaviour in resetting the CLKsync and CLKsys phases

        Args:
            behaviour (int)     : edge sensitivity to the PXI trigger

        """
        pass

    def set_PXI_trigger(PXI):
        """ Sets the PXI trigger which causes the phase reset of CLKsync and CLKsys

        Args:
            PXI (int)           : the PXI trigger number
        """
        pass

    def set_skew(skew):
        """ Sets the skew between PXI_CLK10 and CLKsync in multiples of 10 ns

        Args:
            skew (int)          : the skew value (1 = 10ns, 2 = 20ns, etc.)
        """
        pass

    def set_channel_input_config(channel, fullScale, impedance, coupling):
        """ Sets the input configuration for the specified channel

        Args:
            channel (int)       : the input channel you are modifying
            fullScale (float)   : the full scale input range in volts
            impedance (int)     : the input impedance (0 = Hi-Z, 1 = 50 Ohm)
            coupling (int)      : the channel coupling (0 = DC, 1 = AC)
        """
        pass

    def set_prescaler(channel, prescaler):
        """ Sets the channel sampling frequency via the prescaler

        Args:
            channel (int)       : the input channel you are modifying
            prescaler (int)     : the prescaler value [0..4095]
        """
        pass

    def set_full_scale(channel, full_scale):
        """ Sets the channel full scale input voltage

        Args:
            channel(int)        : the input channel you are modifying
            full_scale (float)  : the input full scale range in volts
        """
        pass
    
    def set_impedance(channel, impedance):
        """ Sets the channel input impedance

        Args:
            channel (int)       : the input channel you are modifying
            impedance (int)     : the input impedance (0 = Hi-Z, 1 = 50 Ohm)
        """
        pass

    def set_coupling(channel, coupling):
        """ Sets the channel coupling

        Args:
            channel (int)       : the input channel you are modifying
            coupling (int)      : the channel coupling (0 = DC, 1 = AC)
        """
        pass

    def set_trigger_mode(channel, mode=None):
        """ Sets the current trigger mode from those defined in SD_AIN_TriggerMode

        Args:
            channel (int)       : the input channel you are modifying
            mode (int)          : the trigger mode drawn from the class SD_AIN_TriggerMode
        """
        if (channel > self.n_channels):
            raise ValueError("The specified channel {ch} exceeds the number of channels ({n})".format(ch=channel, n=self.n_channels))
        if mode not in vars(SD_AIN_TriggerMode):
            raise ValueError("The specified mode {} does not exist.".format(mode))
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
            raise ValueError("The specified channel {ch} exceeds the number of channels ({n})".format(ch=channel, n=self.n_channels))
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

