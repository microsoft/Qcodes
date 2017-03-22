from qcodes.instrument.base import Instrument
from qcodes.utils.validators    import Numbers, Enum, Ints, Strings, Anything
from functools import partial
from warnings import warn
try:
    from keysightSD1 import SD_AIN, SD_TriggerModes, SD_AIN_TriggerMode 
    from .SD_Module import SD_Module, result_parser
except ImportError:
    raise ImportError('To use a Signadyne Digitizer, install the Signadyne module')

class SD_DIG(SD_Module):
    """
    This is the qcodes driver for a generic Signadyne Digitizer of the M32/33XX series.

    Status: beta

    This driver is written with the M3300A in mind.

    """
    def __init__(self, name, chassis, slot, channels, triggers, **kwargs):
        """ Initialises a generic Signadyne digitizer and its parameters

            Args:
                name (str)      : the name of the digitizer card
                channels (int)  : the number of input channels the specified card has
                triggers (int)  : the number of trigger inputs the specified card has
        """
        super().__init__(name, chassis, slot, **kwargs)
        self.SD_AIN = SD_AIN()

        # Open the device, using the specified chassis and slot number
        dig_name = self.SD_AIN.getProductNameBySlot(chassis, slot)
        if isinstance(dig_name, str):
            result_code = self.SD_AIN.openWithSlot(dig_name, chassis, slot)
            if result_code <= 0:
                raise Exception('Could not open SD_DIG '
                                'error code {}'.format(result_code))
        else:
            raise Exception('No SD_DIG found at '
                            'chassis {}, slot {}'.format(chassis, slot))

        self.n_channels = channels
        self.n_triggers = triggers

        ########################################################################
        ### Create a set of internal variables to aid set/get cmds in params ###
        ########################################################################

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
        self.__trigger_mode             = [ 0]*self.n_channels
        # For DAQtriggerConfig
        self.__digital_trigger_mode     = [ 0]*self.n_channels
        self.__digital_trigger_source   = [ 0]*self.n_channels
        self.__analog_trigger_mask      = [ 0]*self.n_channels
        # For DAQtriggerExternalConfig       
        self.__external_source          = [ 0]*self.n_channels
        self.__trigger_behaviour        = [ 0]*self.n_channels
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
            vals=Enum(0,1),
            set_cmd=self.SD_AIN.triggerIOconfig,
            docstring='The trigger direction for digitizer trigger port'
        )

        # for clockSetFrequency
        self.add_parameter(
            'sys_frequency',
            label='CLKsys frequency',
            vals=Ints(),
            set_cmd=self.SD_AIN.clockSetFrequency,
            get_cmd=self.SD_AIN.clockGetFrequency,
            docstring='The frequency of internal CLKsys in Hz'
        )

        # for clockGetSyncFrequency
        self.add_parameter(
            'sync_frequency',
            label='CLKsync frequency',
            vals=Ints(),
            get_cmd=self.SD_AIN.clockGetSyncFrequency,
            docstring='The frequency of internal CLKsync in Hz'
        )

        for n in range(self.n_channels):


            # For channelInputConfig
            self.add_parameter(
                'full_scale_{}'.format(n),
                label='Full scale range for channel {}'.format(n),
                # TODO: validator must be set after device opened
                #vals=Numbers(self.SD_AIN.channelMinFullScale(), self.SD_AIN.channelMaxFullScale())
                set_cmd=partial(self.set_full_scale, channel=n),
                get_cmd=partial(self.get_full_scale, channel=n),
                docstring='The full scale voltage for channel {}'.format(n)
            )

            # For channelTriggerConfig
            self.add_parameter(
                'impedance_{}'.format(n),
                label='Impedance for channel {}'.format(n),
                vals=Enum(0,1),
                set_cmd=partial(self.set_impedance, channel=n),
                get_cmd=partial(self.get_impedance, channel=n),
                docstring='The input impedance of channel {}'.format(n)
            )

            self.add_parameter(
                'coupling_{}'.format(n),
                label='Coupling for channel {}'.format(n),
                vals=Enum(0,1),
                set_cmd=partial(self.set_coupling, channel=n),
                get_cmd=partial(self.get_coupling, channel=n),
                docstring='The coupling of channel {}'.format(n)
            )

            # For channelPrescalerConfig 
            self.add_parameter(
                'prescaler_{}'.format(n),
                label='Prescaler for channel {}'.format(n),
                vals=Ints(0,4095),
                set_cmd=partial(self.set_prescaler, channel=n),
                get_cmd=partial(self.get_prescaler, channel=n),
                docstring='The sampling frequency prescaler for channel {}'.format(n)
            )

            # For channelTriggerConfig
            self.add_parameter(
                'trigger_mode_{}'.format(n), label='Trigger mode for channel {}'.format(n), 
                vals=Enum(0,1,2,3,4,5,6,7),
                set_cmd=partial(self.set_trigger_mode, channel=n),
                docstring='The trigger mode for channel {}'.format(n)
            )

            self.add_parameter(
                'trigger_threshold_{}'.format(n),
                label='Trigger threshold for channel {}'.format(n),
                vals=Numbers(-3,3),
                set_cmd=partial(self.set_trigger_threshold, channel=n),
                docstring='The trigger threshold for channel {}'.format(n)
            )

            # For DAQconfig
            self.add_parameter(
                'points_per_cycle_{}'.format(n),
                label='Points per cycle for channel {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_points_per_cycle, channel=n),
                docstring='The number of points per cycle for DAQ {}'.format(n)
            )

            self.add_parameter(
                'n_cycles_{}'.format(n),
                label='n cycles for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_n_cycles, channel=n),
                docstring='The number of cycles to collect on DAQ {}'.format(n)
            )

            self.add_parameter(
                'DAQ_trigger_delay_{}'.format(n),
                label='Trigger delay for for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_DAQ_trigger_delay, channel=n),
                docstring='The trigger delay for DAQ {}'.format(n)
            )

            self.add_parameter(
                'DAQ_trigger_mode_{}'.format(n),
                label='Trigger mode for for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_DAQ_trigger_mode, channel=n),
                docstring='The trigger mode for DAQ {}'.format(n)
            )

            # For DAQtriggerConfig
            self.add_parameter(
                'digital_trigger_mode_{}'.format(n),
                label='Digital trigger mode for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_digital_trigger_mode, channel=n),
                docstring='The digital trigger mode for DAQ {}'.format(n)
            )

            self.add_parameter(
                'digital_trigger_source_{}'.format(n),
                label='Digital trigger source for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_digital_trigger_source, channel=n),
                docstring='The digital trigger source for DAQ {}'.format(n)
            )

            self.add_parameter(
                'analog_trigger_mask_{}'.format(n),
                label='Analog trigger mask for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_analog_trigger_mask, channel=n),
                docstring='The analog trigger mask for DAQ {}'.format(n)
            )

            # For DAQtriggerExternalConfig
            self.add_parameter(
                'ext_trigger_source_{}'.format(n),
                label='External trigger source for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_ext_trigger_source, channel=n),
                docstring='The trigger source for DAQ {}'.format(n)
            )

            self.add_parameter(
                'ext_trigger_behaviour_{}'.format(n),
                label='External trigger behaviour for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_ext_trigger_behaviour, channel=n),
                docstring='The trigger behaviour for DAQ {}'.format(n)
            )

            # For DAQread
            self.add_parameter(
                'n_points_{}'.format(n),
                label='n points for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_n_points, channel=n),
                docstring='The number of points to be read using DAQread on DAQ {}'.format(n)
            )

            self.add_parameter(
                'timeout_{}'.format(n),
                label='timeout for DAQ {}'.format(n),
                vals=Ints(),
                set_cmd=partial(self.set_timeout, channel=n),
                docstring='The read timeout for DAQ {}'.format(n)
            )



    #######################################################
    ###                 User functions                  ###
    #######################################################

    def DAQ_read(self, DAQ, verbose=False):
        """ Read from the specified DAQ

        Args:
            DAQ (int)       : the input DAQ you are reading from

        Parameters:
            n_points
            timeout
        """
        value = self.SD_AIN.DAQread(DAQ, self.__n_points[DAQ], self.__timeout[DAQ])
        value_name = 'DAQ_read channel {}'.format(DAQ)
        return result_parser(value, value_name, verbose)

    def DAQ_start(self, DAQ, verbose=False):
        """ Start acquiring data or waiting for a trigger on the specified DAQ

        Args:
            DAQ (int)       : the input DAQ you are enabling
        """
        value = self.SD_AIN.DAQstart(DAQ)
        value_name = 'DAQ_start channel {}'.format(DAQ)
        return result_parser(value, value_name, verbose)

    def DAQ_start_multiple(self, DAQ_mask, verbose=False):
        """ Start acquiring data or waiting for a trigger on the specified DAQs

        Args:
            DAQ_mask (int)  : the input DAQs you are enabling, composed as a bitmask
                              where the LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        """
        value = self.SD_AIN.DAQstartMultiple(DAQ_mask)
        value_name = 'DAQ_start_multiple mask {:#b}'.format(DAQ_mask)
        return result_parser(value, value_name, verbose)


    def DAQ_stop(self, DAQ, verbose=False):
        """ Stop acquiring data on the specified DAQ

        Args:
            DAQ (int)       : the DAQ you are disabling
        """
        value = self.SD_AIN.DAQstop(DAQ)
        value_name = 'DAQ_stop channel {}'.format(DAQ)
        return result_parser(value, value_name, verbose)

    def DAQ_stop_multiple(self, DAQ_mask, verbose=False):
        """ Stop acquiring data on the specified DAQs
        
        Args:
            DAQ_mask (int)  : the DAQs you are triggering, composed as a bitmask
                              where the LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        """
        value = self.SD_AIN.DAQstopMultiple(DAQ_mask)
        value_name = 'DAQ_stop_multiple mask {:#b}'.format(DAQ_mask)
        return result_parser(value, value_name, verbose)

    def DAQ_trigger(self, DAQ, verbose=False):
        """ Manually trigger the specified DAQ

        Args:
            DAQ (int)       : the DAQ you are triggering
        """
        value = self.SD_AIN.DAQtrigger(DAQ)
        value_name = 'DAQ_trigger channel {}'.format(DAQ)
        return result_parser(value, value_name, verbose)

    def DAQ_trigger_multiple(self, DAQ_mask, verbose=False):
        """ Manually trigger the specified DAQs
        
        Args:
            DAQ_mask (int)  : the DAQs you are triggering, composed as a bitmask
                              where the LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        """
        value = self.SD_AIN.DAQtriggerMultiple(DAQ_mask)
        value_name = 'DAQ_trigger_multiple mask {:#b}'.format(DAQ_mask)
        return result_parser(value, value_name, verbose)

    def DAQ_flush(self, DAQ, verbose=False):
        """ Flush the specified DAQ

        Args:
            DAQ (int)       : the DAQ you are flushing
        """
        value = self.SD_AIN.DAQflush(DAQ)
        value_name = 'DAQ_flush channel {}'.format(DAQ)
        return result_parser(value, value_name, verbose)

    def DAQ_flush_multiple(self, DAQ_mask, verbose=False):
        """ Flush the specified DAQs
        
        Args:
            DAQ_mask (int)  : the DAQs you are flushing, composed as a bitmask
                              where the LSB is for DAQ_0, bit 1 is for DAQ_1 etc.
        """
        value = self.SD_AIN.DAQflushMultiple(DAQ_mask)
        value_name = 'DAQ_flush_multiple mask {:#b}'.format(DAQ_mask)
        return result_parser(value, value_name, verbose)

    def IO_trigger_write(self, val, verbose=False):
        """ Write a value to the IO trigger port

        Args:
            value (int)     : the binary value to write to the IO port

        """
        # TODO: Check if the port is writable
        value = self.SD_AIN.triggerIOwrite(val)
        value_name = 'IO_trigger_write value {}'.format(val)
        return result_parser(value, value_name, verbose)

    def IO_trigger_read(self, verbose=False):
        """ Write a value to the IO trigger port

        """
        # TODO: Check if the port is readable
        value = self.SD_AIN.triggerIOread()
        value_name = 'IO_trigger_read'
        return result_parser(value, value_name, verbose)

    def clock_reset_phase(self, trigger_behaviour, trigger_source, skew = 0.0, verbose=False):
        """ Reset the clock phase between CLKsync and CLKsys
        
        Args:
            trigger_behaviour (int) : 
            trigger_source    (int) : the PXI trigger number
            [skew]         (double) : the skew between PXI_CLK10 and CLKsync in multiples of 10ns
    
        """
        value = self.SD_AIN.clockResetPhase(trigger_behaviour, trigger_source, skew)
        value_name = 'clock_reset_phase behaviour: {}, source: {}, skew: {}'.format(
                trigger_behaviour, trigger_source, skew)
        return result_parser(value, value_name, verbose)

    #######################################################
    ### Functions used internally to set/get parameters ###
    #######################################################

    def set_CLKsys_frequency(self, frequency, verbose=False):
        """ Sets the CLKsys frequency

        Args:

        frequency (int)         : frequency of CLKsys in Hz

        """
        value = 0;
        value_name = 'set_CLKsys_frequency not implemented'
        return result_parser(value, value_name, verbose)

    def get_prescaler(self, channel, verbose=False):
        """ Gets the channel prescaler value

        Args:
            channel (int)       : the input channel you are observing
        """
        value = self.SD_AIN.channelPrescaler(channel)
        # Update internal parameter for consistency
        self.__prescaler[channel] = value
        value_name = 'get_prescaler'
        return result_parser(value, value_name, verbose)

    def set_prescaler(self, prescaler, channel, verbose=False):
        """ Sets the channel sampling frequency via the prescaler

        Args:
            channel (int)       : the input channel you are configuring
            prescaler (int)     : the prescaler value [0..4095]
        """
        self.__prescaler[channel] = prescaler;
        value = self.SD_AIN.channelPrescalerConfig(channel, prescaler)
        value_name = 'set_prescaler {}'.format(prescaler)
        return result_parser(value, value_name, verbose)

    # channelInputConfig
    # NOTE: When setting any of full_scale, coupling or impedance
    # the initial internal value is used as a placeholder, as all 3 arguments
    # are required at once to the Keysight library
    def get_full_scale(self, channel, verbose=False):
        """ Gets the channel full scale input voltage

        Args:
            channel(int)        : the input channel you are observing
        """
        value = self.SD_AIN.channelFullScale(channel)
        # Update internal parameter for consistency
        self.__full_scale[channel] = value
        value_name = 'get_full_scale'
        return result_parser(value, value_name, verbose)

    def set_full_scale(self, full_scale, channel, verbose=False):
        """ Sets the channel full scale input voltage

        Args:
            channel(int)        : the input channel you are configuring
            full_scale (float)  : the input full scale range in volts
        """
        self.__full_scale[channel] = full_scale
        value = self.SD_AIN.channelInputConfig(channel, self.__full_scale[channel],
                                                        self.__impedance[channel],
                                                        self.__coupling[channel])
        value_name = 'set_full_scale {}'.format(full_scale)
        return result_parser(value, value_name, verbose)
    
    def get_impedance(self, channel, verbose=False):
        """ Gets the channel input impedance

        Args:
            channel (int)       : the input channel you are observing
        """
        value = self.SD_AIN.channelImpedance(channel)
        # Update internal parameter for consistency
        self.__impedance[channel] = value
        value_name = 'get_impedance'
        return result_parser(value, value_name, verbose)

    def set_impedance(self, impedance, channel, verbose=False):
        """ Sets the channel input impedance

        Args:
            channel (int)       : the input channel you are configuring
            impedance (int)     : the input impedance (0 = Hi-Z, 1 = 50 Ohm)
        """
        self.__impedance[channel] = impedance
        value = self.SD_AIN.channelInputConfig(channel, self.__full_scale[channel], 
                                                        self.__impedance[channel],
                                                        self.__coupling[channel])
        value_name = 'set_impedance {}'.format(impedance)
        return result_parser(value, value_name, verbose)

    def get_coupling(self, channel, verbose=False):
        """ Gets the channel coupling

        Args:
            channel (int)       : the input channel you are observing
        """
        value = self.SD_AIN.channelCoupling(channel)
        # Update internal parameter for consistency
        self.__coupling[channel] = value
        value_name = 'get_coupling'
        return result_parser(value, value_name, verbose)

    def set_coupling(self, coupling, channel, verbose=False):
        """ Sets the channel coupling

        Args:
            channel (int)       : the input channel you are configuring
            coupling (int)      : the channel coupling (0 = DC, 1 = AC)
        """
        self.__coupling[channel] = coupling
        value = self.SD_AIN.channelInputConfig(channel, self.__full_scale[channel], 
                                                        self.__impedance[channel],
                                                        self.__coupling[channel])
        value_name = 'set_coupling {}'.format(coupling)
        return result_parser(value, value_name, verbose)

    # channelTriggerConfig
    def set_trigger_mode(self, mode, channel, verbose=False):
        """ Sets the current trigger mode from those defined in SD_AIN_TriggerMode

        Args:
            channel (int)       : the input channel you are configuring
            mode (int)          : the trigger mode drawn from the class SD_AIN_TriggerMode
        """
        if (channel > self.n_channels):
            raise ValueError("The specified channel {ch} exceeds the number of channels ({n})".format(ch=channel, n=self.n_channels))
        self.__trigger_mode[channel] = mode
        value = self.SD_AIN.channelTriggerConfig(channel, self.__analog_trigger_mask[channel],
                                                          self.__trigger_threshold[channel])
        value_name = 'set_trigger_mode {}'.format(mode)
        return result_parser(value, value_name, verbose)

    @deprecated
    def get_trigger_mode(self, channel, verbose=False):
        """ Returns the current trigger mode

        Args:
            channel (int)       : the input channel you are observing
        """
        return self.__trigger_mode[channel]

    def set_trigger_threshold(self, threshold, channel, verbose=False):
        """ Sets the current trigger threshold, in the range of -3V and 3V

        Args:
            channel (int)       : the input channel you are configuring
            threshold (float)   : the value in volts for the trigger threshold
        """
        if (channel > self.n_channels):
            raise ValueError("The specified channel {ch} exceeds the number of channels ({n})".format(ch=channel, n=self.n_channels))
        if (threshold > 3 or threshold < -3):
            raise ValueError("The specified threshold {thresh} V does not exist.".format(thresh=threshold))
        self.__trigger_threshold[channel] = threshold
        value = self.SD_AIN.channelTriggerConfig(channel, self.__analog_trigger_mask[channel],
                                                          self.__trigger_threshold[channel])
        value_name = 'set_trigger_threshold {}'.format(threshold)
        return result_parser(value, value_name, verbose)

    @deprecated
    def get_trigger_threshold(channel, verbose=False):
        """ Returns the current trigger threshold

        Args:
            channel (int)       : the input channel you are observing
        """
        return self.__trigger_threshold[channel]

    # DAQConfig
    def set_points_per_cycle(self, n_points, channel, verbose=False):
        """ Sets the number of points to be collected per trigger

        Args:
            channel (int)       : the input channel you are configuring
        """
        self.__points_per_cycle[channel] = n_points
        value = self.SD_AIN.DAQconfig(channel, self.__points_per_cycle[channel],
                                               self.__n_cycles[channel],
                                               self.__trigger_delay[channel],
                                               self.__trigger_mode[channel])
        value_name = 'set_points_per_cycle'.format(n_points)
        return result_parser(value, value_name, verbose)

    def set_n_cycles(self, n_cycles, channel, verbose=False):
        """ Sets the number of trigger cycles to collect data for

        Args:
            channel (int)       : the input channel you are configuring
            n_cycles (int)      : the number of triggers to collect data from

        """
        self.__n_cycles[channel] = n_cycles
        value = self.SD_AIN.DAQconfig(channel, self.__points_per_cycle[channel],
                                               self.__n_cycles[channel],
                                               self.__trigger_delay[channel],
                                               self.__trigger_mode[channel])
        value_name = 'set_n_cycles {}'.format(n_cycles)
        return result_parser(value, value_name, verbose)

    def set_DAQ_trigger_delay(self, delay, channel, verbose=False):
        """ Sets the trigger delay for the specified trigger source

        Args:
            channel (int)       : the input channel you are configuring
            delay   (int)       : the delay in unknown units
        """
        self.__trigger_delay[channel] = delay
        value = self.SD_AIN.DAQconfig(channel, self.__points_per_cycle[channel],
                                               self.__n_cycles[channel],
                                               self.__trigger_delay[channel],
                                               self.__trigger_mode[channel])
        value_name = 'set_DAQ_trigger_delay {}'.format(delay)
        return result_parser(value, value_name, verbose)

    def set_DAQ_trigger_mode(self, mode, channel, verbose=False):
        """ Sets the trigger mode when using an external trigger 

        Args:
            channel (int)       : the input channel you are configuring
            mode  (int)         : the trigger mode you are using
        """
        self.__trigger_mode[channel] = mode
        value = self.SD_AIN.DAQconfig(channel, self.__points_per_cycle[channel],
                                           self.__n_cycles[channel],
                                           self.__trigger_delay[channel],
                                           self.__trigger_mode[channel])
        value_name = 'set_DAQ_trigger_mode {}'.format(mode)
        return result_parser(value, value_name, verbose)

    # DAQtriggerConfig
    def set_digital_trigger_mode(self, mode, channel, verbose=False):
        """

        Args:
            channel (int)       : the input channel you are configuring
            mode  (int)         : the trigger mode you are using
        """
        self.__digital_trigger_mode[channel] = mode
        value = self.SD_AIN.DAQtriggerConfig(channel, self.__digital_trigger_mode[channel],
                                                      self.__digital_trigger_source[channel],
                                                      self.__analog_trigger_mask[channel])
        value_name = 'set_digital_trigger_mode {}'.format(mode)
        return result_parser(value, value_name, verbose)

    def set_digital_trigger_source(self, source, channel, verbose=False):
        """

        Args:
            channel (int)       : the input channel you are configuring
            source  (int)         : the trigger source you are using
        """
        self.__digital_trigger_source[channel] = source
        value = self.SD_AIN.DAQtriggerConfig(channel, self.__digital_trigger_mode[channel],
                                                  self.__digital_trigger_source[channel],
                                                  self.__analog_trigger_mask[channel])
        value_name = 'set_digital_trigger_source {}'.format(source)
        return result_parser(value, value_name, verbose)

    def set_analog_trigger_mask(self, mask, channel, verbose=False):
        """

        Args:
            channel (int)       : the input channel you are configuring
            mask  (int)         : the trigger mask you are using
        """
        self.__analog_trigger_mask[channel] = mask
        value = self.SD_AIN.DAQtriggerConfig(channel, self.__digital_trigger_mode[channel],
                                                      self.__digital_trigger_source[channel],
                                                      self.__analog_trigger_mask[channel])
        value_name = 'set_analog_trigger_mask {}'.format(mask)
        return result_parser(value, value_name, verbose)

    # DAQtriggerExternalConfig
    def set_ext_trigger_source(self, source, channel, verbose=False):
        """ Sets the trigger source 

        Args:
            channel (int)       : the input channel you are configuring
            source  (int)       : the trigger source you are using
        """
        self.__external_source[channel] = source
        value = self.SD_AIN.DAQtriggerExternalConfig(channel, self.__external_source[channel],
                                                              self.__trigger_behaviour[channel])
        value_name = 'set_ext_trigger_source {}'.format(source)
        return result_parser(value, value_name, verbose)

    def set_ext_trigger_behaviour(self, behaviour, channel, verbose=False):
        """ Sets the trigger source 

        Args:
            channel (int)       : the input channel you are configuring
            behaviour  (int)    : the trigger behaviour you are using
        """
        self.__external_behaviour[channel] = behaviour
        value = self.SD_AIN.DAQtriggerExternalConfig(channel, self.__external_source[channel],
                                                              self.__trigger_behaviour[channel])
        value_name = 'set_ext_trigger_behaviour {}'.format(behaviour)
        return result_parser(value, value_name, verbose)
    
    # DAQread
    def set_n_points(self, n_points, channel):
        """ Sets the trigger source 

        Args:
            channel (int)       : the input channel you are configuring
            n_points  (int)     : the number of points to be read from specified DAQ
        """
        self.__n_points[channel] = n_points

    def set_timeout(self, timeout, channel):
        """ Sets the trigger source 

        Args:
            channel (int)       : the input channel you are configuring
            timeout (int)       : the read timeout in ms for the specified DAQ
        """
        self.__timeout[channel] = timeout


    
