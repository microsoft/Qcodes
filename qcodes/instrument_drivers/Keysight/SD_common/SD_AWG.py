from functools import partial
import numpy as np
from .SD_Module import SD_Module, keysightSD1, SignadyneParameter, error_check, with_error_check

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
# TODO add channels
from qcodes import validators as vals

model_channels = {'M3201A': 4,
                  'M3300A': 4}


class AWGChannel(InstrumentChannel):
    def __init__(self, parent: Instrument, name:str, id:int, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)

        self.awg = self._parent.awg
        self.wave = self._parent.wave
        self.id = id

        # TODO: Joint amplitude and offset validation (-1.5<amp+offset<1.5)
        self.add_parameter('amplitude',
                           label=f'ch{self.id} amplitude',
                           unit='V',
                           set_function=self.awg.channelAmplitude,
                           docstring=f'ch{self.id} amplitude',
                           vals=vals.Numbers(-1.5, 1.5))
        self.add_parameter('offset',
                           label=f'ch{self.id} offset',
                           unit='V',
                           set_function=self.awg.channelOffset,
                           docstring=f'The DC offset of ch{self.id}',
                           vals=vals.Numbers(-1.5, 1.5))

        self.add_parameter('wave_shape',
                           label=f'ch{self.id} wave shape',
                           set_function=self.awg.channelWaveShape,
                           val_mapping={'HiZ': -1, 'none': 0, 'sinusoidal': 1,
                                        'triangular': 2, 'square': 4, 'dc': 5,
                                        'arbitrary': 6, 'partner_channel': 8},
                           docstring=f'The output waveform type of ch{self.id}. '
                                     f'Can be either arbitrary (AWG), or one '
                                     f'of the function generator types.')

        self.add_parameter('trigger_direction',
                           label=f'ch{self.id} trigger direction',
                           val_mapping={'input': 1, 'output': 0},
                           set_function=self.awg.triggerIOconfig,
                           docstring='Determines if trig i/o should be used '
                                     'as a trigger input or trigger output.')

        self.add_parameter('trigger_source',
                           label=f'ch{self.id} trigger source',
                           val_mapping={'trig_in': 0,
                                        **{f'pxi{k}': 4000+k for k in range(1, 9)}},
                           initial_value='trig_in',
                           set_function=self.awg.AWGtriggerExternalConfig,
                           set_args=['trigger_source', 'trigger_behaviour'],
                           docstring='External trigger source used to proceed '
                                     'to next waveform. Only used if the '
                                     'waveform is queued with external trigger '
                                     'mode option. trig_in requires '
                                     'trigger_direction == "input".')

        self.add_parameter('trigger_mode',
                           label=f'ch{self.id} trigger mode',
                            initial_value='rising',
                            val_mapping={'active_high': 1, 'active_low': 2,
                                         'rising': 3, 'falling': 4},
                            set_function=self.SD_AIN.DAQdigitalTriggerConfig,
                            set_args=['trigger_source', 'trigger_mode'],
                            docstring='The digital trigger mode when the '
                                      'waveform is queued with external trigger '
                                      'mode option.')

        # AWG parameters
        self.add_parameter('queue_mode',
                           set_function=self.awg.AWGqueueConfig,
                           val_mapping={'one-shot': 0, 'cyclic': 1})

        # Function generator parameters
        self.add_parameter(f'frequency',
                           label=f'ch{self.id} frequency',
                           unit='Hz',
                           set_function=self.awg.channelFrequency,
                           docstring=f'The frequency of ch{self.id}, only used '
                                     f'for the function generator (wave_shape '
                                     f'is not arbitrary).',
                           vals=vals.Numbers(0, 200e6))
        self.add_parameter(f'phase',
                           label=f'ch{self.id} phase',
                           unit='deg',
                           set_function=self.awg.channelPhase,
                           docstring=f'The phase of ch{self.id}, only used '
                                     f'for the function generator (wave_shape '
                                     f'is not arbitrary).',
                           vals=vals.Numbers(0, 360))
        self.add_parameter(f'IQ',
                           label=f'ch{self.id} IQ modulation',
                           val_mapping={'on': 1, 'off': 0},
                           set_function=self.awg.modulationIQconfig,
                           docstring=f'Enable or disable IQ modulation for '
                                     f'ch{self.id}. If enabled, IQ modulation '
                                     f'will be applied to the function '
                                     f'generator signal using the AWG.')
        self.add_parameter(f'angle_modulation',
                           label=f'ch{self.id} angle modulation',
                           val_mapping={'none': 0, 'frequency': 1, 'phase': 2},
                           set_function=self.awg.modulationAngleConfig,
                           set_args=['angle_modulation', 'deviation_gain'],
                           docstring=f'Type of modulation to use for the '
                                     f'function generator. Can be frequency or '
                                     f'phase.')
        self.add_parameter(f'deviation_gain',
                           label=f'ch{self.id} angle modulation',
                           vals=vals.Numbers(),
                           set_function=self.awg.modulationAngleConfig,
                           initial_value=0,
                           set_args=['angle_modulation', 'deviation_gain'],
                           docstring=f'Function generator modulation gain.'
                                     f'To be used with angle_modulation')
        @with_error_check
        def config_angle_modulation(self,
                                    channel_number: int,
                                    modulation_type: int,
                                    deviation_gain: int):
            """
            Configures the modulation in frequency/phase for the selected channel

            Args:
                channel_number: the number of the channel to configure
                modulation_type: the modulation type the AWG is used for
                    No Modulation           :   0
                    Frequency Modulation    :   1
                    Phase Modulation        :   2
                deviation_gain: gain for the modulating signal
            """
            return self.awg.modulationAngleConfig(channel_number,
                                                  modulation_type,
                                                  deviation_gain)

    def add_parameter(self, name: str,
                      parameter_class: type=SignadyneParameter, **kwargs):
        """Use SignadyneParameter by default"""
        super().add_parameter(name=name, parameter_class=parameter_class,
                              parent=self, **kwargs)

    @with_error_check
    def reset_phase(self):
        """Resets the function generator accumulated phase for this channel.

        This accumulated phase is the result of the phase continuous operation
        of the product, used by the function generator.
        """
        return self.awg.channelPhaseReset(self.id)

    @with_error_check
    def flush_waveforms(self):
        """Empties the queue of the selected AWG.

        Waveforms are not removed from the onboard RAM.
        """
        self.awg.AWGflush(self.id)

    @with_error_check
    def queue_waveform(self,
                       waveform_number: int,
                       trigger_mode: int,
                       start_delay: int,
                       cycles: int,
                       prescaler: int):
        """Queues the specified waveform in one of the AWGs of the module.

        The waveform must be already loaded in the module onboard RAM.
        """
        return self.awg.AWGqueueWaveform(self.id, waveform_number, trigger_mode,
                                         start_delay, cycles, prescaler)

    @with_error_check
    def waveform_from_array(self,
                            trigger_mode: int,
                            start_delay: int,
                            cycles: int,
                            prescaler: int,
                            waveform_type: int,
                            waveform_data_a: np.ndarray,
                            waveform_data_b: np.ndarray = None,
                            padding_mode: int = 0):
        """Load, queue and start a single waveform on a channel from an array

        Args:
            trigger_mode: trigger method to launch the waveform
                Auto                        :   0
                Software/HVI                :   1
                Software/HVI (per cycle)    :   5
                External trigger            :   2
                External trigger (per cycle):   6
            start_delay: defines the delay between trigger and wf launch
                given in multiples of 10ns.
            cycles: number of times the waveform is repeated once launched
                zero = infinite repeats
            prescaler: waveform prescaler value, to reduce eff. sampling rate
            waveform_type: waveform type
            waveform_data_a: array with waveform points
            waveform_data_b: array with waveform points, only for the waveforms
                                        which have a second component

        Returns:
            availableRAM: available onboard RAM in waveform points,
                or negative numbers for errors
        """
        return self.awg.AWGfromArray(self.id, trigger_mode, start_delay,
                                     cycles, prescaler, waveform_type,
                                     waveform_data_a, waveform_data_b, padding_mode)

    @with_error_check
    def awg_from_file(self,
                      waveform_file: str,
                      trigger_mode: int,
                      start_delay: int,
                      cycles: int,
                      prescaler: int,
                      padding_mode: int = 0) -> int:
        """Load, queue and start a single waveform on a channel from a file

        Args:
            waveform_file (str): file containing the waveform points
            trigger_mode: trigger method to launch the waveform
                Auto                        :   0
                Software/HVI                :   1
                Software/HVI (per cycle)    :   5
                External trigger            :   2
                External trigger (per cycle):   6
            start_delay: defines the delay between trigger and wf launch
                given in multiples of 10ns.
            cycles: number of times the waveform is repeated once launched
                zero = infinite repeats
            prescaler: waveform prescaler value, to reduce eff. sampling rate

        Returns:
            availableRAM: available onboard RAM in waveform points,
                or negative numbers for errors
        """
        return self.awg.AWGFromFile(self.id, waveform_file, trigger_mode,
                                    start_delay, cycles, prescaler,
                                    padding_mode)

    @with_error_check
    def configure_markers(self,
                          marker_mode: int,
                          pxi_mask: int,
                          io_mask: int,
                          trigger_value: int,
                          sync_mode: int,
                          length: int,
                          delay: int):
        """Configures the queue markers (pxi's & trig_out) for the selected awg.

        This allows control of the internal PXI triggers and the external IO
        trigger port by the current state of the waveform queue.

        Note:
            The same marker configuration is applied for every waveform in the
            sequence.

        Args:
            marker_mode: Operation mode of the queue
                Disabled                    : 0
                On WF start                 : 1
                On WF start after WF delay  : 2
                On every cycle              : 3
                End (not implemented)       : 4

            pxi_mask: select/deselect the internal pxi trigger lines
                            (bit0 = PXI0 etc., 1 = selected)
            io_mask: select/deselect the external io port (1 = selected)
            trigger_value: the value to write to the selected trigger ports
            sync_mode: the mode to synchronise the pulse with
                Immediate : 0
                Sync10    : 1
            length: marker pulse length in multiples of TCLK * 5
            delay: marker pulse delay in multiples of TCLK * 5
        """
        return self.awg.AWGqueueMarkerConfig(self.id, marker_mode, pxi_mask,
                                             io_mask, trigger_value, sync_mode,
                                             length, delay)
    @with_error_check
    def start(self):
        """Starts the selected AWG from the beginning of its queue.

        The generation will start immediately or when a trigger is received,
        depending on the trigger selection of the first waveform in the queue
        and provided that at least one waveform is queued in the AWG.
        """
        self.awg.AWGstart(self.id)

    @with_error_check
    def pause(self):
        """Pauses the selected AWG, leaving the last waveform point at the
        output, and ignoring all incoming triggers.
        The waveform generation can be resumed calling awg_resume
        """
        self.awg.AWGpause(self.id)

    @with_error_check
    def resume(self):
        """Resumes the selected AWG, from the current position of the queue.
        """
        self.awg.AWGresume(self.id)

    def stop(self):
        """Stops the selected AWG, setting the output to zero and resetting the
        AWG queue to its initial position.
        All following incoming triggers are ignored.
        """
        self.awg.AWGstop(self.id)

    def jump_next_waveform(self):
        """Forces a jump to the next waveform in the awg queue.

        The jump is executed once the current waveform has finished a complete
        cycle.
        """
        self.awg.AWGjumpNextWaveform(self.id)

    def trigger(self):
        """Triggers the selected AWG.

        The waveform waiting in the current position of the queue is launched,
        provided it is configured with VI/HVI Trigger.
        """
        self.awg.AWGtrigger(self.id)


class SD_AWG(SD_Module):
    """
    This is the general SD_AWG driver class that implements shared parameters and functionality among all PXIe-based
    AWG cards by Keysight. (series M32xxA and M33xxA)

    This driver was written to be inherited from by a specific AWG card driver (e.g. M3201A).

    This driver was written with the M3201A card in mind.

    This driver makes use of the Python library provided by Keysight as part of the SD1 Software package (v.2.01.00).
    """

    def __init__(self, name, model, chassis, slot, channels=None, triggers=8,
                 **kwargs):
        super().__init__(name, model, chassis, slot, triggers, **kwargs)

        if channels is None:
            channels = model_channels[self.model]
        self.n_channels = channels

        # Create instance of keysight SD_AOU class
        self.awg = keysightSD1.SD_AOU()

        # Create an instance of keysight SD_Wave class
        self.wave = keysightSD1.SD_Wave()

        # Open the device using the specified chassis and slot number
        self.initialize(chassis=chassis, slot=slot)

        self.add_parameter('trigger_io',
                           label='trigger io',
                           get_function=self.awg.triggerIOread,
                           set_function=self.awg.triggerIOwrite,
                           docstring='The trigger input value, 0 (OFF) or 1 (ON)',
                           vals=vals.Enum(0, 1))
        self.add_parameter('clock_frequency',
                           label='clock frequency',
                           unit='Hz',
                           get_cmd=self.awg.clockGetFrequency,
                           set_cmd=self.awg.clockSetFrequency,
                           docstring='The real hardware clock frequency in Hz',
                           vals=vals.Numbers(100e6, 500e6))
        self.add_parameter('clock_sync_frequency',
                           label='clock sync frequency',
                           unit='Hz',
                           get_cmd=self.awg.clockGetSyncFrequency,
                           docstring='The frequency of the internal CLKsync in Hz')

    def add_parameter(self, name: str,
                      parameter_class: type=SignadyneParameter, **kwargs):
        """Use SignadyneParameter by default"""
        super().add_parameter(name=name, parameter_class=parameter_class,
                              parent=self, **kwargs)

    def initialize(self, chassis, slot):
        """Open connection to AWG

        Args:
            chassis: Signadyne chassis number (usually 1)
            slot: Module slot in chassis

        Returns:
            Name of AWG

        Raises:
            AssertionError if connection to digitizer was unsuccessful
        """
        # Open the device, using the specified chassis and slot number
        awg_name = self.awg.getProductNameBySlot(chassis, slot)
        assert isinstance(awg_name, str), \
            f"No SD_AWG found at chassis {chassis}, slot {slot}"

        result_code = self.awg.openWithSlot(awg_name, chassis, slot)
        assert result_code > 0, f'Could not open SD_AWG error code {error_code}'


    def off(self):
        """
        Stops the AWGs and sets the waveform of all channels to 'No Signal'
        """

        for i in range(self.n_channels):
            self.awg_stop(i)
            self.set_channel_wave_shape(wave_shape=0, channel_number=i)

    @with_error_check
    def reset_clock_phase(self,
                          trigger_behaviour: int,
                          trigger_source: int,
                          skew: float = 0.0):
        """Sets the module in a sync state, waiting for the first trigger to
        reset the phase of the internal clocks CLKsync and CLKsys

        Args:
            trigger_behaviour: value indicating the trigger behaviour
                Active High     :   1
                Active Low      :   2
                Rising Edge     :   3
                Falling Edge    :   4

            trigger_source: value indicating external trigger source
                External I/O Trigger    :   0
                PXI Trigger [0..n]      :   4000+n

            skew: the skew between PXI_CLK10 and CLKsync in multiples of 10ns
        """
        return self.awg.clockResetPhase(trigger_behaviour, trigger_source, skew)

    @with_error_check
    def reset_phase_channels(self, channel_mask: int):
        """
        Resets the accumulated phase of the selected channels simultaneously.

        Args:
            channel_mask: Mask to select the channel to reset.
                (LSB is channel 0, bit 1 is channel 1 etc.)

        Example:
            reset_multiple_channel_phase(5) would reset the phase of channel 0 and 2
        """
        # TODO rid mask
        return self.awg.channelPhaseResetMultiple(channel_mask)

    @with_error_check
    def config_clock_io(self, clock_config):
        """
        Configures the operation of the clock output connector (CLK)

        Args:
            clock_config: clock connector function
                Disable       : 0   The CLK connector is disabled
                CLKref Output : 1   A copy of the reference clock is available
                                    at the CLK connector
        """
        return self.awg.clockIOconfig(clock_config)

    #
    # Waveform related functions
    #

    @with_error_check
    def load_waveform(self,
                      waveform_object: keysightSD1.SD_Wave,
                      waveform_number: int) -> int:
        """Loads the specified waveform SD_Wave into the module onboard RAM.

        Waveforms must be created first as an instance of the SD_Wave class.
        To load a waveform directly from an array, use load_waveform_array.
        Not sure which method should be preferred.

        Args:
            waveform_object: pointer to the waveform object
            waveform_number: waveform number to identify the waveform
                in subsequent related function calls.

        Returns:
            availableRAM: available onboard RAM in waveform points,
                or negative numbers for errors
        """
        return self.awg.waveformLoad(waveform_object, waveform_number)

    @with_error_check
    def load_waveform_array(self,
                            waveform_type: int,
                            data_raw: np.ndarray,
                            waveform_number: int) -> int:
        """Loads the specified waveform array into the module onboard RAM.

        Alternatively, load_waveform can be used, which requires an SD_Wave
        waveform object. Not sure which method should be preferred.

        Args:
            waveform_type: waveform type
            data_raw: array with waveform points
            waveform_number: waveform number to identify the waveform
                in subsequent related function calls.

        Returns:
            availableRAM: available onboard RAM in waveform points,
                or negative numbers for errors
        """
        return self.awg.waveformLoadInt16(waveform_type, data_raw, waveform_number)

    @with_error_check
    def reload_waveform(self,
                        waveform_object: keysightSD1.SD_Wave,
                        waveform_number: int,
                        padding_mode: int = 0):
        """Replaces a waveform located in the module onboard RAM.

        The size of the new waveform must be smaller than or
        equal to the existing waveform.

        Args:
            waveform_object: pointer to the waveform object
            waveform_number: waveform number to identify the waveform
                in subsequent related function calls.
            padding_mode:
                0:  the waveform is loaded as it is, zeros are added at the
                    end if the number of points is not a multiple of the number
                    required by the AWG.
                1:  the waveform is loaded n times (using DMA) until the total
                    number of points is multiple of the number required by the
                    AWG. (only works for waveforms with even number of points)

        Returns:
            availableRAM: available onboard RAM in waveform points,
            or negative numbers for errors
        """
        return self.awg.waveformReLoad(waveform_object, waveform_number, padding_mode)

    @with_error_check
    def reload_waveform_int16(self,
                              waveform_type: int,
                              data_raw: np.ndarray,
                              waveform_number: int,
                              padding_mode: int = 0):
        """Replaces a waveform located in the module onboard RAM.

        The size of the new waveform must be smaller than or
        equal to the existing waveform.

        Args:
            waveform_type: waveform type
            data_raw: array with waveform points
            waveform_number: waveform number to identify the waveform
                in subsequent related funcion calls.
            padding_mode:
                0:  the waveform is loaded as it is, zeros are added at the
                    end if the number of points is not a multiple of the number
                    required by the AWG.
                1:  the waveform is loaded n times (using DMA) until the total
                    number of points is multiple of the number required by the
                    AWG. (only works for waveforms with even number of points)

        Returns:
            availableRAM: available onboard RAM in waveform points,
                or negative numbers for errors
        """
        return self.awg.waveformReLoadArrayInt16(waveform_type, data_raw,
                                                 waveform_number, padding_mode)

    @with_error_check
    def flush_waveforms(self):
        """Deletes all waveforms from the module onboard RAM and flushes all
        the AWG queues.
        """
        return self.awg.waveformFlush()

    #
    # AWG related functions
    #

    def start_channels(self, awg_mask):
        """Starts the selected AWGs from the beginning of their queues.

        The generation will start immediately or when a trigger is received,
        depending on the trigger selection of the first waveform in their queues
        and provided that at least one waveform is queued in these AWGs.

        Args:
            awg_mask (int): Mask to select the awgs to start (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGstartMultiple(awg_mask)

    def pause_channels(self, awg_mask):
        """Pauses the selected AWGs, leaving the last waveform point at the
        output, and ignoring all incoming triggers.
        The waveform generation can be resumed calling awg_resume_multiple

        Args:
            awg_mask (int): Mask to select the awgs to pause (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGpauseMultiple(awg_mask)

    def resume_channels(self, awg_mask):
        """
        Resumes the selected AWGs, from the current positions of their respective queue.

        Args:
            awg_mask (int): Mask to select the awgs to resume (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGresumeMultiple(awg_mask)

    def stop_channels(self, awg_mask):
        """Stops the selected AWGs, setting their output to zero and resetting
        their AWG queues to the initial positions.

        All following incoming triggers are ignored.

        Args:
            awg_mask (int): Mask to select the awgs to stop (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGstopMultiple(awg_mask)


    def trigger_channels(self, awg_mask):
        """Triggers the selected AWGs.

        The waveform waiting in the current position of the queue is launched,
        provided it is configured with VI/HVI Trigger.

        Args:
            awg_mask (int): Mask to select the awgs to be triggered (LSB is awg 0, bit 1 is awg 1 etc.)
        """
        self.awg.AWGtriggerMultiple(awg_mask)



    #
    # Functions related to creation of SD_Wave objects
    #

    @staticmethod
    def new_waveform_from_file(waveform_file):
        """
        Creates a SD_Wave object from data points contained in a file.
        This waveform object is stored in the PC RAM, not in the onboard RAM.

        Args:
            waveform_file (str): file containing the waveform points

        Returns:
            waveform (SD_Wave): pointer to the waveform object,
            or negative numbers for errors
        """
        wave = keysightSD1.SD_Wave()
        result = wave.newFromFile(waveform_file)
        result_parser(result)
        return wave

    @staticmethod
    def new_waveform_from_double(waveform_type, waveform_data_a, waveform_data_b=None):
        """
        Creates a SD_Wave object from data points contained in an array.
        This waveform object is stored in the PC RAM, not in the onboard RAM.

        Args:
            waveform_type (int): waveform type
            waveform_data_a (array): array of (float) with waveform points
            waveform_data_b (array): array of (float) with waveform points,
                                     only for the waveforms which have a second component

        Returns:
            waveform (SD_Wave): pointer to the waveform object,
            or negative numbers for errors
        """
        wave = keysightSD1.SD_Wave()
        result = wave.newFromArrayDouble(waveform_type, waveform_data_a, waveform_data_b)
        # Do not parse result because the result integer may overflow to a
        # negative number
        return wave

    @staticmethod
    def new_waveform_from_int(waveform_type, waveform_data_a, waveform_data_b=None):
        """
        Creates a SD_Wave object from data points contained in an array.
        This waveform object is stored in the PC RAM, not in the onboard RAM.

        Args:
            waveform_type (int): waveform type
            waveform_data_a (array): array of (int) with waveform points
            waveform_data_b (array): array of (int) with waveform points,
                                     only for the waveforms which have a second component

        Returns:
            waveform (SD_Wave): pointer to the waveform object,
            or negative numbers for errors
        """
        wave = keysightSD1.SD_Wave()
        result = wave.newFromArrayInteger(waveform_type, waveform_data_a, waveform_data_b)
        result_parser(result)
        return wave

    @staticmethod
    def get_waveform_status(waveform, verbose=False):
        value = waveform.getStatus()
        value_name = 'waveform_status'
        return result_parser(value, value_name, verbose)

    @staticmethod
    def get_waveform_type(waveform, verbose=False):
        value = waveform.getType()
        value_name = 'waveform_type'
        return result_parser(value, value_name, verbose)
