import os
import clr  # Import pythonnet to talk to dll
from System import Array
from time import sleep
from functools import partial
from qcodes import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals




class ArbStudio1104(Instrument):
    def __init__(self, name, dll_path, **kwargs):
        super().__init__(name, **kwargs)

        # Add .NET assembly to pythonnet
        # This allows importing its functions
        clr.System.Reflection.Assembly.LoadFile(dll_path)
        from clr import ActiveTechnologies
        self._api = ActiveTechnologies.Instruments.AWG4000.Control

        # Instrument constants (set here since api is only defined above)
        self._trigger_sources = {'stop': self._api.TriggerSource.Stop,
                                 'start': self._api.TriggerSource.Start,
                                 'event_marker': self._api.TriggerSource.Event_Marker,
                                 'dc_trigger_in': self._api.TriggerSource.DCTriggerIN,
                                 'fp_trigger_in': self._api.TriggerSource.FPTriggerIN}
        self._trigger_sensitivity_edges = {'rising': self._api.SensitivityEdge.RisingEdge,
                                           'falling': self._api.SensitivityEdge.RisingEdge}
        self._trigger_actions= {'start': self._api.TriggerAction.TriggerStart,
                                'stop': self._api.TriggerAction.TriggerStop,
                                'ignore': self._api.TriggerAction.TriggerIgnore}

        # Get device object
        self._device = self._api.DeviceSet().DeviceList[0]

        self.initialize()
        self._channels = [self._device.GetChannel(k) for k in range(4)]

        # Initialize waveforms and sequences
        self._waveforms = [[] for k in range(4)]

        for ch in range(1,5):
            self.add_parameter('ch{}_trigger_mode'.format(ch),
                               label='Channel {} trigger mode'.format(ch),
                               set_cmd=partial(self._set_trigger_mode, ch),
                               vals=vals.Strings())

            self.add_parameter('ch{}_trigger_source'.format(ch),
                               label='Channel {} trigger source'.format(ch),
                               set_cmd=partial(self._set_trigger_source, ch),
                               vals=vals.Enum('stop', 'start', 'event_marker', 'dc_trigger_in',
                                              'fp_trigger_in'))

            self.add_parameter('ch{}_samplig_rate_prescaler'.format(ch),
                               label='Channel {} sampling rate prescaler'.format(ch),
                               get_cmd=partial(self._get_sampling_rate_prescaler, ch), #Typo is intentional
                               set_cmd=partial(self._set_sampling_rate_prescaler, ch),
                               vals=vals.MultiType(Multiples(2), vals.Enum(1)))

            self.add_parameter('ch{}_sequence'.format(ch),
                               parameter_class=ManualParameter,
                               label='Channel {} Sequence'.format(ch),
                               initial_value=[],
                               vals=vals.Anything()) # Can we test for an (int, int) tuple list?

            self.add_function('ch{}_add_waveform'.format(ch),
                              call_cmd=partial(self._add_waveform, ch),
                              args=[vals.Anything()]) # Can we test for a float list/array?

            self.add_function('ch{}_clear_waveforms'.format(ch),
                              call_cmd=self._waveforms[ch-1].clear)

        self.add_parameter('max_voltage',
                           parameter_class=ManualParameter,
                           label='Maximum waveform voltage',
                           units='V',
                           initial_value=2.5,
                           vals=vals.Numbers())  # Can we test

        # TODO Need to implement frequency interpolation for channel pairs
        # self.add_parameter('frequency_interpolation',
        #                    label='DAC frequency interpolation factor',
        #                    vals=vals.Enum(1, 2, 4))

        self.add_parameter('trigger_sensitivity_edge',
                           parameter_class=ManualParameter,
                           initial_value='rising',
                           label='Trigger sensitivity edge for in/out',
                           vals=vals.Enum('rising', 'falling'))

        self.add_parameter('trigger_action',
                           parameter_class=ManualParameter,
                           initial_value='start',
                           label='Trigger action',
                           vals=vals.Enum('start', 'stop', 'ignore'))
    def initialize(self):
        # Create empty array of four channels.
        # These are only necessary for initialization
        channels = Array.CreateInstance(self._api.Functionality, 4)
        # Initialize each of the channels
        channels[0] = self._api.Functionality.ARB
        channels[1] = self._api.Functionality.ARB
        channels[2] = self._api.Functionality.ARB
        channels[3] = self._api.Functionality.ARB

        # Initialise ArbStudio
        return_msg = self._device.Initialize(channels)
        assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS, \
            "Error initializing Arb: {}".format(return_msg.ErrorDescription)

    def _get_sampling_rate_prescaler(self, ch):
        return self._channels[ch - 1].SampligRatePrescaler

    def _set_sampling_rate_prescaler(self, ch, prescaler):
        self._channels[ch - 1].SampligRatePrescaler = prescaler

    def _set_trigger_mode(self, ch, trigger_mode_string):
        #Create dictionary with possible TriggerMode objects
        trigger_modes = {'single': self._api.TriggerMode.Single,
                         'continuous': self._api.TriggerMode.Continuous,
                         'stepped': self._api.TriggerMode.Stepped,
                         'burst': self._api.TriggerMode.Burst}

        #Transform trigger mode to lowercase, such that letter case does not matter
        trigger_mode_string = trigger_mode_string.lower()
        trigger_mode = trigger_modes[trigger_mode_string]
        return_msg = self._channels[ch-1].SetTriggerMode(trigger_mode)
        assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS, \
            "Error setting Arb channel {} trigger mode: {}".format(ch, return_msg.ErrorDescription)

    def _set_trigger_source(self, ch, trigger_source_str):
        if trigger_source_str == 'internal':
            return_msg = self._channels[ch-1].SetInternalTrigger()
        else:
            # Collect external trigger arguments
            trigger_source = self._trigger_sources[trigger_source_str]
            trigger_sensitivity_edge = self._trigger_sensitivity_edges[self.trigger_sensitivity_edge()]
            trigger_action = self._trigger_actions[self.trigger_action()]

            return_msg = self._channels[ch-1].SetExternalTrigger(trigger_source,
                                                                 trigger_sensitivity_edge,
                                                                 trigger_action)

        assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS, \
            "Error setting Arb channel {} trigger source to {}: {}".format(ch, trigger_source_str,
                                                                           return_msg.ErrorDescription)

    def _add_waveform(self, channel, waveform):
        assert len(waveform)%2 == 0, 'Waveform must have an even number of points'
        assert len(waveform)> 2, 'Waveform must have at least four points'
        assert max(waveform) <= self.max_voltage(), 'Waveform may not exceed {} V'.format(self.max_voltage())
        self._waveforms[channel - 1].append(waveform)

    def load_waveforms(self, channels=[1, 2, 3, 4]):
        waveforms_list = []
        for ch, channel in enumerate(self._channels):
            waveforms_array = self._waveforms[ch]
            # Initialize array of waves
            waveforms = Array.CreateInstance(self._api.WaveformStruct,len(waveforms_array))
            # We have to create separate wave instances and load them into the waves array one by one
            for k, waveform_array in enumerate(waveforms_array):
                wave = self._api.WaveformStruct()
                wave.Sample = waveform_array
                waveforms[k] = wave
            return_msg = channel.LoadWaveforms(waveforms)
            waveforms_list.append(waveforms)
            assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS,\
                "Loading waveforms Error: {}".format(return_msg.ErrorDescription)
        return waveforms_list

    def load_sequence(self, channels=[1, 2, 3, 4]):
        sequence_list = []
        for ch in channels:
            channel = self._channels[ch-1]
            channel_sequence = eval("self.ch{}_sequence()".format(ch))
            # Initialize sequence array
            sequence = Array.CreateInstance(self._api.GenerationSequenceStruct,len(channel_sequence))
            for k, subsequence_info in enumerate(channel_sequence):
                subsequence = self._api.GenerationSequenceStruct()
                if isinstance(subsequence_info, int):
                    subsequence.WaveformIndex = subsequence_info
                    # Set repetitions to 1 (default) if subsequence info is an int
                    subsequence.Repetitions = 1
                elif isinstance(subsequence_info, tuple):
                    assert len(subsequence_info) == 2, \
                        'A subsequence tuple must be of the form (WaveformIndex, Repetitions)'
                    subsequence.WaveformIndex = subsequence_info[0]
                    subsequence.Repetitions = subsequence_info[1]
                else:
                    raise TypeError("A subsequence must be either an int or (int, int) tuple")
                sequence[k] = subsequence

            sequence_list.append(sequence)

            # Set transfermode to USB (seems to be a fixed function)
            trans = Array.CreateInstance(self._api.TransferMode, 1)
            return_msg = channel.LoadGenerationSequence(sequence, trans[0], True)
            assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS, \
                "Loading sequence Error: {}".format(return_msg.ErrorDescription);
        return sequence_list

    def run(self, channels=[1, 2, 3, 4]):
        """
        Run sequences on given channels
        Args:
            channels: List of channels to run, starting at 1 (default all)

        Returns:
            None
        """
        return_msg = self._device.RUN(channels)
        assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS,\
            "Running ArbStudio error: {}".format(return_msg.ErrorDescription)

    def stop(self, channels=[1, 2, 3, 4]):
        """
        Stop sequence on given channels
        Args:
            channels: List of channels to stop, starting at 1 (default all)

        Returns:
            None
        """
        return_msg = self._device.STOP()
        assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS,\
            "Stopping ArbStudio error: {}".format(return_msg.ErrorDescription)

        # A stop command seems to reset trigger sources. For the channels that had a trigger source,
        # this will reset it to its previoius value
        for ch in range(1,5):
            trigger_source = eval('self.ch{}_trigger_source.get_latest()'.format(ch))
            if trigger_source:
                eval("self.ch{}_trigger_source(trigger_source)".format(ch))


class Multiples(vals.Ints):
    '''
    requires an integer
    optional parameters min_value and max_value enforce
    min_value <= value <= max_value
    divisor enforces that value % divisor == 0
    '''

    def __init__(self, divisor=1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(divisor, int):
            raise TypeError('divisor must be an integer')
        self._divisor = divisor

    def validate(self, value, context=''):
        super().validate(value=value, context=context)
        if not value % self._divisor == 0:
            raise TypeError('{} is not a multiple of {}; {}'.format(
                repr(value), repr(self._divisor), context))

    def __repr__(self):
        return super().__repr__()[:-1] + ', Multiples of {}>'.format(self._divisor)