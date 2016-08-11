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

        # Get device object
        self._device = self._api.DeviceSet().DeviceList[0]

        self.initialize()
        self._channels = [self._device.GetChannel(k) for k in range(4)]

        # Initialize waveforms and sequences
        self._waveforms = [[]]*4
        self._sequences = [[]]*4

        for ch in range(1,5):
            self.add_parameter('ch{}_trigger_mode'.format(ch),
                               label='Trigger mode',
                               set_cmd=partial(self._set_trigger_mode, ch),
                               vals=vals.Strings())

            self.add_parameter('ch{}_sequence'.format(ch),
                               parameter_class=ManualParameter,
                               label='Sequence',
                               vals=vals.Anything()) # Can we test for an (int, int) tuple list?

            self.add_function('ch{}_set_internal_trigger'.format(ch),
                              call_cmd=self._channels[ch-1].SetInternalTrigger)

            self.add_function('ch{}_add_waveform'.format(ch),
                              call_cmd=self._waveforms[ch-1].append,
                              args=[vals.Anything()]) # Can we test for a float list/array?

            self.add_function('ch{}_clear_waveforms'.format(ch),
                              call_cmd=self._waveforms[ch-1].clear)

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

    def load_waveforms(self):
        for ch, channel in enumerate(self._channels):
            channel_waveforms = self._waveforms[ch]
            # Initialize array of waves
            waves = Array.CreateInstance(self._api.WaveformStruct,len(channel_waveforms))
            # We have to create separate wave instances and load them into the waves array one by one
            for k, waveform in enumerate(channel_waveforms):
                wave = self._api.WaveformStruct()
                wave.Sample = waveform
                waves[k] = wave
            return_msg = channel.LoadWaveforms(waves)
            assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS,\
                "Loading waveforms Error: {}".format(return_msg.ErrorDescription)

    def load_sequence(self):
        for ch, channel in enumerate(self._channels):
            channel_sequence = self._sequences[ch]
            sequence = Array.CreateInstance(self._api.GenerationSequenceStruct,len(channel_sequence))
            for k, subsequence_info in enumerate(channel_sequence):
                subsequence = self._api.GenerationSequenceStruct()
                if isinstance(subsequence_info, int):
                    subsequence.WaveFormIndex = subsequence_info
                    # Set repetitions to 1 (default) if subsequence info is an int
                    subsequence.Repetitions = subsequence_info[1]
                elif isinstance(subsequence_info, tuple):
                    assert len(subsequence_info) == 2, \
                        'A subsequence tuple must be of the form (WaveformIndex, Repetitions)'
                    subsequence.WaveFormIndex = subsequence_info[0]
                    subsequence.Repetitions = subsequence_info[1]
                sequence[k] = subsequence

            # Set transfermode to USB (seems to be a fixed function)
            trans = Array.CreateInstance(self._api.TransferMode, 1)
            return_msg = channel.LoadGenerationSequence(sequence, trans[0], True)
            assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS, \
                "Loading sequence Error: {}".format(return_msg.ErrorDescription);

    def run(self, channels=[1, 2, 3, 4]):
        """
        Run sequences on given channels
        Args:
            channels: List of channels to run (default all)

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
            channels: List of channels to stop (default all)

        Returns:
            None
        """
        return_msg = self._device.STOP()
        assert return_msg.ErrorSource == self._api.ErrorCodes.RES_SUCCESS,\
            "Stopping ArbStudio error: {}".format(return_msg.ErrorDescription)
