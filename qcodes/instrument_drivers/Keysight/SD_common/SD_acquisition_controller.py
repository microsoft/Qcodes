import warnings

from .SD_DIG import *
from qcodes.instrument.parameter import MultiParameter, ManualParameter
from qcodes.instrument.base import Instrument
import numpy as np
from qcodes import MultiParameter

class AcquisitionController(Instrument):
    """
    This class represents all choices that the end-user has to make regarding
    the data-acquisition. this class should be subclassed to program these
    choices.

    The basic structure of an acquisition is:

        - call to keysight internal configuration
        - call to acquisitioncontroller.pre_start_capture
        - Call to the start capture of the Keysight board
        - call to acquisitioncontroller.pre_acquire
        - return acquisitioncontroller.post_acquire

    Attributes:
        _keysight: a reference to the keysight instrument driver
    """
    def __init__(self, name, keysight_name, **kwargs):
        """
        :param keysight_name: The name of the keysight instrument on the server
        :return: nothing
        """
        super().__init__(name, **kwargs)
        self._keysight = self.find_instrument(keysight_name,
                                            instrument_class=SD_DIG)

        # Names and shapes must have initial value, even through they will be
        # overwritten in set_acquisition_settings. If we don't do this, the
        # remoteInstrument will not recognize that it returns multiple values.
        self.add_parameter(name="acquisition",
                           parameter_class=KeysightAcquisitionParameter,
                           acquisition_controller=self)

    def _get_keysight(self):
        """
        returns a reference to the keysight instrument. A call to self._keysight is
        quicker, so use that if in need for speed
        return: reference to the Keysight instrument
        """
        return self._keysight

    def do_acquisition(self):
        """
        Performs an acquisition using the acquisition settings
        Returns:
            None
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def pre_start_capture(self):
        """
        Use this method to prepare yourself for the data acquisition
        The Keysight instrument will call this method right before
        'daq_start' is called
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def pre_acquire(self):
        """
        This method is called immediately after 'daq_start' is called
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def post_acquire(self):
        """
        This method should return any information you want to save from this
        acquisition. The acquisition method from the Keysight driver will use
        this data as its own return value

        Returns:
            this function should return all relevant data that you want
            to get from the acquisition
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')
    
    def _ch_array_to_mask(self, channel_selection):
        """
        This method is a helper function to translate an array of channel numbers
        into a binary mask.

        Returns:
            a binary mask of channels
        """
        mask = 0
        for ch in channel_selection():
            mask |= 1 << ch
        return mask

class Triggered_Controller(AcquisitionController):
    def __init__(self, name, keysight_name, **kwargs):
        """ 
        Initialises a generic Keysight digitizer and its parameters

        Args:
            name (str)      : the name of the digitizer card
            channels (int)  : the number of input channels the specified card has
            triggers (int)  : the number of trigger inputs the specified card has
        """
        # Set the average mode of the device
        self._average_mode = kwargs.pop('average_mode', 'none')
        super().__init__(name, keysight_name, **kwargs)

        self.add_parameter(
            'average_mode',
            parameter_class=ManualParameter,
            initial_value=self._average_mode,
            vals=Enum('none', 'point', 'trace'),
            docstring='The averaging mode used for acquisition, either none, point or trace'
        )

        self.add_parameter(
            'channel_selection',
            parameter_class=ManualParameter,
            vals=Anything(),
            docstring='The list of channels on which to acquire data.'
        )

        self.add_parameter(
            'trigger_channel',
            vals=Enum(0, 1, 2, 3, 4, 5, 6, 7),
            set_cmd=self.set_trigger_channel,
            docstring='The channel on which acquisition is triggered.'
        )

        self.add_parameter(
            'sample_rate',
            vals=Numbers(),
            set_cmd=self._set_all_prescalers,
            docstring='Sets the sample rate for all channels.'
        )

        self.add_parameter(
            'trigger_edge',
            vals=Enum('rising', 'falling', 'both'),
            val_mapping={'rising' : 1, 'falling' : 2, 'both' : 3},
            set_cmd=self.set_trigger_edge,
            docstring='Sets the trigger edge sensitivity for the active acquisition controller.'
        )


        self.add_parameter(
            'samples_per_record',
            vals=Ints(),
            set_cmd=self._set_all_points_per_cycle,
            docstring='The number of points to capture per trace.'
        )

        self.add_parameter(
            'traces_per_acquisition',
            vals=Ints(),
            set_cmd=self._set_all_n_cycles,
            docstring='The number of traces to capture per acquisition.'
        )

        self.add_parameter(
            'read_timeout',
            vals=Ints(),
            set_cmd=self._set_all_read_timeout,
            docstring='The maximum time (s) spent trying to read a single channel.'
        )

        # Set all channels to trigger by hardware
        for ch in range(8):
            self._keysight.parameters['DAQ_trigger_mode_{}'.format(ch)].set(3)

    @property
    def trigger_threshold(self):
        return self._keysight.parameters['trigger_threshold_{}'.format(
                                         self.trigger_channel.get_latest())]

    def set_trigger_edge(self, edge):
        """
        Sets the trigger edge for the given trigger channel.

        Args:
            edge (int) : the edge to trigger on
        """
        self._keysight.parameters['trigger_edge_{}'.format(self.trigger_channel.get_latest())](edge)

    def set_trigger_channel(self, tch):
        """
        Sets the source channel with which to trigger acquisition on.

        Args:
            tch (int)   : the number of the trigger channel
        """
        for ch in self.channel_selection():
            self._keysight.parameters['analog_trigger_mask_{}'.format(ch)].set(1<<tch)
        # Ensure latest trigger edge setting for the current trigger channel
        self.trigger_channel._save_val(tch)
        self.trigger_edge(self.trigger_edge.get_latest() or 'rising')
        self.trigger_threshold(self.trigger_threshold.get_latest() or 0)

    def _get_keysight(self):
        """
        returns a reference to the Keysight instrument. A call to self._keysight is
        quicker, so use that if in need for speed

        return: reference to the Keysight instrument
        """
        return self._keysight

    def acquire(self):
        buffers = {ch: np.zeros((self.traces_per_acquisition.get_latest(),
                                      self.samples_per_record.get_latest()))
                        for ch in self.channel_selection()}

        for trace in range(self.traces_per_acquisition.get_latest()):
            for ch in self.channel_selection():
                ch_data = self._keysight.daq_read(ch)
                if (len(ch_data) != 0):
                    buffers[ch][trace] = ch_data
        return buffers

    def start(self):
        self._keysight.daq_start_multiple(self._ch_array_to_mask(self.channel_selection))

    def do_acquisition(self):
        """
        Performs an acquisition using the acquisition settings
        Returns:
            records : a numpy array of channelized data
        """
        self.pre_start_capture()
        self.start()
        self.pre_acquire()
        data = self.acquire()
        return self.post_acquire(data)

    def pre_start_capture(self):
        """
        Use this method to prepare yourself for the data acquisition
        The Keysight instrument will call this method right before
        'daq_start' is called
        """
        self._keysight.daq_stop_multiple(self._ch_array_to_mask(self.channel_selection))
        self._keysight.daq_flush_multiple(self._ch_array_to_mask(self.channel_selection))

    def pre_acquire(self):
        """
        This method is called immediately after 'daq_start' is called
        """
        pass

    def post_acquire(self, buffers):
        """
        This method should return any information you want to save from this
        acquisition. The acquisition method from the Keysight driver will use
        this data as its own return value

        Returns:
            this function should return all relevant data that you want
            to get from the acquisition
        """

        if self.average_mode() == 'none':
            data = buffers
        elif self.average_mode() == 'trace':
            data = {ch : np.mean(buffers[ch], axis=0) for ch in self.channel_selection()}
        elif self.average_mode() == 'point':
            data = {ch: np.mean(buffers[ch]) for ch in self.channel_selection()}
            
        return data

    def _set_all_prescalers(self, sample_rate):
        """
            This method sets the channelised parameters for data acquisition
            all at once. This must be set after channel_selection is modified.

            Args:
                n_points (int)  : the number of points to capture per trace
        """
        prescaler = 100e6/sample_rate - 1
        real_rate = 100e6/(round(prescaler)+1)
        if abs(sample_rate - real_rate)/sample_rate > 0.1:
            warnings.warn('The chosen sample rate deviates by more than 10% from the closest achievable rate, real sample rate will be {}'.format(real_rate))
        for ch in self.channel_selection():
            self._keysight.parameters['prescaler_{}'.format(ch)].set(int(prescaler))

    def _set_all_points_per_cycle(self, n_points):
        """
        This method sets the channelised parameters for data acquisition 
        all at once. This must be set after channel_selection is modified.

        Args:
            n_points (int)  : the number of points to capture per trace

        """
        for ch in self.channel_selection():
            self._keysight.parameters['points_per_cycle_{}'.format(ch)].set(n_points)
            self._keysight.parameters['n_points_{}'.format(ch)].set(n_points)



    def _set_all_n_cycles(self, n_cycles):
        """
        This method sets the channelised parameters for data acquisition 
        all at once. This must be set after channel_selection is modified.

        Args:
            n_cycles (int)  : the number of traces to capture

        """
        for ch in self.channel_selection():
            self._keysight.parameters['n_cycles_{}'.format(ch)].set(n_cycles)

    def _set_all_read_timeout(self, timeout):
        """
        This method sets the channelised parameters for data acquisition
        all at once. This must be set after channel_selection is modified.

        Args:
            timeout (int)  : the maximum time (ms) to wait for single channel read.
        """
        for ch in self.channel_selection():
            self._keysight.parameters['timeout_{}'.format(ch)].set(timeout)

    def __call__(self, *args, **kwargs):
        return "Triggered"

class KeysightAcquisitionParameter(MultiParameter):
    def __init__(self, acquisition_controller=None, **kwargs):
        self.acquisition_controller = acquisition_controller
        super().__init__(snapshot_value=False,
                         names=[''], shapes=[()], **kwargs)

    @property
    def names(self):
        if self.acquisition_controller is None or \
                not hasattr(self.acquisition_controller, 'channel_selection'):
            return ['']
        else:
            return tuple(['ch{}_signal'.format(ch) for ch in
                          self.acquisition_controller.channel_selection])

    @names.setter
    def names(self, names):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    @property
    def labels(self):
        return self.names

    @labels.setter
    def labels(self, labels):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    @property
    def units(self):
        return ['V'] * len(self.names)

    @units.setter
    def units(self, units):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    @property
    def shapes(self):
        if hasattr(self.acquisition_controller, 'average_mode'):
            average_mode = self.acquisition_controller.average_mode()

            if average_mode == 'point':
                shape = ()
            elif average_mode == 'trace':
                shape = (self.acquisition_controller.samples_per_record.get_latest(),)
            else:
                shape = (self.acquisition_controller.traces_per_acquisition.get_latest(),
                         self.acquisition_controller.samples_per_record.get_latest())
            return tuple([shape] * len(self.acquisition_controller.channel_selection))
        else:
            return tuple(() * len(self.names))

    @shapes.setter
    def shapes(self, shapes):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    def get(self):
        data = self.acquisition_controller.do_acquisition()
        return [data[ch] for ch in self.acquisition_controller.channel_selection]
