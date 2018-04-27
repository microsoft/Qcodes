from time import time
import numpy as np

# TODO remove asterisk
from .SD_DIG import *
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import MultiParameter
from qcodes.utils.validators import Numbers, Multiples


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
        super().__init__(name, keysight_name, **kwargs)

        self.add_parameter(
            'average_mode',
            set_cmd=None,
            initial_value='none',
            vals=Enum('none', 'point', 'trace'),
            docstring='The averaging mode used for acquisition, either none, point or trace'
        )

        self.add_parameter(
            'channel_selection',
            set_cmd=None,
            vals=Anything(),
            docstring='The list of channels on which to acquire data.'
        )

        self.add_parameter(
            'trigger_channel',
            vals=Enum('trig_in', *[f'ch{k}' for k in range(8)],
                      *[f'pxi{k}' for k in range(1, 9)]),
            set_cmd=self.set_trigger_channel,
            docstring='The channel on which acquisition is triggered.'
        )

        self.add_parameter(
            'sample_rate',
            vals=Numbers(min_value=1),
            set_parser=lambda val: int(round(val)),
            set_cmd=self._set_all_prescalers,
            docstring='Sets the sample rate for all channels.'
        )

        self.add_parameter(
            'trigger_edge',
            vals=Enum('rising', 'falling', 'both'),
            set_cmd=lambda edge: self._keysight.channels(self.trigger_channel).trigger_edge(edge),
            docstring='Sets the trigger edge sensitivity for the active acquisition controller.'
        )

        # TODO: Convert samples to duration
        self.add_parameter(
            'trigger_delay_samples',
            vals=Numbers(),
            initial_value=0,
            set_cmd=lambda delay: self.active_channels.trigger_delay(delay),
            docstring='Sets the trigger delay before starting acquisition.'
        )


        self.add_parameter(
            'samples_per_trace',
            vals=Multiples(divisor=2, min_value=2),
            set_parser=lambda val: int(round(val)),
            set_cmd=lambda n_points: self.active_channels.points_per_cycle(n_points),
            docstring='The number of points to capture per trace.'
        )

        self.add_parameter(
            'traces_per_acquisition',
            vals=Numbers(min_value=1),
            set_parser=lambda val: int(round(val)),
            set_cmd=self._set_all_n_cycles,
            docstring='The number of traces to capture per acquisition.'
        )

        self.add_parameter(
            'traces_per_read',
            set_cmd=None,
            vals=Numbers(min_value=1),
            set_parser=lambda val: int(round(val)),
            # set_cmd=self._set_all_n_points,
            docstring='The number of traces to get per read. '
                     'Can be use to break acquisition into multiple reads.'
        )

        self.add_parameter(
            'timeout',
            vals=Numbers(min_value=0),
            set_cmd=None,
            unit='s',
            docstring='The maximum time (s) spent trying to read a single channel.'
                      'An acquisition request is sent every timeout_interval.'
        )
        self.add_parameter(
            'timeout_interval',
            vals=Numbers(min_value=0),
            set_cmd=self._set_all_timeout,
            unit='s',
            docstring='The maximum time (s) spent trying to read a single channel.'
        )

        # Set all channels to trigger by hardware
        for ch in range(8):
            self._keysight.parameters[f'DAQ_trigger_mode_{ch}'].set(3)

        self.buffers = {}

    @property
    def trigger_threshold(self):
        return self._keysight.parameters['trigger_threshold_{}'.format(
                                         self.trigger_channel.get_latest())]

    @property
    def active_channels(self):
        return self.keysight.channels(self.channel_selection())

    def set_trigger_channel(self, trigger_channel):
        """
        Sets the source channel with which to trigger acquisition on.

        Args:
            trigger_channel (int)   : the number of the trigger channel
        """
        if trigger_channel == 'trig_in':
            raise NotImplementedError()
            # digital_trigger_source
            # DAQdigitaltriggerconfig
            # triggerIOconfig
        elif trigger_channel.startswith('pxi'):
            raise NotImplementedError()
        else: # Analog channel
            self.active_channels.analog_trigger_mask(1 << trigger_channel)

        # TODO: check if this works
        # Ensure latest trigger edge setting for the current trigger channel
        self.trigger_channel._save_val(trigger_channel)
        self.trigger_edge(self.trigger_edge())
        self.trigger_threshold(self.trigger_threshold())

    def acquire(self):
        # Initialize record of acquisition times
        self.acquisition_times = [[] for ch in self.channel_selection()]
        self.last_acquisition_time = time()

        self.buffers = {ch: np.zeros((self.traces_per_acquisition(),
                                      self.samples_per_trace()))
                        for ch in self.channel_selection()}

        # We loop over channels first, even though it would make more sense to
        # First loop over read iteration. The latter method gave two issues:
        # scrambling of data between channels, and only getting data after
        # timeout interval passed, resulting in very slow acquisitions.
        for k, channel in enumerate(self.active_channels):
            ch = channel.id
            acquired_traces = 0
            while acquired_traces < self.traces_per_acquisition():
                traces_to_get = min(self.traces_per_read(),
                                    self.traces_per_acquisition() - acquired_traces)

                samples_to_get = traces_to_get * self.samples_per_trace()
                # We need an even number of samples per read
                samples_to_get_even = samples_to_get + samples_to_get % 2

                channel.n_points(samples_to_get_even)
                logger.debug(f'Acquiring {samples_to_get} points from DAQ{ch}.')

                t0 = time()
                while time() - t0 < self.timeout():
                    channel_data = self._keysight.daq_read(ch)
                    if len(channel_data):
                        break
                    else:
                        logger.warning('No data acquired within timeout interval.'
                                       ' SD_DIG.minimum_timeout_interval should'
                                       ' be increased.')
                else:
                    raise RuntimeError(f'Could not acquire data on ch{ch}. '
                                       f'Timeout {self.timeout():.3f}s')

                # Record acquisition time
                self.acquisition_times[k].append(time() - self.last_acquisition_time)
                self.last_acquisition_time = time()

                if samples_to_get % 2:
                    # Remove final point needed to make samples per read even
                    channel_data = channel_data[:-1]
                # Segment read data to 2D array of traces
                read_traces = channel_data.reshape((traces_to_get,
                                                    self.samples_per_trace()))
                self.buffers[ch][acquired_traces:acquired_traces+traces_to_get] = read_traces

                acquired_traces += traces_to_get

        return self.buffers

    def start(self):
        channel_mask = self._ch_array_to_mask(self.channel_selection)
        self._keysight.daq_start_multiple(channel_mask)

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
        # n_points = self.traces_per_acquisition.get_latest() * \
        #            self.samples_per_trace.get_latest()
        # for ch in self.channel_selection():
        #     self._keysight.parameters['n_points_{}'.format(ch)].set(n_points)
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
            warnings.warn('The chosen sample rate deviates by more than 10% from the closest achievable rate,\
                           real sample rate will be {}'.format(real_rate))
        self.active_channels.prescaler(int(prescaler))

    def _set_all_n_points(self, n_points):
        """
        This method sets the channelised parameters for data acquisition
        all at once. This must be set after channel_selection is modified.

        Args:
            n_points (int)  : the number of points to read per daq_read call

        """
        self.active_channels.n_points(n_points)

    def _set_all_trigger_delay(self, n_points):
        """
        This method sets the channelised parameters for data acquisition
        all at once. This must be set after channel_selection is modified.

        Args:
            n_points (int)  : the number of points to read per daq_read call

        """
        for ch in self.channel_selection():
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

    def _set_all_timeout(self, timeout):
        """
        This method sets the channelised parameters for data acquisition
        all at once. This must be set after channel_selection is modified.

        Args:
            timeout (float)  : the maximum time (s) to wait for single channel read.
        """
        if timeout == 0:
            warnings.warn(f'Timeout of {timeout} s is too small, '
                          f'setting to 0 which disables timeout')
        for ch in self.channel_selection():
            self._keysight.parameters[f'timeout_{ch}'](timeout)

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
                          self.acquisition_controller.channel_selection()])

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
                shape = (self.acquisition_controller.samples_per_trace.get_latest(),)
            else:
                shape = (self.acquisition_controller.traces_per_acquisition.get_latest(),
                         self.acquisition_controller.samples_per_trace.get_latest())
            return tuple([shape] * len(self.acquisition_controller.channel_selection))
        else:
            return tuple(() * len(self.names))

    @shapes.setter
    def shapes(self, shapes):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    def get(self):
        data = self.acquisition_controller.do_acquisition()
        return [data[ch] for ch in self.acquisition_controller.channel_selection()]
