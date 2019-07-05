from time import time
import numpy as np
import logging

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import MultiParameter
from qcodes.utils import validators as vals


logger = logging.getLogger(__name__)


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
    def __init__(self, name, digitizer, **kwargs):
        super().__init__(name, **kwargs)
        self.digitizer = digitizer

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
        'start_channels' is called
        """
        raise NotImplementedError(
            'This method should be implemented in a subclass')

    def pre_acquire(self):
        """
        This method is called immediately after 'start_channels' is called
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


class Triggered_Controller(AcquisitionController):
    def __init__(self, name: str, digitizer: Instrument, **kwargs):
        """
        Initialises a generic Keysight digitizer and its parameters

        Args:
            name: Acquisition controller name
            digitizer: Keysight digitizer instrument
        """
        super().__init__(name, digitizer, **kwargs)

        self.add_parameter(
            'average_mode',
            set_cmd=None,
            initial_value='none',
            vals=vals.Enum('none', 'point', 'trace'),
            docstring='The averaging mode used for acquisition, either none, point or trace'
        )

        self.add_parameter(
            'channel_selection',
            set_cmd=None,
            vals=vals.Lists(),
            docstring='The list of channels on which to acquire data.'
        )

        # Set_cmds are lambda to ensure current active_channels is used
        self.add_parameter(
            'sample_rate',
            vals=vals.Numbers(min_value=1),
            set_parser=self._sample_rate_to_prescaler,
            set_cmd=lambda prescaler: self.active_channels.prescaler(prescaler),
            docstring='Sets the sample rate for all channels. The sample rate '
                      'will be converted to a prescaler, which must be an int, '
                      'and could thus result in a modified sample rate'
        )

        self.add_parameter(
            'trigger_channel',
            vals=vals.Enum('trig_in', *[f'ch{k}' for k in range(8)],
                           *[f'pxi{k}' for k in range(8)]),
            set_cmd=self.set_trigger_channel,
            docstring='The channel on which acquisition is triggered.'
        )

        self.add_parameter(
            'analog_trigger_edge',
            vals=vals.Enum('rising', 'falling', 'both'),
            initial_value='rising',
            set_cmd=lambda edge: self.active_channels.analog_trigger_edge(edge),
            docstring='Sets the trigger edge sensitivity for the active acquisition controller.'
        )

        self.add_parameter(
            'analog_trigger_threshold',
            vals=vals.Numbers(-3, 3),
            initial_value=1,
            set_cmd=lambda threshold: self.active_channels.analog_trigger_threshold(threshold),
            docstring=f'the value in volts for the trigger threshold'
        )

        self.add_parameter(
            'digital_trigger_mode',
            vals=vals.Enum('active_high', 'active_low', 'rising', 'falling'),
            initial_value='rising',
            set_cmd=lambda mode: self.active_channels.digital_trigger_mode(mode),
            docstring='Sets the digital trigger mode for the active trigger channel.'
        )

        self.add_parameter(
            'trigger_delay_samples',
            vals=vals.Numbers(),
            initial_value=0,
            set_parser=int,
            set_cmd=lambda delay: self.active_channels.trigger_delay_samples(delay),
            docstring='Sets the trigger delay before starting acquisition.'
        )

        self.add_parameter(
            'samples_per_trace',
            vals=vals.Multiples(divisor=2, min_value=2),
            set_parser=lambda val: int(round(val)),
            set_cmd=lambda samples: self.active_channels.points_per_cycle(samples),
            docstring='The number of points to capture per trace.'
        )

        self.add_parameter(
            'traces_per_acquisition',
            vals=vals.Numbers(min_value=1),
            set_parser=lambda val: int(round(val)),
            set_cmd=lambda n_cycles: self.active_channels.n_cycles(n_cycles),
            docstring='The number of traces to capture per acquisition. '
                      'Must be set after channel_selection is modified.'
        )

        self.add_parameter(
            'traces_per_read',
            set_cmd=None,
            vals=vals.Numbers(min_value=1),
            set_parser=lambda val: int(round(val)),
            docstring='The number of traces to get per read. '
                      'Can be use to break acquisition into multiple reads.'
        )

        self.add_parameter(
            'timeout',
            vals=vals.Numbers(min_value=0),
            set_cmd=None,
            unit='s',
            docstring='The maximum time (s) spent trying to read a single channel. '
                      'An acquisition request is sent every timeout_interval. '
                      'This must be set after channel_selection is modified.'
        )
        self.add_parameter(
            'timeout_interval',
            vals=vals.Numbers(min_value=0),
            set_cmd=lambda timeout: self.active_channels.timeout(timeout),
            unit='s',
            docstring='The maximum time (s) spent trying to read a single channel. '
                      'This must be set after channel_selection is modified.'
        )
        self.buffers = {}
        self.is_acquiring = False

    @property
    def _trigger_channel(self):
        if self._trigger_channel().startswith('ch'):
            return self.digitizer.channels[self.trigger_channel()]
        else:
            return None

    @property
    def active_channels(self):
        return self.digitizer.channels[self.channel_selection()]

    def set_trigger_channel(self, trigger_channel: str):
        """
        Sets the source channel with which to trigger acquisition on.

        Also ensures the trigger edge and trigger threshold are updated

        Args:
            trigger_channel: the number of the trigger channel
        """
        if trigger_channel == 'trig_in':
            self.digitizer.trigger_direction('out')
            self.active_channels.trigger_mode('digital')
            self.active_channels.digital_trigger_source('trig_in')
            self.active_channels.digital_trigger_mode(self.digital_trigger_mode())
        elif trigger_channel.startswith('pxi'):
            self.active_channels.trigger_mode('digital')
            self.active_channels.digital_trigger_source(trigger_channel)
            self.active_channels.digital_trigger_mode(self.digital_trigger_mode())
        else:  # Analog channel
            self.active_channels.trigger_mode('analog')
            trigger_id = int(trigger_channel[-1])
            self.active_channels.analog_trigger_mask(1 << trigger_id)
            # Explicitly save val to ensure trigger_edge and threshold work.
            self.trigger_channel._save_val(trigger_channel)
            self.analog_trigger_edge(self.analog_trigger_edge())
            self.analog_trigger_threshold(self.analog_trigger_threshold())

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

                channel_data = []
                while time() - t0 < self.timeout():
                    channel_data_read = channel.read()
                    if len(channel_data_read):
                        channel_data = np.append(channel_data, channel_data_read)
                        # adjust next number of acquisition points
                        channel.n_points(samples_to_get_even - len(channel_data))
                        assert len(channel_data) <= samples_to_get_even, 'For some reason got more samples than requested.'
                        if len(channel_data) == samples_to_get_even:
                            break
                        # assert len(channel_data) == samples_to_get_even, 'For some reason did not get the right number of samples.' \
                        #                                                  'Try increasing the timeout interval.'

                    else:
                        logger.warning('No data acquired within timeout interval.'
                                       ' SD_DIG.minimum_timeout_interval should'
                                       ' be increased.')
                else:
                    raise RuntimeError(f'Failed to acquire {samples_to_get_even} samples, '
                                       f'got {len(channel_data)} on ch{ch}. '
                                       f'Timeout {self.timeout():.3f}s')

                channel_data = np.array(channel_data)

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
        self.digitizer.start_channels(self.channel_selection())

    def do_acquisition(self):
        """
        Performs an acquisition using the acquisition settings
        Returns:
            records : a numpy array of channelized data
        """
        try:
            assert not self.is_acquiring, "Digitizer is already acquiring"
            self.is_acquiring = True
            self.pre_start_capture()
            self.start()
            self.pre_acquire()
            data = self.acquire()
        finally:
            self.is_acquiring = False

        return self.post_acquire(data)

    def pre_start_capture(self):
        """
        Use this method to prepare yourself for the data acquisition
        The Keysight instrument will call this method right before
        'start_channels' is called
        """
        self.digitizer.stop_channels(self.channel_selection())
        self.digitizer.flush_channels(self.channel_selection())

    def pre_acquire(self):
        """
        This method is called immediately after 'start_channels' is called
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
            data = {ch: np.mean(buffers[ch], axis=0) for ch in self.channel_selection()}
        elif self.average_mode() == 'point':
            data = {ch: np.mean(buffers[ch]) for ch in self.channel_selection()}
        else:
            raise NotImplementedError(f'Average mode {self.average_mode()} '
                                      f'not implemented')
        return data

    def _sample_rate_to_prescaler(self, sample_rate: float, tolerance=0.1):
        """Converts a sample rate to a prescaler.

            The actual sample rate may be different, since the prescaler must be
            an integer. A warning is raised if the relative mismatch is larger
            than the tolerance.

            Args:
                sample_rate: Sample rate to convert to a prescaler.
                tolerance: Maximum relative mismatch between target sample rate
                    and actual sample rate. A warning is raised if not satisfied
        """
        system_frequency = self.digitizer.system_frequency()
        prescaler = round(system_frequency/sample_rate - 1)
        real_rate = system_frequency/(round(prescaler)+1)
        if abs(sample_rate - real_rate)/sample_rate > tolerance:
            logger.warning('The chosen sample rate deviates by more than 10% from '
                           'the closest achievable rate, real sample rate will '
                           f'be {real_rate}')
        return prescaler

    def __call__(self, *args, **kwargs):
        return "Triggered"


class KeysightAcquisitionParameter(MultiParameter):
    def __init__(self, acquisition_controller: AcquisitionController, **kwargs):
        self.acquisition_controller = acquisition_controller
        super().__init__(snapshot_value=False,
                         names=[''], shapes=[()], **kwargs)

    @property
    def names(self):
        try:
            return tuple([f'ch{ch}_signal' for ch in
                          self.acquisition_controller.channel_selection()])
        except AttributeError:
            return []

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
                shape = (self.acquisition_controller.samples_per_trace(),)
            else:
                shape = (self.acquisition_controller.traces_per_acquisition(),
                         self.acquisition_controller.samples_per_trace())
            return tuple([shape] * len(self.acquisition_controller.channel_selection()))
        else:
            return tuple(() * len(self.names))

    @shapes.setter
    def shapes(self, shapes):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    def get_raw(self):
        data = self.acquisition_controller.do_acquisition()
        return [data[ch] for ch in self.acquisition_controller.channel_selection()]
