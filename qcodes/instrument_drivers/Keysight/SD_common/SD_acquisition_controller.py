import warnings

from .SD_DIG import *
from qcodes.instrument.base import Instrument
import numpy as np

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
        #TODO FIX finding of instrument
        self._keysight = self.find_instrument(keysight_name,
                                            instrument_class=SD_DIG)

        self._acquisition_settings = {}
        self._fixed_acquisition_settings = {}
        self.add_parameter(name="acquisition_settings",
                           get_cmd=lambda: self._acquisition_settings)

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

class Triggered_Controller(AcquisitionController):
    def __init__(self, name, chassis, slot, channels, triggers, **kwargs):
        """ Initialises a generic Keysight digitizer and its parameters

            Args:
                name (str)      : the name of the digitizer card
                channels (int)  : the number of input channels the specified card has
                triggers (int)  : the number of trigger inputs the specified card has
        """
        self.add_parameter(
            'average_mode',
            parameter_class = ManualParameter,
            initial_value='none',
            vals=Enum('none', 'point', 'trace'),
            docstring='The averaging mode used for acquisition, either none, point or trace'
        )
        # Set the average mode of the device
        self.average_mode.set(kwargs.pop('average_mode', 'none'))
        super().__init__(name, chassis, slot, **kwargs)

        self.add_parameter(
            'channel_selection',
            parameter_class = ManualParameter,
            docstring='A mask of the channels to be acquired'
        )    

        self.add_parameter(
            'samples_per_record',
            parameter_class = ManualParameter,
            vals=Int(),
            set_cmd=self._set_all_points_per_cycle,
            docstring='The number of points to capture per trace'
        )

        self.add_parameter(
            'traces_per_acquisition',
            parameter_class = ManualParameter,
            vals=Int(),
            set_cmd=self._set_all_n_cycles,
            docstring='The number of traces to capture per acquisition'
        )

        self.read_timeout = dict()
        for ch in range(channels):
            self.read_timeout[ch] = self._keysight.parameters['timeout_{}'.format(ch)]


    def _get_keysight(self):
        """
        returns a reference to the Keysight instrument. A call to self._keysight is
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
        records = self._keysight.acquire(acquisition_controller=self,
                                       **self._acquisition_settings)
        return records

#    def pre_start_capture(self):
#        """
#        Use this method to prepare yourself for the data acquisition
#        The Keysight instrument will call this method right before
#        'daq_start' is called
#        """

#    def pre_acquire(self):
#        """
#        This method is called immediately after 'daq_start' is called
#        """

    def post_acquire(self):
        """
        This method should return any information you want to save from this
        acquisition. The acquisition method from the Keysight driver will use
        this data as its own return value

        Returns:
            this function should return all relevant data that you want
            to get from the acquisition
        """
        if self.average_mode() == 'none':
            data = self.buffers
        elif self.average_mode() == 'trace':
            data = [np.mean(buf, axis=0) for buf in self.buffers]
        elif self.average_mode() == 'point':
            data = [np.mean(buf) for buf in self.buffers]

        return data

    def _set_all_points_per_cycle(self, n_points):
        """
        This method sets the channelised parameters for data acquisition 
        all at once. This must be set after channel_selection is modified.

        Args:
            n_points (int)  : the number of points to capture per trace

        """
        for ch in self.channel_selection:
            self._keysight.parameters['points_per_cycle_{}'.format(ch)].set(n_points)



    def _set_all_n_cycles(self, n_cycles):
        """
        This method sets the channelised parameters for data acquisition 
        all at once. This must be set after channel_selection is modified.

        Args:
            n_cycles (int)  : the number of traces to capture

        """
        for ch in self.channel_selection:
            self._keysight.parameters['n_cycles_{}'.format(ch)].set(n_cycles)


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
                shape = (self.acquisition_controller.samples_per_record,)
            else:
                shape = (self.acquisition_controller.traces_per_acquisition,
                         self.acquisition_controller.samples_per_record)
            return tuple([shape] * self.acquisition_controller.number_of_channels)
        else:
            return tuple(() * len(self.names))

    @shapes.setter
    def shapes(self, shapes):
        # Ignore setter since getter is extracted from acquisition controller
        pass

    def get(self):
        return self.acquisition_controller.do_acquisition()
