import warnings

from .SD_DIG import *
from qcodes.instrument.base import Instrument
from numpy import ndarray

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

    def verify_acquisition_settings(self, **kwargs):
        """
        Ensure that none of the fixed acquisition settings are overwritten
        Args:
            **kwargs: List of acquisition settings

        Returns:
            acquisition settings wwith fixed settings
        """
        for key, val in self._fixed_acquisition_settings.items():
            if kwargs.get(key, val) != val:
                logging.warning('Cannot set {} to {}. Defaulting to {}'.format(
                    key, kwargs[key], val))
            kwargs[key] = val
        return kwargs

    def get_acquisition_setting(self, setting):
        """
        Obtain an acquisition setting for the Keysight.
        It checks if the setting is in AcquisitionController._acquisition_settings
        If not, it will retrieve the driver's latest parameter value

        Args:
            setting: acquisition setting to look for

        Returns:
            Value of the acquisition setting
        """
        if setting in self._acquisition_settings.keys():
            return self._acquisition_settings[setting]
        else:
            # Must get latest value, since it may not be updated in the controller
            return self._keysight.parameters[setting].get_latest()

    def update_acquisition_settings(self, **kwargs):
        """
        Updates acquisition settings after first verifying that none of the
        fixed acquisition settings are overwritten. Any pre-existing settings
        that are not overwritten remain.

        Args:
            **kwargs: acquisition settings

        Returns:
            None
        """
        kwargs = self.verify_acquisition_settings(**kwargs)
        self._acquisition_settings.update(**kwargs)

    def set_acquisition_settings(self, **kwargs):
        """
        Sets acquisition settings after first verifying that none of the
        fixed acquisition settings are overwritten. Any pre-existing settings
        that are not overwritten are removed.

        Args:
            **kwargs: acquisition settings

        Returns:
            None
        """
        kwargs = self.verify_acquisition_settings(**kwargs)
        self._acquisition_settings = kwargs

    def do_acquisition(self):
        """
        Performs an acquisition using the acquisition settings
        Returns:
            None
        """
        records = self._keysight.acquire(acquisition_controller=self,
                                       **self._acquisition_settings)
        return records

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
        """ Initialises a generic Signadyne digitizer and its parameters

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

    def _get_keysight(self):
        """
        returns a reference to the Keysight instrument. A call to self._keysight is
        quicker, so use that if in need for speed

        return: reference to the Keysight instrument
        """
        return self._keysight

    def verify_acquisition_settings(self, **kwargs):
        """
        Ensure that none of the fixed acquisition settings are overwritten
        Args:
            **kwargs: List of acquisition settings

        Returns:
            acquisition settings with fixed settings
        """
        for key, val in self._fixed_acquisition_settings.items():
            if kwargs.get(key, val) != val:
                logging.warning('Cannot set {} to {}. Defaulting to {}'.format(
                    key, kwargs[key], val))
            kwargs[key] = val
        return kwargs

    def get_acquisition_setting(self, setting):
        """
        Obtain an acquisition setting for the Keysight.
        It checks if the setting is in AcquisitionController._acquisition_settings
        If not, it will retrieve the driver's latest parameter value

        Args:
            setting: acquisition setting to look for

        Returns:
            Value of the acquisition setting
        """
        if setting in self._acquisition_settings.keys():
            return self._acquisition_settings[setting]
        else:
            # Must get latest value, since it may not be updated in the controller
            return self._keysight.parameters[setting].get_latest()

    def update_acquisition_settings(self, **kwargs):
        """
        Updates acquisition settings after first verifying that none of the
        fixed acquisition settings are overwritten. Any pre-existing settings
        that are not overwritten remain.

        Args:
            **kwargs: acquisition settings

        Returns:
            None
        """
        kwargs = self.verify_acquisition_settings(**kwargs)
        self._acquisition_settings.update(**kwargs)

    def set_acquisition_settings(self, **kwargs):
        """
        Sets acquisition settings after first verifying that none of the
        fixed acquisition settings are overwritten. Any pre-existing settings
        that are not overwritten are removed.

        Args:
            **kwargs: acquisition settings

        Returns:
            None
        """
        kwargs = self.verify_acquisition_settings(**kwargs)
        self._acquisition_settings = kwargs

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

#    def post_acquire(self):
#        """
#        This method should return any information you want to save from this
#        acquisition. The acquisition method from the Keysight driver will use
#        this data as its own return value
#
#        Returns:
#            this function should return all relevant data that you want
#            to get from the acquisition
#        """


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
