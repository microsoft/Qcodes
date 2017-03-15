import logging
from .ATS import AcquisitionController
import numpy as np
import qcodes.instrument_drivers.AlazarTech.acq_helpers as helpers
from .acquisition_parameters import AlazarMultiArray


class Basic_Acquisition_Controller(AcquisitionController):
    """
    This class represents an acquisition controller. It is designed to be used
    primarily to check the function of the Alazar driver and returns the
    samples on channel A and channel B, averaging over records and buffers

    args:
        name: name for this acquisition_controller as an instrument
        alazar_name: the name of the alazar instrument such that this controller
            can communicate with the Alazar
        **kwargs: kwargs are forwarded to the Instrument base class
    """

    def __init__(self, name, alazar_name, **kwargs):
        self.number_of_channels = 2
        # make a call to the parent class and by extension,
        # create the parameter structure of this class
        super().__init__(name, alazar_name, **kwargs)

        self.add_parameter(name='acquisition',
                           parameter_class=AlazarMultiArray,
                           names=('A', 'B'))

    def update_acquisition_kwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire

        Args:
            **kwargs:

        """
        self.samples_per_record = kwargs['samples_per_record']
        self.acquisition.shapes = (
            (self.samples_per_record,), (self.samples_per_record,))
        self.acquisition.acquisition_kwargs.update(**kwargs)

    def pre_start_capture(self):
        """
        See AcquisitionController
        """
        alazar = self._get_alazar()
        if self.samples_per_record != alazar.samples_per_record.get():
            raise Exception('Instrument samples_per_record settings does '
                            'not match acq controller value, most likely '
                            'need to call update_acquisition_settings')
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        self.board_info = alazar.get_idn()
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

    def pre_acquire(self):
        """
        See AcquisitionController
        """
        pass

    def handle_buffer(self, data):
        """
        Function which is called during the Alazar acquire each time a buffer
        is completed. In this acquisition controller these buffers are just
        added together (ie averaged)
        """
        self.buffer += data

    def post_acquire(self):
        """
        Function which is called at the end of the Alazar acquire function to
        signal completion and trigger data processing. This acquisition
        controller has averaged over the buffers acquired so has one buffer of
        data which is splits into records and channels, averages over records
        and returns the samples for each channel.

        Returns:
            - recordA a numpy array of channel A acquisition
            - recordB a numpy array of channel B acquisition
        """

        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S00B, S01A, S01B...S10A, S10B, S11A, S11B...
        # where SXYZ is record X, sample Y, channel Z.

        # break buffer up into records, averages over them and returns samples
        records_per_acquisition = (self.buffers_per_acquisition *
                                   self.records_per_buffer)

        recA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recA += self.buffer[i0:i1:self.number_of_channels]
        recordA = np.uint16(recA / records_per_acquisition)

        recB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels + 1)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recB += self.buffer[i0:i1:self.number_of_channels]
        recordB = np.uint16(recB / records_per_acquisition)

        # converts to volts if bits per sample is 12 (as ATS9360)
        bps = self.board_info['bits_per_sample']
        if bps == 12:
            volt_rec_A = helpers.sample_to_volt_u12(recordA, bps)
            volt_rec_B = helpers.sample_to_volt_u12(recordB, bps)
        else:
            logging.warning('sample to volt conversion does not exist for bps '
                            '!= 12, raw samples centered on 0 and returned')
            volt_rec_A = recordA - np.mean(recordA)
            volt_rec_B = recordB - np.mean(recordB)

        return volt_rec_A, volt_rec_B
