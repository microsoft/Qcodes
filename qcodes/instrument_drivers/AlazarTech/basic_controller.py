import logging
from .ATS import AcquisitionController
import numpy as np
import qcodes.instrument_drivers.AlazarTech.acq_helpers as helpers
from qcodes import Parameter


class SampleSweep(Parameter):
    """
    Hardware controlled parameter class for Alazar acquisition. To be used with
    Acquisition Controller (tested with ATS9360 board)

    Alazar Instrument 'acquire' returns a buffer of data each time a buffer is
    filled (channels * samples * records) which is processed by the
    post_acquire function of the Acquisition Controller and finally the
    processed result is returned when the SampleSweep parameter is called.

    :args:
    name: name for this parameter
    instrument: acquisition controller instrument this parameter belongs to
    """

    def __init__(self, name, instrument):
        super().__init__(name)
        self._instrument = instrument
        self.acquisition_kwargs = {}
        self.names = ('A', 'B')

    def get(self):
        """
        Gets the samples for channels A and B by calling acquire
        on the alazar (which in turn calls the processing functions of the
        aqcuisition controller before returning the reshaped data averaged
        over records and buffers)

        returns:
        recordA: numpy array of channel A acquisition
        recordB: numpy array of channel B acquisition
        """
        recordA, recordB = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **self.acquisition_kwargs)
        return recordA, recordB


class Basic_Acquisition_Controller(AcquisitionController):
    """
    This class represents an acquisition controller. It is designed to be used
    primarily to check the function of the Alazar driver and returns the
    samples on channel A and channel B, averaging over recoirds and buffers

    args:
    name: name for this acquisition_conroller as an instrument
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
                           parameter_class=SampleSweep)

    def update_acquisition_kwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.samples_per_record = kwargs['samples_per_record']
        self.acquisition.shapes = (
            (self.samples_per_record,), (self.samples_per_record,))
        self.acquisition.acquisition_kwargs.update(**kwargs)

    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
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
        :return:
        """
        pass

    def handle_buffer(self, data):
        """
        Function which is called during the Alazar acquire each time a buffer
        is completed. In this acquistion controller these buffers are just
        added together (ie averaged)
        :return:
        """
        self.buffer += data

    def post_acquire(self):
        """
        Function which is called at the end of the Alazar acquire function to
        signal completion and trigger data processing. This acquisition
        controller has averaged over the buffers acquired so has one buffer of
        data which is splits into records and channels, averages over records
        and returns the samples for each channel.

        return:
        recordA: numpy array of channel A acquisition
        recordB: numpy array of channel B acquisition
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
