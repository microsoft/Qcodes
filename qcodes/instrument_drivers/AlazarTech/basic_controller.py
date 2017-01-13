import logging
from .ATS import AcquisitionController
import numpy as np
from qcodes import Parameter


class SampleSweep(Parameter):
    """
    Hardware controlled parameter class for Alazar acquisition. To be used with
    Acquisition Controller (tested with ATS9360 board)

    Instrument returns an buffer of data (channels * samples * records) which
    is processed by the post_acquire function of the Acquidiyion Controller
    """

    def __init__(self, name, instrument):
        super().__init__(name)
        self._instrument = instrument
        self.acquisition_kwargs = {}
        self.names = ('A', 'B')

    def get(self):
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

    TODO(nataliejpg) make dtype general or get from alazar
    """

    def __init__(self, name, alazar_name, **kwargs):
        self.samples_per_record = 0
        self.records_per_buffer = 0
        self.buffers_per_acquisition = 0
        self.number_of_channels = 2
        self.board_info = None
        self.buffer = None
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
        npts = kwargs['samples_per_record']
        self.acquisition.shapes = ((npts,), (npts,))
        self.acquisition.acquisition_kwargs.update(**kwargs)

    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
        """
        alazar = self._get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
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
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        pass

    def handle_buffer(self, data):
        """
        See AcquisitionController
        :return:
        """
        self.buffer += data

    def post_acquire(self):
        """
        See AcquisitionController
        :return:
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

        bps = self.board_info['bits_per_sample']
        if bps == 12:
            volt_rec_A = sample_to_volt_u12(recordA, bps)
            volt_rec_B = sample_to_volt_u12(recordB, bps)
        else:
            logging.warning('sample to volt conversion does not exist for bps '
                            '!= 12, raw samples centered on 0 and returned')
            volt_rec_A = recordA - np.mean(recordA)
            volt_rec_B = recordB - np.mean(recordB)

        return volt_rec_A, volt_rec_B


def sample_to_volt_u12(raw_samples, bps):
    # right_shift 16-bit sample by 4 to get 12 bit sample
    shifted_samples = np.right_shift(raw_samples, 4)

    # Alazar calibration
    code_zero = (1 << (bps - 1)) - 0.5
    code_range = (1 << (bps - 1)) - 0.5

    # TODO(nataliejpg) make this not hard coded
    input_range_volts = 1
    # Convert to volts
    volt_samples = np.float64(input_range_volts *
                              (shifted_samples - code_zero) / code_range)

    return volt_samples
