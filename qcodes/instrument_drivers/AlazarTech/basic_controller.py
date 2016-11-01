from .ATS import AcquisitionController
import numpy as np


class Basic_Acquisition_Controller(AcquisitionController):
    """
    This class represents an acquisition controller. It is designed to be used
    primarily to check the function of the Alazar driver and should be used
    with one buffer and one record to return each of the samples unprocessed

    args:
    name: name for this acquisition_conroller as an instrument
    alazar_name: the name of the alazar instrument such that this controller
        can communicate with the Alazar
    **kwargs: kwargs are forwarded to the Instrument base class
    """

    def __init__(self, name, alazar_name, **kwargs):
        self.acquisitionkwargs = {}
        self.samples_per_record = None
        self.bits_per_sample = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.allocated_buffers = None # needed??
        # TODO(damazter) (S) this is not very general:
        self.number_of_channels = 2
        self.buffer = None
        # make a call to the parent class and by extension, create the parameter
        # structure of this class
        super().__init__(name, alazar_name, **kwargs)
        self.add_parameter("acquisition", get_cmd=self.do_acquisition)

    def update_acquisitionkwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisitionkwargs.update(**kwargs)

    def do_acquisition(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        value = self._get_alazar().acquire(acquisition_controller=self,
                                           **self.acquisitionkwargs)
        return value

    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
        """
        alazar = self._get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
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
        # average all records in a buffer
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        if records_per_acquisition != 1:
            raise ValueError(
                'records per acquisition and buffers per acquisition should be set to 1 for this acquisition controller')
        # one of these two...
        # expects S00A, S01A, ...S0B, S1B,...
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record
            i1 = i0 + self.samples_per_record
            recordA += self.buffer[i0:i1] / records_per_acquisition

        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record + len(self.buffer) // 2
            i1 = i0 + self.samples_per_record
            recordB += self.buffer[i0:i1] / records_per_acquisition

        # 1
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S00B, S01A, S01B ..., S10A, S10B, ...
        # where SXYZ is record X, sample Y, channel Z.
        records_per_acquisition = (self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordA += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels + 1)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordB += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        return recordA, recordB
