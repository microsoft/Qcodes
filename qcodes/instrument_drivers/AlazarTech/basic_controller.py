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
        self.acquisitionkwargs = {}
        self.names = ('A', 'B')
        self.units = ('', '')
        self.setpoint_names = (('sample_num',), ('sample_num',))
        self.setpoints = ((1,), (1,))
        self.shapes = ((1,), (1,))

    def update_acquisition_kwargs(self, **kwargs):
        # needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up
        if 'samples_per_record' in kwargs:
            npts = kwargs['samples_per_record']
            n = tuple(np.arange(npts))
            self.setpoints = ((n,), (n,))
            self.shapes = ((npts,), (npts,))
        else:
            raise ValueError('samples_per_record must be specified')
        # updates dict to be used in acquisition get call
        self.acquisitionkwargs.update(**kwargs)

    def get(self):
        recordA, recordB = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **self.acquisitionkwargs)
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

    TODO(nataliejpg) num of channels
    TODO(nataliejpg) make dtype general or get from alazar
    """

    def __init__(self, name, alazar_name, **kwargs):
        self.samples_per_record = 0
        self.records_per_buffer = 0
        self.buffers_per_acquisition = 0
        # TODO(damazter) (S) this is not very general:
        self.number_of_channels = 2
        self.buffer = None
        # make a call to the parent class and by extension,
        # create the parameter structure of this class
        super().__init__(name, alazar_name, **kwargs)

        self.add_parameter(name='acquisition',
                           parameter_class=SampleSweep)

    def update_acquisitionkwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisition.update_acquisition_kwargs(**kwargs)

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
                               self.number_of_channels,
                               dtype=np.uint16)

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
        recordA = np.zeros(self.samples_per_record, dtype=np.uint16)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordA += np.uint16(self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        recordB = np.zeros(self.samples_per_record, dtype=np.uint16)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels + 1)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordB += np.uint16(self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)
        
        volt_rec_A = self.sample_to_volt(recordA)
        volt_rec_B = self.sample_to_volt(recordB)

        return volt_rec_A, volt_rec_B

    def sample_to_volt(self, raw_samples):
        # right_shift 16-bit sample by 4 to get 12 bit sample
        shifted_samples = np.right_shift(raw_samples,4)
        
        # Alazar calibration
        bps = 12
        input_range_volts = 0.8
        code_zero = (1 << (bps - 1)) - 0.5
        code_range = (1 << (bps - 1)) - 0.5
        
        # Convert to volts
        volt_samples = input_range_volts * (shifted_samples - code_zero) / code_range
        
        return volt_samples