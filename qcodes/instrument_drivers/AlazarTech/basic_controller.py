from .ATS import AcquisitionController
import numpy as np
from qcodes import Parameter


class SampleSweep(Parameter):
    """
    Hardware controlled parameter class for Rohde Schwarz RSZNB20 trace.

    Instrument returns an list of transmission data in the form of a list of
    complex numbers taken from a frequency sweep.
    """
    def __init__(self, name, instrument):
        super().__init__(name)
        self._instrument = instrument
        #self.npts = npts
        self.acquisitionkwargs = {}
        self.names = ('A', 'B')
        self.units = ('', '')
        self.setpoint_names = (('sample_num',), ('sample_num',))
        self.setpoints = ((1,), (1,))
        self.shapes = ((1,), (1,))
        
    def update_acquisition_kwargs(self, **kwargs):
        if 'samples_per_record' in kwargs:
            npts = kwargs['samples_per_record']
            n = tuple(np.arange(npts))
            self.setpoints = ((n,), (n,))
            self.shapes = ((npts,), (npts,))
        else:
            raise ValueError('samples_per_record not in kwargs at time of update')
        self.acquisitionkwargs.update(**kwargs)

    def get(self):
        recordA, recordB = self._instrument._get_alazar().acquire(
            acquisition_controller = self._instrument,
            **self.acquisitionkwargs)
#        recordA, recordB = self._instrument.do_acquisition()
        return recordA, recordB


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
        #self.acquisitionkwargs = {}
        self.samples_per_record = None
        self.bits_per_sample = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.allocated_buffers = None  # needed??
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

    def do_acquisition(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        valueA, valueB = self._get_alazar().acquire(acquisition_controller=self,
                                           **self.acquisitionkwargs)
        return valueA, valueB

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
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        if records_per_acquisition != 1:
            raise ValueError(
                'records and buffers should be set to 1 for this acquisition controller')
        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S01A, ...S0B, S1B,...
        # where SXYZ is record X, sample Y, channel Z.

        # breaks buffer up into records, averages over them and returns samples
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
