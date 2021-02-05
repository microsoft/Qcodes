from typing import Any, Optional, Dict, Tuple

from .ATS import AcquisitionController, OutputType
import math
import numpy as np


# DFT AcquisitionController
class Demodulation_AcquisitionController(AcquisitionController[float]):
    """
    This class represents an example acquisition controller. End users will
    probably want to use something more sophisticated. It will average all
    buffers and then perform a fourier transform on the resulting average trace
    for one frequency component. The amplitude of the result of channel_a will
    be returned.

    Args:
        name: name for this acquisition_conroller as an instrument

        alazar_name: the name of the alazar instrument such that this controller
            can communicate with the Alazar

        demodulation_frequency: the selected component for the fourier transform

        **kwargs: kwargs are forwarded to the Instrument base class

    """
    def __init__(
            self,
            name: str,
            alazar_name: str,
            demodulation_frequency: float,
            **kwargs: Any):
        self.demodulation_frequency = demodulation_frequency
        self.acquisitionkwargs: Dict[str, Any] = {}
        self.samples_per_record = 0
        self.records_per_buffer = 0
        self.buffers_per_acquisition = 0
        self.number_of_channels = 2
        self.cos_list = None
        self.sin_list = None
        self.buffer: Optional[np.ndarray] = None
        # make a call to the parent class and by extension, create the parameter
        # structure of this class
        super().__init__(name, alazar_name, **kwargs)
        self.add_parameter("acquisition", get_cmd=self.do_acquisition)

    def update_acquisitionkwargs(self, **kwargs: Any) -> None:
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisitionkwargs.update(**kwargs)

    def do_acquisition(self) -> float:
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        value: float = self._get_alazar().acquire(acquisition_controller=self,
                                           **self.acquisitionkwargs)
        return value

    def pre_start_capture(self) -> None:
        """
        See AcquisitionController
        """
        alazar = self._get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        sample_speed = alazar.get_sample_rate()
        integer_list = np.arange(self.samples_per_record)
        angle_list = (2 * np.pi * self.demodulation_frequency / sample_speed *
                      integer_list)

        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

    def pre_acquire(self) -> None:
        """
        See AcquisitionController
        :return:
        """
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        pass

    def handle_buffer(
            self,
            data: np.ndarray,
            buffer_number: Optional[int] = None
    ) -> None:
        """
        See AcquisitionController
        :return:
        """
        assert self.buffer is not None
        self.buffer += data

    def post_acquire(self) -> float:
        """
        See AcquisitionController
        :return:
        """
        assert self.buffer is not None
        alazar = self._get_alazar()
        # average all records in a buffer
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
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

        if self.number_of_channels == 2:
            # fit channel A and channel B
            res1 = self.fit(recordA)
            res2 = self.fit(recordB)
            #return [alazar.signal_to_volt(1, res1[0] + 127.5),
            #        alazar.signal_to_volt(2, res2[0] + 127.5),
            #        res1[1], res2[1],
            #        (r es1[1] - res2[1]) % 360]
            return alazar.signal_to_volt(1, res1[0] + 127.5)
        else:
            raise Exception("Could not find CHANNEL_B during data extraction")

    def fit(self, buf: np.ndarray) -> Tuple[float, float]:
        """
        the DFT is implemented in this method
        :param buf: buffer to perform the transform on
        :return: return amplitude and phase of the resulted transform
        """
        # Discrete Fourier Transform
        RePart = np.dot(buf - 127.5, self.cos_list) / self.samples_per_record
        ImPart = np.dot(buf - 127.5, self.sin_list) / self.samples_per_record

        # the factor of 2 in the amplitude is due to the fact that there is
        # a negative frequency as well
        ampl = 2 * np.sqrt(RePart ** 2 + ImPart ** 2)

        # see manual page 52!!! (using unsigned data)
        return ampl, math.atan2(ImPart, RePart) * 360 / (2 * math.pi)
