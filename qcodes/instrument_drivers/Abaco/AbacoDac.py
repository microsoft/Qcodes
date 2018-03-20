from contextlib import contextmanager
import numpy as np
import io
from typing import List
from math import ceil
import shutil
import time

from qcodes.instrument.ip import IPInstrument



class AbacoDAC(IPInstrument):
    V_PP_DC = 1.7  # Data sheet FMC144 user manual p. 14
    V_PP_AC = 1.0  # Data sheet FMC144 user manual p. 14
    DAC_RESOLUTION_BITS = 16
    SAMPLES_IN_BUFFER_DIVISOR = 4
    # FILENAME = r"\\PCX79470-0001RG\Users\4DSP-PCIX490-0001\OneDrive - Microsoft\4DSP\sw_23_10_2017_sandbox\PRJ0161_WorkstationApp\waveform\test2_{}.txt"
    FILENAME = "test_{}.{}"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, persistent=False, terminator='')
        with self.temporary_timeout(11):
            print("asked returned {}".format(self.ask("init_state\n")))
            print("asked returned {}".format(self.ask("init_state\n")))
        # cls.ask("init_state")
        # time.sleep(1)
        # cls.ask("config_state")
        # glWaveFileMask=test_
        pass

    @contextmanager
    def temporary_timeout(self, timeout):
        old_timeout = self._timeout
        self.set_timeout(timeout)
        yield
        self.set_timeout(old_timeout)


    @classmethod
    def _create_file(cls, data: List[np.ndarray], dformat):
        res = cls._makeTextDataFile(data, dformat)
        if dfomart == 1:
            file_access = 'w'
        else:
            file_access = 'wb'
        # write files to disk
        for i in range(2):
            with open(cls.FILENAME.format(i, 'txt'), file_access) as fd:
                res.seek(0)
                shutil.copyfileobj(res, fd)

    @classmethod
    def create_txt_file(cls, data: List[np.ndarray]):
        cls._create_file(data, dformat=1)

    @classmethod
    def create_dat_file(cls, data: List[np.ndarray]):
        cls._create_file(data, dformat=2)

    @classmethod
    def _makeTextDataFile(cls, data: List[np.ndarray], dformat: int) -> str:
        """
        This function produces a text data file for the abaco DAC that
        specifies the waveforms. Samples are represented by integer values.
        The file has the following strucutre:
        (lines starting with '#' are not part of the file)
        #--Header--------------------------------------------------------------
        <number of blocks>
        <total number of samples channel 1>
        <total number of samples channel 2>
        ...
        <total number of samples channel 8>
        #--Block 1-------------------------------------------------------------
        <sample 1, channel 1>
        <s1-c2>
        <s1-c3>
        ...
        <s1-c8>
        <s2-c1>
        <s2-c2>
        ....
        <sN-c8>
        #--Block 2-------------------------------------------------------------
        <s1-c8>
        ....
        <sN-c8>
        #--Block 3-------------------------------------------------------------
        ...
        Please note that all blocks have to have the same length
        Args:
            data: The actual waveform data as a list of matrices of shape
                  (n_points, n_channels) for each block. n_channels must be
                  identical for each block
        """
        # checking input data
        n_blocks = len(data)
        n_channels = data[0].shape[1]
        block_size = max([a.shape[0] for a in data])
        d = cls.SAMPLES_IN_BUFFER_DIVISOR
        padded_block_size = -(-block_size//d)*d
        total_num_samples = padded_block_size*n_blocks
        # writing the header
        if dformat == 1:
            output = io.StringIO()
        elif dformat == 2:
            output = io.BytesIO()

        cls.write_sample(output, n_blocks, dformat)
        for i in range(n_channels):
            cls.write_sample(output, total_num_samples, dformat)


        # writing the waveform of each block
        for block in data:
            for i_sample in range(block.shape[0]):
                for i_channel in range(block.shape[1]):
                    current_sample = cls._voltage_to_int(
                        block[i_sample, i_channel])
                    cls.write_sample(output, current_sample, dformat)

            # padding
            for i_sample in range(block.shape[0], padded_block_size):
                for i_channel in range(block.shape[1]):
                    current_sample = cls._voltage_to_int(0)
                    cls.write_sample(output, current_sample, dformat)
        return output

    @classmethod
    def _voltage_to_int(cls, v):
        return int(round(v/cls.V_PP_DC * 2**(cls.DAC_RESOLUTION_BITS-1)))

    @staticmethod
    def write_sample(stream, sample, dformat):
        if dformat == 1:
            print('{}'.format(sample), file=stream)
        elif dformat == 2:
            stream.write(sample.to_bytes(4, byteorder='big', signed=True))


