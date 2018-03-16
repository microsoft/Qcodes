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

    def __init__(cls, *args, **kwargs) -> None:
        # super().__init__(*args, **kwargs, persistent=False, terminator='')
        # cls.ask("init_state")
        # time.sleep(1)
        # cls.ask("config_state")
        # glWaveFileMask=test_
        pass

    @classmethod
    def _create_file(cls, data: List[np.ndarray], dformat):
        res = cls._makeTextDataFile(data, dformat)
        if dfomart == 1:
            file_access = 'w'
        elif:
            file_access = 'wb'
        # write files to disk
        for i in range(2):
            with open(cls.FILENAME.format(i, 'txt'), file_access) as fd:
                res.seek(0)
                shutil.copyfileobj(res, fd)

    @classmethod
    def create_txt_file(cls, data: List[np.ndarray]):
        cls._create_file(data: List[np.ndarray], dformat=1)

    @classmethod
    def create_dat_file(cls, data: List[np.ndarray]):
        cls._create_file(data: List[np.ndarray], dformat=2)

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


import broadbean as bb
ramp = bb.PulseAtoms.ramp  # args: start, stop
sine = bb.PulseAtoms.sine  # args: freq, ampl, off


# seq1 = bb.Sequence()
# # Create the blueprints
# bp_fill = bb.BluePrint()
# bp_fill.setSR(1e9)
# bp_fill.insertSegment(0, ramp, (0, 0), dur=6*28e-9)
# bp_square = bb.BluePrint()
# bp_square.setSR(1e9)
# bp_square.insertSegment(0, ramp, (0, 0), dur=100e-9)
# bp_square.insertSegment(1, ramp, (1e-3, 1e-3), name='top', dur=100e-9)
# bp_square.insertSegment(2, ramp, (0, 0), dur=100e-9)
# bp_boxes = bp_square + bp_square + bp_fill
# #
# bp_sine = bb.BluePrint()
# bp_sine.setSR(1e9)
# bp_sine.insertSegment(0, sine, (3.333e6, 1.5e-3, 0), dur=300e-9)
# bp_sineandboxes = bp_sine + bp_square + bp_fill

# # create elements
# elem1 = bb.Element()
# elem1.addBluePrint(1, bp_boxes)
# elem1.addBluePrint(3, bp_sineandboxes)
# #
# elem2 = bb.Element()
# elem2.addBluePrint(3, bp_boxes)
# elem2.addBluePrint(1, bp_sineandboxes)

# # Fill bp1 the sequence
# seq1.addElement(1, elem1)  # Call signature: seq. pos., element
# seq1.addElement(2, elem2)
# seq1.addElement(3, elem1)

# # set its sample rate
# seq1.setSR(elem1.SR)
# seq1.setChannelAmplitude(1, 1e-3)  # Call signature: channel, amplitude (peak-to-peak)
# seq1.setChannelAmplitude(3, 1e-3)

# seq1.plotSequence()


bp1= bb.BluePrint()
bp1.setSR(1e9)
for i in range(3):
    bp1.insertSegment(0, ramp, (0, 1.5), dur=3e-6)
    bp1.insertSegment(0, ramp, (1.5, 0.0), dur=3e-6)



elem1 = bb.Element()
for i in range(8):
    elem1.addBluePrint(i, bp1)

seq = bb.Sequence()
seq.addElement(1, elem1)
# seq.addElement(2, elem1)

seq.setSR(elem1.SR)
for i in range(8):
    seq.setChannelAmplitude(i, 1e-3)
# seq.setChannelAmplitude(2, 1e-3)
seq.plotSequence()

package = seq.outputForAbacoDacFile()


import qcodes as qc
qc.Instrument.close_all()
abaco = AbacoDAC('abaco', '172.20.3.94', port=27015)
abaco.create_txt_file(package)
abaco.create_dat_file(package)


# 782/11: self.ask("init_state")
# 833/4: abaco.ask("glWaveFileMask=test_")
# 833/5: abaco.ask("glWaveFileMask=test_")
# 833/6: abaco.ask("init_state")
# 833/7: abaco.ask("config_state")
# 835/2: abaco.ask("glWaveFileMask=test_")
# 835/3: abaco.ask("init_state")
# 835/4: abaco.ask("config_state")
# 835/5: abaco.ask("load_waveform_state")
# 835/6: abaco.ask("enable_upload_state")
# 835/7: abaco.ask("enable_offload_state")
# 835/9: abaco.ask("disable_offload_state")
# 835/10: abaco.ask("glWaveFileMask=test2_")
# 835/11: abaco.ask("load_waveform_state")
# 835/12: abaco.ask("enable_offload_state")
