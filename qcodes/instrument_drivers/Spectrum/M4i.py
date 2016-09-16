#
# **************************************************************************
#
# simple_rep_single.py                           (c) Spectrum GmbH , 11/2009
#
# **************************************************************************
#
# Example for all SpcMDrv based (M2i and M4i) analog replay cards.
# Shows a simple standard mode example using only the few necessary commands
#
# Feel free to use this source for own projects and modify it in any kind
#
# **************************************************************************
#

#%%
import sys
import qcodes
from qcodes.instrument.base import Instrument
try:
    import pyspcm
    from pyspcm import *
except ImportError:
    print("to use the M4i driver install the pyspcm module")
    raise
import logging

#%%
#
# **************************************************************************
# szTypeToName: doing name translation
# **************************************************************************
#


def szTypeToName(lCardType):
    sName = ''
    lVersion = (lCardType & TYP_VERSIONMASK)
    if (lCardType & TYP_SERIESMASK) == TYP_M2ISERIES:
        sName = 'M2i.%04x' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M2IEXPSERIES:
        sName = 'M2i.%04x-Exp' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M3ISERIES:
        sName = 'M3i.%04x' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M3IEXPSERIES:
        sName = 'M3i.%04x-Exp' % lVersion
    elif (lCardType & TYP_SERIESMASK) == TYP_M4IEXPSERIES:
        sName = 'M4i.%04x-x8' % lVersion
    else:
        sName = 'unknown type'
    return sName

#
# **************************************************************************
# main
# **************************************************************************
#

from functools import partial


class M4i(Instrument):

    def __init__(self, name, cardid='spcm0', **kwargs):
        """ Driver for the Spectrum M4i.44xx-x8 cards """
        super().__init__(name, **kwargs)

        self.hCard = pyspcm.spcm_hOpen("spcm0")
        if self.hCard is None:
            logging.warning("M4i: no card found\n")

        # add parameters for getting
        self.add_parameter('max_sample_rate', label='max sample rate',
                           units='Hz', get_cmd=self.get_max_sample_rate, docstring='The maximumum sample rate')
        self.add_parameter('memory', label='memory',
                           units='bytes', get_cmd=self.get_card_memory, docstring='Amount of memory on card')
        self.add_parameter('pcidate', label='pcidate',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_PCIDATE), docstring='The PCI date')
        self.add_parameter('serial_number', label='serial number',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_PCISERIALNO), docstring='The serial number of the board')
        self.add_parameter('channel_count', label='channel count',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_CHCOUNT), docstring='Return number of enabled channels')
        self.add_parameter('input_path_count', label='input path count',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_READAIPATHCOUNT), docstring='Return number of analog input paths')
        self.add_parameter('input_ranges_count', label='input ranges count',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_READIRCOUNT), docstring='Return number of input ranges for the current input path')
        self.add_parameter('input_path_features', label='input path features',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_READAIFEATURES), docstring='Return a bitmap of features for current input path')
        self.add_parameter('available_card_modes', label='available card modes',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_AVAILCARDMODES), docstring='Return a bitmap of available card modes')
        self.add_parameter('card_status', label='card status',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_M2STATUS), docstring='Return a bitmap for the status information')

        # buffer handling
        self.add_parameter('user_available_length', label='user available length',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_DATA_AVAIL_USER_LEN), docstring='returns the number of currently to the user available bytes inside a sample data transfer')
        self.add_parameter('user_available_position', label='user available position',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_DATA_AVAIL_USER_POS), docstring='returns the position as byte index where the currently available data samles start')
        self.add_parameter('buffer_fill_size', label='buffer fill size',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_FILLSIZEPROMILLE), docstring='returns the current fill size of the on-board memory (FIFO buffer) in promille (1/1000)')

        self.add_parameter('available_clock_modes', label='available clock modes',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_AVAILCLOCKMODES), docstring='returns a bitmask in which the bits of the clock modes are set, if available')

        # triggering
        self.add_parameter('available_trigger_or_mask', label='available trigger or mask',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_AVAILORMASK), docstring='bitmask, in which all bits of sources for the OR mask are set, if available')
        self.add_parameter('available_channel_or_mask', label='available channel or mask',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH_AVAILORMASK0), docstring='bitmask, in which all bits of sources/channels (0-31) for the OR mask are set, if available')
        self.add_parameter('available_trigger_and_mask', label='available trigger and mask',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_AVAILANDMASK), docstring='bitmask, in which all bits of sources for the AND mask are set, if available')
        self.add_parameter('available_channel_and_mask', label='available channel and mask',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH_AVAILANDMASK0), docstring='bitmask, in which all bits of sources/channels (0-31) for the AND mask are set, if available')
        self.add_parameter('available_trigger_delay', label='available trigger delay',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_AVAILDELAY), docstring='contains the maximum available delay as decimal integer value')
        self.add_parameter('available_external_trigger_modes', label='available external trigger modes',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_EXT0_AVAILMODES), docstring='bitmask showing all available trigger modes for external 0 (main analog trigger input)')
        self.add_parameter('external_trigger_min_level', label='external trigger min level',
                           units='mV', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_EXT_AVAIL0_MIN), docstring='returns the minimum trigger level')
        self.add_parameter('external_trigger_max_level', label='external trigger max level',
                           units='mV', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_EXT_AVAIL0_MAX), docstring='returns the maximum trigger level')
        self.add_parameter('external_trigger_level_step_size', label='external trigger level step size',
                           units='mV', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_EXT_AVAIL0_STEP), docstring='returns the step size of the trigger level')
        self.add_parameter('available_channel_trigger_modes', label='available channel trigger modes',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH_AVAILMODES), docstring='bitmask, in which all bits of the modes for the channel trigger are set')
        self.add_parameter('trigger_counter', label='trigger counter',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_TRIGGERCOUNTER), docstring='returns the number of triger events since acquisition start')

        # Option star-hub
        self.add_parameter('starhub_count', label='starhub count',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_SYNC_READ_SYNCCOUNT), docstring='number of cards that are connected to star hub')

        # add parameters for setting and getting (read/write direction registers)

        # TODO: write a validator, it is not allowed to enable 3 channels (only 0,1,2 or 4)
        self.add_parameter('enable_channels', label='Channels enabled', get_cmd=partial(self._param32bit, pyspcm.SPC_CHENABLE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_CHENABLE), docstring='Set and get enabled channels')

        # analog input path functions
        self.add_parameter('input_path_0', label='input path 0', get_cmd=partial(self._param32bit, pyspcm.SPC_PATH0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_PATH0), docstring='Set and get analog input path for channel 0')
        self.add_parameter('input_path_1', label='input path 1', get_cmd=partial(self._param32bit, pyspcm.SPC_PATH1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_PATH1), docstring='Set and get analog input path for channel 1')
        self.add_parameter('input_path_2', label='input path 2', get_cmd=partial(self._param32bit, pyspcm.SPC_PATH2),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_PATH2), docstring='Set and get analog input path for channel 2')
        self.add_parameter('input_path_3', label='input path 3', get_cmd=partial(self._param32bit, pyspcm.SPC_PATH3),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_PATH3), docstring='Set and get analog input path for channel 3')
        self.add_parameter('input_path', label='input path', get_cmd=partial(self._param32bit, pyspcm.SPC_READAIPATH),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_READAIPATH), docstring='Select the input path which is used')

        # channel range functions
        self.add_parameter('range_channel_0', label='range channel 0', get_cmd=partial(self._param32bit, pyspcm.SPC_AMP0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_AMP0), docstring='Set and get input range of channel 0')
        self.add_parameter('range_channel_1', label='range channel 1', get_cmd=partial(self._param32bit, pyspcm.SPC_AMP1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_AMP1), docstring='Set and get input range of channel 1')
        self.add_parameter('range_channel_2', label='range channel 2', get_cmd=partial(self._param32bit, pyspcm.SPC_AMP2),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_AMP2), docstring='Set and get input range of channel 2')
        self.add_parameter('range_channel_3', label='range channel 3', get_cmd=partial(self._param32bit, pyspcm.SPC_AMP3),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_AMP3), docstring='Set and get input range of channel 3')

        # input termination functions
        self.add_parameter('termination_0', label='termination 0', get_cmd=partial(self._param32bit, pyspcm.SPC_50OHM0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_50OHM0), docstring='if 1 sets termination to 50 Ohm, otherwise 1 MOhm for channel 0')
        self.add_parameter('termination_1', label='termination 1', get_cmd=partial(self._param32bit, pyspcm.SPC_50OHM1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_50OHM1), docstring='if 1 sets termination to 50 Ohm, otherwise 1 MOhm for channel 1')
        self.add_parameter('termination_2', label='termination 2', get_cmd=partial(self._param32bit, pyspcm.SPC_50OHM2),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_50OHM2), docstring='if 1 sets termination to 50 Ohm, otherwise 1 MOhm for channel 2')
        self.add_parameter('termination_3', label='termination 3', get_cmd=partial(self._param32bit, pyspcm.SPC_50OHM3),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_50OHM3), docstring='if 1 sets termination to 50 Ohm, otherwise 1 MOhm for channel 3')

        # input coupling
        self.add_parameter('ACDC_coupling_0', label='ACDC coupling 0', get_cmd=partial(self._param32bit, pyspcm.SPC_ACDC0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_ACDC0), docstring='if 1 sets the AC coupling, otherwise sets the DC coupling for channel 0')
        self.add_parameter('ACDC_coupling_1', label='ACDC coupling 1', get_cmd=partial(self._param32bit, pyspcm.SPC_ACDC1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_ACDC1), docstring='if 1 sets the AC coupling, otherwise sets the DC coupling for channel 1')
        self.add_parameter('ACDC_coupling_2', label='ACDC coupling 2', get_cmd=partial(self._param32bit, pyspcm.SPC_ACDC2),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_ACDC2), docstring='if 1 sets the AC coupling, otherwise sets the DC coupling for channel 2')
        self.add_parameter('ACDC_coupling_3', label='ACDC coupling 3', get_cmd=partial(self._param32bit, pyspcm.SPC_ACDC3),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_ACDC3), docstring='if 1 sets the AC coupling, otherwise sets the DC coupling for channel 3')

        # AC/DC offset compensation
        self.add_parameter('ACDC_offs_compensation_0', label='ACDC offs compensation 0', get_cmd=partial(self._param32bit, pyspcm.SPC_ACDC_OFFS_COMPENSATION0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_ACDC_OFFS_COMPENSATION0), docstring='if 1 enables compensation, if 0 disables compensation for channel 0')
        self.add_parameter('ACDC_offs_compensation_1', label='ACDC offs compensation 1', get_cmd=partial(self._param32bit, pyspcm.SPC_ACDC_OFFS_COMPENSATION1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_ACDC_OFFS_COMPENSATION1), docstring='if 1 enables compensation, if 0 disables compensation for channel 1')
        self.add_parameter('ACDC_offs_compensation_2', label='ACDC offs compensation 2', get_cmd=partial(self._param32bit, pyspcm.SPC_ACDC_OFFS_COMPENSATION2),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_ACDC_OFFS_COMPENSATION2), docstring='if 1 enables compensation, if 0 disables compensation for channel 2')
        self.add_parameter('ACDC_offs_compensation_3', label='ACDC offs compensation 3', get_cmd=partial(self._param32bit, pyspcm.SPC_ACDC_OFFS_COMPENSATION3),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_ACDC_OFFS_COMPENSATION3), docstring='if 1 enables compensation, if 0 disables compensation for channel 3')

        # anti aliasing filter (Bandwidth limit)
        self.add_parameter('anti_aliasing_filter_0', label='anti aliasing filter 0', get_cmd=partial(self._param32bit, pyspcm.SPC_FILTER0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_FILTER0), docstring='if 1 selects bandwidth limit, if 0 sets to full bandwidth for channel 0')
        self.add_parameter('anti_aliasing_filter_1', label='anti aliasing filter 1', get_cmd=partial(self._param32bit, pyspcm.SPC_FILTER1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_FILTER1), docstring='if 1 selects bandwidth limit, if 0 sets to full bandwidth for channel 1')
        self.add_parameter('anti_aliasing_filter_2', label='anti aliasing filter 2', get_cmd=partial(self._param32bit, pyspcm.SPC_FILTER2),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_FILTER2), docstring='if 1 selects bandwidth limit, if 0 sets to full bandwidth for channel 2')
        self.add_parameter('anti_aliasing_filter_3', label='anti aliasing filter 3', get_cmd=partial(self._param32bit, pyspcm.SPC_FILTER3),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_FILTER3), docstring='if 1 selects bandwidth limit, if 0 sets to full bandwidth for channel 3')

        # acquisition modes
        self.add_parameter('card_mode', label='card mode', get_cmd=partial(self._param32bit, pyspcm.SPC_CARDMODE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_CARDMODE), docstring='defines the used operating mode')

        # wait command
        self.add_parameter('timeout', label='timeout', get_cmd=partial(self._param32bit, pyspcm.SPC_TIMEOUT),
                           units='ms', set_cmd=partial(self._set_param32bit, pyspcm.SPC_TIMEOUT), docstring='defines the timeout for wait commands')

        # Single acquisition mode memory, pre- and posttrigger (pretrigger = memory size - posttrigger)
        self.add_parameter('data_memory_size', label='data memory size', get_cmd=partial(self._param32bit, pyspcm.SPC_MEMSIZE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_MEMSIZE), docstring='sets the memory size in samples per channel')
        self.add_parameter('posttrigger_memory_size', label='posttrigger memory size', get_cmd=partial(self._param32bit, pyspcm.SPC_POSTTRIGGER),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_POSTTRIGGER), docstring='sets the number of samples to be recorded after trigger event')

        # FIFO single acquisition length and pretrigger
        self.add_parameter('pretrigger_memory_size', label='pretrigger memory size', get_cmd=partial(self._param32bit, pyspcm.SPC_PRETRIGGER),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_PRETRIGGER), docstring='sets the number of samples to be recorded before trigger event')
        self.add_parameter('segment_size', label='segment size', get_cmd=partial(self._param32bit, pyspcm.SPC_SEGMENTSIZE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_SEGMENTSIZE), docstring='length of segments to acquire')
        self.add_parameter('total_segments', label='total segments', get_cmd=partial(self._param32bit, pyspcm.SPC_LOOPS),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_LOOPS), docstring='number of segments to acquire in total. Setting 0 makes it run until stopped by user')

        # converting ADC samples to voltage values
        self.add_parameter('ADC_to_voltage', label='ADC to voltage', get_cmd=partial(self._param32bit, pyspcm.SPC_MIINST_MAXADCVALUE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_MIINST_MAXADCVALUE), docstring='contains the decimal code (in LSB) of the ADC full scale value')

        # clock generation
        self.add_parameter('clock_mode', label='clock mode', get_cmd=partial(self._param32bit, pyspcm.SPC_CLOCKMODE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_CLOCKMODE), docstring='defines the used clock mode or reads out the actual selected one')
        # NOTE: write and read are written separately, so we should test if this fuction works:
        self.add_parameter('sample_rate', label='sample rate', get_cmd=partial(self._param32bit, pyspcm.SPC_SAMPLERATE),
                           units='Hz', set_cmd=partial(self._set_param32bit, pyspcm.SPC_SAMPLERATE), docstring='write the sample rate for internal sample generation or read rate nearest to desired')

        # triggering
        self.add_parameter('trigger_or_mask', label='trigger or mask', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_ORMASK),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_ORMASK), docstring='defines the events included within the  trigger OR mask card')
        self.add_parameter('channel_or_mask', label='channel or mask', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH_ORMASK0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH_ORMASK0), docstring='includes the channels (0-31) within the channel trigger OR mask of the card')
        self.add_parameter('trigger_and_mask', label='trigger and mask', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_ANDMASK),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_ANDMASK), docstring='defines the events included within the  trigger AND mask card')
        self.add_parameter('channel_and_mask', label='channel and mask', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH_ANDMASK0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH_ANDMASK0), docstring='includes the channels (0-31) within the channel trigger AND mask of the card')
        self.add_parameter('trigger_delay', label='trigger delay', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_DELAY),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_DELAY), docstring='defines the delay for the detected trigger events')
        self.add_parameter('external_trigger_mode', label='external trigger mode', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_EXT0_MODE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_EXT0_MODE), docstring='defines the external trigger mode for the external SMA connector trigger input')
        self.add_parameter('trigger_termination', label='trigger termination', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_TERM),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_TERM), docstring='A 1 sets the 50 Ohm termination, a 0 sets high impedance termination')
        self.add_parameter('external_trigger_input_coupling', label='external trigger input coupling', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_EXT0_ACDC),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_EXT0_ACDC), docstring='A 1 sets the AC coupling for the external trigger, a 0 sets DC')
        self.add_parameter('external_trigger_level_0', label='external trigger level 0', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_EXT0_LEVEL0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_EXT0_LEVEL0), docstring='trigger level 0 for external trigger')
        self.add_parameter('external_trigger_level_1', label='external trigger level 1', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_EXT0_LEVEL1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_EXT0_LEVEL1), docstring='trigger level 1 for external trigger')
        self.add_parameter('trigger_mode_channel_0', label='trigger mode channel 0', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH0_MODE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH0_MODE), docstring='sets the triggre mode for channel 0')
        self.add_parameter('trigger_mode_channel_1', label='trigger mode channel 1', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH1_MODE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH1_MODE), docstring='sets the triggre mode for channel 1')
        self.add_parameter('trigger_mode_channel_2', label='trigger mode channel 2', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH2_MODE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH2_MODE), docstring='sets the triggre mode for channel 2')
        self.add_parameter('trigger_mode_channel_3', label='trigger mode channel 3', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH3_MODE),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH3_MODE), docstring='sets the triggre mode for channel 3')
        self.add_parameter('trigger_channel_0_level_0', label='trigger channel 0 level 0', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH0_LEVEL0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH0_LEVEL0), docstring='triggre level 0 channel 0')
        self.add_parameter('trigger_channel_1_level_0', label='trigger channel 1 level 0', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH1_LEVEL0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH1_LEVEL0), docstring='triggre level 0 channel 1')
        self.add_parameter('trigger_channel_2_level_0', label='trigger channel 2 level 0', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH2_LEVEL0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH2_LEVEL0), docstring='triggre level 0 channel 2')
        self.add_parameter('trigger_channel_3_level_0', label='trigger channel 3 level 0', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH3_LEVEL0),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH3_LEVEL0), docstring='triggre level 0 channel 3')
        self.add_parameter('trigger_channel_0_level_1', label='trigger channel 0 level 1', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH0_LEVEL1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH0_LEVEL1), docstring='triggre level 1 channel 0')
        self.add_parameter('trigger_channel_1_level_1', label='trigger channel 1 level 1', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH1_LEVEL1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH1_LEVEL1), docstring='triggre level 1 channel 1')
        self.add_parameter('trigger_channel_2_level_1', label='trigger channel 2 level 1', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH2_LEVEL1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH2_LEVEL1), docstring='triggre level 1 channel 2')
        self.add_parameter('trigger_channel_3_level_1', label='trigger channel 3 level 1', get_cmd=partial(self._param32bit, pyspcm.SPC_TRIG_CH3_LEVEL1),
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_TRIG_CH3_LEVEL1), docstring='triggre level 1 channel 3')

        # add parameters for setting (write only registers)

        # Buffer handling
        self.add_parameter('card_available_length', label='card available length',
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_DATA_AVAIL_CARD_LEN), docstring='writes the number of bytes that the card can now use for sample data transfer again')

        # General
        self.add_parameter('general_command', label='general command',
                           set_cmd=partial(self._set_param32bit, pyspcm.SPC_M2CMD), docstring='executes a command for the card or data transfer')

    def close(self):
        """ Close handle to the card """
        if self.hCard is not None:
            pyspcm.spcm_vClose(self.hCard)
            self.hCard = None
        super().close()

    def get_card_type(self, verbose=0):
        """ Read card type """
        # read type, function and sn and check for D/A card
        lCardType = int32(0)
        spcm_dwGetParam_i32(self.hCard, SPC_PCITYP, byref(lCardType))
        if verbose:
            print('card_type: %s' % szTypeToName(lCardType.value))
        return (lCardType.value)

    # only works if the error was not caused by running the entire program (and therefore making a new M4i object)
    def get_error_info32bit(self):
        """ Read an error from the error register """
        dwErrorReg = uint32(0)
        lErrorValue = int32(0)

        spcm_dwGetErrorInfo_i32(self.hCard, byref(dwErrorReg), byref(lErrorValue), None)
        return (dwErrorReg.value, lErrorValue.value)

    def _param64bit(self, param):
        """ Read a 64-bit parameter from the device """
        data = int64(0)
        pyspcm.spcm_dwGetParam_i64(self.hCard, param, byref(data))
        return (data.value)

    def _param32bit(self, param):
        """ Read a 32-bit parameter from the device """
        data = int32(0)
        pyspcm.spcm_dwGetParam_i32(self.hCard, param, byref(data))
        return (data.value)

    def _set_param32bit(self, param, value):
        """ Read a 32-bit parameter from the device """
        pyspcm.spcm_dwSetParam_i32(self.hCard, param, value)

    def get_max_sample_rate(self, verbose=0):
        """ Return max sample rate """
        # read type, function and sn and check for D/A card
        value = self._param32bit(pyspcm.SPC_PCISAMPLERATE)
        if verbose:
            print('max_sample_rate: %s' % (value))
        return value

    def get_card_memory(self, verbose=0):
        data = int64(0)
        spcm_dwGetParam_i64(self.hCard, pyspcm.SPC_PCIMEMSIZE, byref(data))
        if verbose:
            print('card_memory: %s' % (data.value))
        return (data.value)