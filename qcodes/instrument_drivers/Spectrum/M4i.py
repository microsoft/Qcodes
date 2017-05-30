# **************************************************************************
#
# Driver file for M4i.44x-x8
#
# **************************************************************************
#
# QuTech
#
# Written by: Luka Bavdaz, Marco Tagliaferri, Pieter Eendebak
# Also see: http://spectrum-instrumentation.com/en/m4i-platform-overview
#

#%%
import logging
import numpy as np
import ctypes as ct
from functools import partial
from qcodes.utils.validators import Enum, Numbers, Anything
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
try:
    import pyspcm
except ImportError:
    raise ImportError('to use the M4i driver install the pyspcm module')

#%% Helper functions


def szTypeToName(lCardType):
    """ Convert card type to string

    This function is taken from an example provided by Spectrum GmbH
    """
    sName = ''
    lVersion = (lCardType & pyspcm.TYP_VERSIONMASK)
    if (lCardType & pyspcm.TYP_SERIESMASK) == pyspcm.TYP_M2ISERIES:
        sName = 'M2i.%04x' % lVersion
    elif (lCardType & pyspcm.TYP_SERIESMASK) == pyspcm.TYP_M2IEXPSERIES:
        sName = 'M2i.%04x-Exp' % lVersion
    elif (lCardType & pyspcm.TYP_SERIESMASK) == pyspcm.TYP_M3ISERIES:
        sName = 'M3i.%04x' % lVersion
    elif (lCardType & pyspcm.TYP_SERIESMASK) == pyspcm.TYP_M3IEXPSERIES:
        sName = 'M3i.%04x-Exp' % lVersion
    elif (lCardType & pyspcm.TYP_SERIESMASK) == pyspcm.TYP_M4IEXPSERIES:
        sName = 'M4i.%04x-x8' % lVersion
    else:
        sName = 'unknown type'
    return sName

#%% Main driver class


class M4i(Instrument):

    def __init__(self, name, cardid='spcm0', **kwargs):
        """Driver for the Spectrum M4i.44xx-x8 cards.

        For more information see: http://spectrum-instrumentation.com/en/m4i-platform-overview

        Example:

            Example usage for acquisition with channel 2 using an external trigger
            that triggers multiple times with trigger mode HIGH::

                m4 = M4i(name='M4i', server_name=None)
                m4.enable_channels(pyspcm.CHANNEL2)
                m4.set_channel_settings(2,mV_range, input_path, termination, coupling, compensation)
                m4.set_ext0_OR_trigger_settings(pyspcm.SPC_TM_HIGH,termination,coupling,level0)
                calc = m4.multiple_trigger_acquisition(mV_range,memsize,seg_size,posttrigger_size)

        Todo:
          Whenever an error occurs (including validation errors) the python
          console needs to be restarted


        """
        super().__init__(name, **kwargs)

        self.hCard = pyspcm.spcm_hOpen(cardid)
        if self.hCard is None:
            logging.warning("M4i: no card found\n")

        # add parameters for getting
        self.add_parameter('card_id',
                           label='card id',
                           parameter_class=ManualParameter,
                           initial_value=cardid,
                           vals=Anything(),
                           docstring='The card ID')
        self.add_parameter('max_sample_rate',
                           label='max sample rate',
                           unit='Hz',
                           get_cmd=self.get_max_sample_rate,
                           docstring='The maximumum sample rate')
        self.add_parameter('memory',
                           label='memory',
                           unit='bytes',
                           get_cmd=self.get_card_memory,
                           docstring='Amount of memory on card')
        self.add_parameter('resolution',
                           label='resolution',
                           unit='bits',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_MIINST_BITSPERSAMPLE),
                           docstring='Resolution of the card')
        self.add_parameter('pcidate',
                           label='pcidate',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_PCIDATE),
                           docstring='The PCI date')
        self.add_parameter('serial_number',
                           label='serial number',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_PCISERIALNO),
                           docstring='The serial number of the board')
        self.add_parameter('channel_count',
                           label='channel count',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_CHCOUNT),
                           docstring='Return number of enabled channels')
        self.add_parameter('input_path_count',
                           label='input path count',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_READAIPATHCOUNT),
                           docstring='Return number of analog input paths')
        self.add_parameter('input_ranges_count',
                           label='input ranges count',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_READIRCOUNT),
                           docstring='Return number of input ranges for the current input path')
        self.add_parameter('input_path_features',
                           label='input path features',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_READAIFEATURES),
                           docstring='Return a bitmap of features for current input path')
        self.add_parameter('available_card_modes',
                           label='available card modes',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_AVAILCARDMODES),
                           docstring='Return a bitmap of available card modes')
        self.add_parameter('card_status',
                           label='card status',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_M2STATUS),
                           docstring='Return a bitmap for the status information')
        self.add_parameter('read_range_min_0',
                           label='read range min 0', unit='mV',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_READRANGEMIN0),
                           docstring='Return the lower border of input range 0')

        # buffer handling
        self.add_parameter('user_available_length',
                           label='user available length',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_DATA_AVAIL_USER_LEN),
                           docstring='returns the number of currently to the user available bytes inside a sample data transfer')
        self.add_parameter('user_available_position',
                           label='user available position',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_DATA_AVAIL_USER_POS),
                           docstring='returns the position as byte index where the currently available data samles start')
        self.add_parameter('buffer_fill_size',
                           label='buffer fill size',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_FILLSIZEPROMILLE),
                           docstring='returns the current fill size of the on-board memory (FIFO buffer) in promille (1/1000)')

        # triggering
        self.add_parameter('available_trigger_or_mask',
                           label='available trigger or mask',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_AVAILORMASK),
                           docstring='bitmask, in which all bits of sources for the OR mask are set, if available')
        self.add_parameter('available_channel_or_mask',
                           label='available channel or mask',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_CH_AVAILORMASK0),
                           docstring='bitmask, in which all bits of sources/channels (0-31) for the OR mask are set, if available')
        self.add_parameter('available_trigger_and_mask',
                           label='available trigger and mask',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_AVAILANDMASK),
                           docstring='bitmask, in which all bits of sources for the AND mask are set, if available')
        self.add_parameter('available_channel_and_mask',
                           label='available channel and mask',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_CH_AVAILANDMASK0),
                           docstring='bitmask, in which all bits of sources/channels (0-31) for the AND mask are set, if available')
        self.add_parameter('available_trigger_delay',
                           label='available trigger delay',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_AVAILDELAY),
                           docstring='contains the maximum available delay as decimal integer value')
        self.add_parameter('available_external_trigger_modes',
                           label='available external trigger modes',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_EXT0_AVAILMODES),
                           docstring='bitmask showing all available trigger modes for external 0 (main analog trigger input)')
        self.add_parameter('external_trigger_min_level',
                           label='external trigger min level',
                           unit='mV',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_EXT_AVAIL0_MIN),
                           docstring='returns the minimum trigger level')
        self.add_parameter('external_trigger_max_level',
                           label='external trigger max level',
                           unit='mV',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_EXT_AVAIL0_MAX),
                           docstring='returns the maximum trigger level')
        self.add_parameter('external_trigger_level_step_size',
                           label='external trigger level step size',
                           unit='mV',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_EXT_AVAIL0_STEP),
                           docstring='returns the step size of the trigger level')
        self.add_parameter('available_channel_trigger_modes',
                           label='available channel trigger modes',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_CH_AVAILMODES),
                           docstring='bitmask, in which all bits of the modes for the channel trigger are set')
        self.add_parameter('trigger_counter',
                           label='trigger counter',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIGGERCOUNTER),
                           docstring='returns the number of triger events since acquisition start')
        # data per sample
        self.add_parameter('bytes_per_sample',
                           label='bytes per sample',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_MIINST_BYTESPERSAMPLE),
                           docstring='returns the number of bytes per sample')
        self.add_parameter('bits_per_sample',
                           label='bits per sample',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_MIINST_BITSPERSAMPLE),
                           docstring='returns the number of bits per sample')

        # available clock modes
        self.add_parameter('available_clock_modes',
                           label='available clock modes',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_AVAILCLOCKMODES),
                           docstring='returns a bitmask in which the bits of the clock modes are set, if available')

        # converting ADC samples to voltage values
        self.add_parameter('ADC_to_voltage',
                           label='ADC to voltage',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_MIINST_MAXADCVALUE),
                           docstring='contains the decimal code (in LSB) of the ADC full scale value')

        self.add_parameter('oversampling_factor',
                           label='oversampling factor',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_OVERSAMPLINGFACTOR),
                           docstring='Reads the oversampling factor')

        # add parameters for setting and getting (read/write direction
        # registers)

        self.add_parameter('enable_channels',
                           label='Channels enabled',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_CHENABLE),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_CHENABLE),
                           vals=Enum(1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 15),
                           docstring='Set and get enabled channels')

        # analog input path functions
        # TODO: change Enum validator to set_parser for the numbered functions
        # if we want string inputs

        self.add_parameter('read_input_path',
                           label='read input path',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_READAIPATH),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_READAIPATH),
                           vals=Enum(0, 1, 2, 3),
                           docstring='Select the input path which is used to read out the features')

        for i in [0, 1, 2, 3]:
            self.add_parameter('input_path_{}'.format(i),
                               label='input path {}'.format(i),
                               get_cmd=partial(self._param32bit, getattr(
                                   pyspcm, 'SPC_PATH{}'.format(i))),
                               set_cmd=partial(self._set_param32bit, getattr(
                                   pyspcm, 'SPC_PATH{}'.format(i))),
                               vals=Enum(0, 1),
                               docstring='Set and get analog input path for channel {}'.format(i))

            # channel range functions
            # TODO: check the input path to set the right validator (either by
            # directly calling input_path_x() or by storing a variable)
            self.add_parameter('range_channel_{}'.format(i),
                               label='range channel {}'.format(i),
                               get_cmd=partial(self._param32bit, getattr(
                                   pyspcm, 'SPC_AMP{}'.format(i))),
                               set_cmd=partial(self._set_param32bit, getattr(
                                   pyspcm, 'SPC_AMP{}'.format(i))),
                               vals=Enum(200, 500, 1000, 2000,
                                         2500, 5000, 10000),
                               docstring='Set and get input range of channel {}'.format(i))

            # input termination functions
            self.add_parameter('termination_{}'.format(i),
                               label='termination {}'.format(i),
                               get_cmd=partial(self._param32bit, getattr(
                                   pyspcm, 'SPC_50OHM{}'.format(i))),
                               set_cmd=partial(self._set_param32bit, getattr(
                                   pyspcm, 'SPC_50OHM{}'.format(i))),
                               vals=Enum(0, 1),
                               docstring='if 1 sets termination to 50 Ohm, otherwise 1 MOhm for channel {}'.format(i))

            # input coupling
            self.add_parameter('ACDC_coupling_{}'.format(i),
                               label='ACDC coupling {}'.format(i),
                               get_cmd=partial(self._param32bit, getattr(
                                   pyspcm, 'SPC_ACDC{}'.format(i))),
                               set_cmd=partial(self._set_param32bit, getattr(
                                   pyspcm, 'SPC_ACDC{}'.format(i))),
                               vals=Enum(0, 1),
                               docstring='if 1 sets the AC coupling, otherwise sets the DC coupling for channel {}'.format(i))

            # AC/DC offset compensation
            self.add_parameter('ACDC_offs_compensation_{}'.format(i),
                               label='ACDC offs compensation {}'.format(i),
                               get_cmd=partial(self._get_compensation, i),
                               set_cmd=partial(self._set_compensation, i),
                               vals=Enum(0, 1),
                               docstring='if 1 enables compensation, if 0 disables compensation for channel {}'.format(i))

            # anti aliasing filter (Bandwidth limit)
            self.add_parameter('anti_aliasing_filter_{}'.format(i),
                               label='anti aliasing filter {}'.format(i),
                               get_cmd=partial(self._param32bit, getattr(
                                   pyspcm, 'SPC_FILTER{}'.format(i))),
                               set_cmd=partial(self._set_param32bit, getattr(
                                   pyspcm, 'SPC_FILTER{}'.format(i))),
                               vals=Enum(0, 1),
                               docstring='if 1 selects bandwidth limit, if 0 sets to full bandwidth for channel {}'.format(i))

            self.add_parameter('channel_{}'.format(i),
                               label='channel {}'.format(i),
                               unit='a.u.',
                               get_cmd=partial(self._read_channel, i))

        # acquisition modes
        # TODO: If required, the other acquisition modes can be added to the
        # validator
        self.add_parameter('card_mode',
                           label='card mode',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_CARDMODE),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_CARDMODE),
                           vals=Enum(pyspcm.SPC_REC_STD_SINGLE, pyspcm.SPC_REC_STD_MULTI, pyspcm.SPC_REC_STD_GATE, pyspcm.SPC_REC_STD_ABA,
                                     pyspcm.SPC_REC_FIFO_SINGLE, pyspcm.SPC_REC_FIFO_MULTI, pyspcm.SPC_REC_FIFO_GATE, pyspcm.SPC_REC_FIFO_ABA, pyspcm.SPC_REC_STD_AVERAGE),
                           docstring='defines the used operating mode')

        # wait command
        self.add_parameter('timeout',
                           label='timeout',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TIMEOUT),
                           unit='ms',
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TIMEOUT),
                           docstring='defines the timeout for wait commands')

        # Single acquisition mode memory, pre- and posttrigger (pretrigger = memory size - posttrigger)
        # TODO: improve the validators to make them take into account the
        # current state of the instrument
        self.add_parameter('data_memory_size',
                           label='data memory size',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_MEMSIZE),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_MEMSIZE),
                           vals=Numbers(min_value=16),
                           docstring='sets the memory size in samples per channel')
        self.add_parameter('posttrigger_memory_size',
                           label='posttrigger memory size',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_POSTTRIGGER),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_POSTTRIGGER),
                           docstring='sets the number of samples to be recorded after trigger event')

        # FIFO single acquisition length and pretrigger
        self.add_parameter('pretrigger_memory_size',
                           label='pretrigger memory size',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_PRETRIGGER),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_PRETRIGGER),
                           docstring='sets the number of samples to be recorded before trigger event')
        self.add_parameter('segment_size',
                           label='segment size',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_SEGMENTSIZE),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_SEGMENTSIZE),
                           docstring='length of segments to acquire')
        self.add_parameter('total_segments',
                           label='total segments',
                           get_cmd=partial(self._param32bit, pyspcm.SPC_LOOPS),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_LOOPS),
                           docstring='number of segments to acquire in total. Setting 0 makes it run until stopped by user')

        # clock generation
        self.add_parameter('clock_mode',
                           label='clock mode',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_CLOCKMODE),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_CLOCKMODE),
                           vals=Enum(pyspcm.SPC_CM_INTPLL, pyspcm.SPC_CM_QUARTZ2,
                                     pyspcm.SPC_CM_EXTREFCLOCK, pyspcm.SPC_CM_PXIREFCLOCK),
                           docstring='defines the used clock mode or reads out the actual selected one')
        self.add_parameter('sample_rate',
                           label='sample rate',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_SAMPLERATE),
                           unit='Hz',
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_SAMPLERATE),
                           docstring='write the sample rate for internal sample generation or read rate nearest to desired')
        self.add_parameter('special_clock',
                           label='special clock',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_SPECIALCLOCK),
                           unit='Hz',
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_SPECIALCLOCK),
                           docstring='Activate/Deactivate the special clock mode (lower and more sampling clock rates)')

        # triggering
        self.add_parameter('trigger_or_mask',
                           label='trigger or mask',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_ORMASK),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TRIG_ORMASK),
                           vals=Enum(pyspcm.SPC_TMASK_NONE, pyspcm.SPC_TMASK_SOFTWARE,
                                     pyspcm.SPC_TMASK_EXT0, pyspcm.SPC_TMASK_EXT1),
                           docstring='defines the events included within the  trigger OR mask card')
        self.add_parameter('channel_or_mask',
                           label='channel or mask',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_CH_ORMASK0),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TRIG_CH_ORMASK0),
                           docstring='includes the channels (0-31) within the channel trigger OR mask of the card')
        self.add_parameter('trigger_and_mask',
                           label='trigger and mask',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_ANDMASK),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TRIG_ANDMASK),
                           vals=Enum(pyspcm.SPC_TMASK_NONE,
                                     pyspcm.SPC_TMASK_EXT0, pyspcm.SPC_TMASK_EXT1),
                           docstring='defines the events included within the  trigger AND mask card')
        self.add_parameter('channel_and_mask',
                           label='channel and mask',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_CH_ANDMASK0),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TRIG_CH_ANDMASK0),
                           docstring='includes the channels (0-31) within the channel trigger AND mask of the card')
        self.add_parameter('trigger_delay',
                           label='trigger delay',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_DELAY),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TRIG_DELAY),
                           docstring='defines the delay for the detected trigger events')
        self.add_parameter('external_trigger_mode',
                           label='external trigger mode',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_EXT0_MODE),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TRIG_EXT0_MODE),
                           docstring='defines the external trigger mode for the external SMA connector trigger input')
        self.add_parameter('external_trigger_termination',
                           label='external trigger termination',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_TERM),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TRIG_TERM),
                           vals=Enum(0, 1),
                           docstring='A 1 sets the 50 Ohm termination, a 0 sets high impedance termination')
        self.add_parameter('external_trigger_input_coupling',
                           label='external trigger input coupling',
                           get_cmd=partial(self._param32bit,
                                           pyspcm.SPC_TRIG_EXT0_ACDC),
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_TRIG_EXT0_ACDC),
                           vals=Enum(0, 1),
                           docstring='A 1 sets the AC coupling for the external trigger, a 0 sets DC')

        for l in [0, 1]:
            self.add_parameter('external_trigger_level_{}'.format(l),
                               label='external trigger level {}'.format(l),
                               get_cmd=partial(self._param32bit, getattr(
                                   pyspcm, 'SPC_TRIG_EXT0_LEVEL{}'.format(l))),
                               set_cmd=partial(self._set_param32bit, getattr(
                                   pyspcm, 'SPC_TRIG_EXT0_LEVEL{}'.format(l))),
                               docstring='trigger level {} for external trigger'.format(l))

        for i in [0, 1, 2, 3]:
            self.add_parameter('trigger_mode_channel_{}'.format(i),
                               label='trigger mode channel {}'.format(i),
                               get_cmd=partial(self._param32bit, getattr(
                                   pyspcm, 'SPC_TRIG_CH{}_MODE'.format(i))),
                               set_cmd=partial(self._set_param32bit, getattr(
                                   pyspcm, 'SPC_TRIG_CH{}_MODE'.format(i))),
                               docstring='sets the trigger mode for channel {}'.format(i))
            for l in [0, 1]:
                self.add_parameter('trigger_channel_{}_level_{}'.format(i, l),
                                   label='trigger channel {} level {}'.format(
                                       i, l),
                                   get_cmd=partial(self._param32bit, getattr(
                                       pyspcm, 'SPC_TRIG_CH{}_LEVEL{}'.format(i, l))),
                                   set_cmd=partial(self._set_param32bit, getattr(
                                       pyspcm, 'SPC_TRIG_CH{}_LEVEL{}'.format(i, l))),
                                   docstring='trigger level {} channel {}'.format(l, i))

        # add parameters for setting (write only registers)

        # Buffer handling
        self.add_parameter('card_available_length',
                           label='card available length',
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_DATA_AVAIL_CARD_LEN),
                           docstring='writes the number of bytes that the card can now use for sample data transfer again')

        # General
        self.add_parameter('general_command',
                           label='general command',
                           set_cmd=partial(self._set_param32bit,
                                           pyspcm.SPC_M2CMD),
                           docstring='executes a command for the card or data transfer')

        # memsize used for simple channel read-out
        self._channel_memsize = 2**12

    # checks if requirements for the compensation get and set functions are met
    def _get_compensation(self, i):
        # if HF enabled
        if(getattr(self, 'input_path_{}'.format(i))() == 1):
            self._param32bit(
                getattr(pyspcm, 'SPC_ACDC_OFFS_COMPENSATION{}'.format(i)))
        else:
            logging.warning(
                "M4i: HF path not set, ignoring ACDC offset compensation get\n")

    def _set_compensation(self, i, value):
        # if HF enabled
        if(getattr(self, 'input_path_{}'.format(i))() == 1):
            self._set_param32bit(
                getattr(pyspcm, 'SPC_ACDC_OFFS_COMPENSATION{}'.format(i)), value)
        else:
            logging.warning(
                "M4i: HF path not set, ignoring ACDC offset compensation set\n")

    def active_channels(self):
        """ Return a list with the indices of the active channels """
        x = bin(self.enable_channels())[2:]
        return [i for i in range(len(x)) if x[i]]

    def get_idn(self):
        return dict(zip(('vendor', 'model', 'serial', 'firmware'), ('Spectrum_GMBH', szTypeToName(self.get_card_type()), self.serial_number(), ' ')))

    def convert_to_voltage(self, data, input_range):
        """convert an array of numbers to an array of voltages."""
        resolution = self.ADC_to_voltage()
        return data * input_range / resolution

    def initialize_channels(self, channels=None, mV_range=1000, input_path=0,
                            termination=0, coupling=0, compensation=None, memsize=2**12):
        """ Setup channels of the digitizer for simple readout using Parameters

        The channels can be read out using the Parmeters `channel_0`, `channel_1`, ...

        Args:
            channels (list): list of channels to setup
            mV_range, input_path, termination, coupling, compensation: passed
                to the set_channel_settings function
            memsize (int): memory size to use for simple channel readout
        """
        allchannels = 0
        self._channel_memsize = memsize
        self.data_memory_size(memsize)
        if channels is None:
            channels = range(4)
        for ch in channels:
            self.set_channel_settings(ch, mV_range, input_path=input_path,
                                      termination=termination, coupling=coupling, compensation=compensation)
            allchannels = allchannels + getattr(pyspcm, 'CHANNEL%d' % ch)

        self.enable_channels(allchannels)

    def _channel_mask(self, channels=range(4)):
        """ Return mask for specified channels

        Args:
            channels (list): list of channel indices
        Returns:
            cx (int): channel mask
        """
        cx = 0
        for c in channels:
            cx += getattr(pyspcm, 'CHANNEL{}'.format(c))
        return cx

    def _read_channel(self, channel, memsize=None):
        """ Helper function to read out a channel

        Before a channel is measured all channels are enabled to ensure we can
        read out channels without the overhead of changing channels.
        """
        if memsize is None:
            memsize = self._channel_memsize
        posttrigger_size = int(memsize / 2)
        mV_range = getattr(self, 'range_channel_%d' % channel).get()
        cx = self._channel_mask()
        self.enable_channels(cx)
        data = self.single_software_trigger_acquisition(
            mV_range, memsize, posttrigger_size)
        active = self.active_channels()
        data = data.reshape((-1, len(active)))
        value = np.mean(data[:, channel])
        return value

    def set_channel_settings(self, i, mV_range, input_path, termination, coupling, compensation=None):
        # initialize
        getattr(self, 'input_path_{}'.format(i))(input_path)  # 0: 1 MOhm
        getattr(self, 'termination_{}'.format(i))(termination)  # 0: DC
        getattr(self, 'ACDC_coupling_{}'.format(i))(coupling)  # 0: DC
        getattr(self, 'range_channel_{}'.format(i))(
            mV_range)  # note: set after voltage range
        # can only be used with DC coupling and 50 Ohm path (hf)
        if compensation is not None:
            getattr(self, 'ACDC_offs_compensation_{}'.format(i))(compensation)

    def set_ext0_OR_trigger_settings(self, trig_mode, termination, coupling, level0, level1=None):

        self.channel_or_mask(0)
        self.external_trigger_mode(trig_mode)  # trigger mode
        self.trigger_or_mask(pyspcm.SPC_TMASK_EXT0)  # external trigger
        self.external_trigger_termination(termination)  # 1: 50 Ohm
        self.external_trigger_input_coupling(coupling)  # 0: DC
        self.external_trigger_level_0(level0)  # mV
        if(level1 != None):
            self.external_trigger_level_1(level1)  # mV

    # Note: the levels need to be set in bits, not voltages! (between -8191 to
    # 8191 for 14 bits)
    def set_channel_OR_trigger_settings(self, i, trig_mode, bitlevel0, bitlevel1=None):
        """When a channel is used for triggering it must be enabled during the
        acquisition."""
        self.trigger_or_mask(0)
        self.channel_or_mask(getattr(pyspcm, 'SPC_TMASK0_CH{}'.format(i)))
        getattr(self, 'trigger_channel_{}_level_0'.format(i))(bitlevel0)
        if(bitlevel1 != None):
            getattr(self, 'trigger_channel_{}_level_1'.format(i))(bitlevel1)
        getattr(self, 'trigger_mode_channel_{}'.format(i))(
            trig_mode)  # trigger mode

    def _stop_acquisition(self):

        # close acquisition
        self.general_command(pyspcm.M2CMD_DATA_STOPDMA)

        # invalidate buffer
        self._invalidate_buf(pyspcm.SPCM_BUF_DATA)

        self.general_command(pyspcm.M2CMD_CARD_STOP)

    # TODO: if multiple channels are used at the same time, the voltage conversion needs to be updated
    # TODO: the data also needs to be organized nicely (currently it
    # interleaves the data)
    def multiple_trigger_acquisition(self, mV_range, memsize, seg_size, posttrigger_size):

        self.card_mode(pyspcm.SPC_REC_STD_MULTI)  # multi

        self.data_memory_size(memsize)
        self.segment_size(seg_size)
        self.posttrigger_memory_size(posttrigger_size)
        numch = bin(self.enable_channels()).count("1")

        self.general_command(pyspcm.M2CMD_CARD_START |
                             pyspcm.M2CMD_CARD_ENABLETRIGGER | pyspcm.M2CMD_CARD_WAITREADY)

        # setup software buffer
        buffer_size = ct.c_int16 * memsize * numch
        data_buffer = (buffer_size)()
        data_pointer = ct.cast(data_buffer, ct.c_void_p)

        # data acquisition
        self._def_transfer64bit(
            pyspcm.SPCM_BUF_DATA, pyspcm.SPCM_DIR_CARDTOPC, 0, data_pointer, 0, 2 * memsize * numch)
        self.general_command(pyspcm.M2CMD_DATA_STARTDMA |
                             pyspcm.M2CMD_DATA_WAITDMA)

        # convert buffer to numpy array
        data = ct.cast(data_pointer, ct.POINTER(buffer_size))
        output = np.frombuffer(data.contents, dtype=ct.c_int16)

        self._stop_acquisition()

        voltages = self.convert_to_voltage(output, mV_range / 1000)

        return voltages

    def single_trigger_acquisition(self, mV_range, memsize, posttrigger_size):

        self.card_mode(pyspcm.SPC_REC_STD_SINGLE)  # single

        # set memsize and posttrigger
        self.data_memory_size(memsize)
        self.posttrigger_memory_size(posttrigger_size)
        numch = bin(self.enable_channels()).count("1")

        self.general_command(pyspcm.M2CMD_CARD_START |
                             pyspcm.M2CMD_CARD_ENABLETRIGGER | pyspcm.M2CMD_CARD_WAITREADY)

        # setup software buffer
        buffer_size = ct.c_int16 * memsize * numch
        data_buffer = (buffer_size)()
        data_pointer = ct.cast(data_buffer, ct.c_void_p)

        # data acquisition
        self._def_transfer64bit(
            pyspcm.SPCM_BUF_DATA, pyspcm.SPCM_DIR_CARDTOPC, 0, data_pointer, 0, 2 * memsize * numch)
        self.general_command(pyspcm.M2CMD_DATA_STARTDMA |
                             pyspcm.M2CMD_DATA_WAITDMA)

        # convert buffer to numpy array
        data = ct.cast(data_pointer, ct.POINTER(buffer_size))
        output = np.frombuffer(data.contents, dtype=ct.c_int16)

        self._stop_acquisition()

        voltages = self.convert_to_voltage(output, mV_range / 1000)

        return voltages

    def gated_trigger_acquisition(self, mV_range, memsize, pretrigger_size, posttrigger_size):
        """doesn't work completely as expected, it triggers even when the
        trigger level is set outside of the signal range it also seems to
        additionally acquire some wrong parts of the wave, but this also exists
        in SBench6, so it is not a problem caused by this code."""

        self.card_mode(pyspcm.SPC_REC_STD_GATE)  # gated

        # set memsize and posttrigger
        self.data_memory_size(memsize)
        self.pretrigger_memory_size(pretrigger_size)
        self.posttrigger_memory_size(posttrigger_size)
        numch = bin(self.enable_channels()).count("1")

        self.general_command(pyspcm.M2CMD_CARD_START |
                             pyspcm.M2CMD_CARD_ENABLETRIGGER | pyspcm.M2CMD_CARD_WAITREADY)

        # setup software buffer
        buffer_size = ct.c_int16 * memsize * numch
        data_buffer = (buffer_size)()
        data_pointer = ct.cast(data_buffer, ct.c_void_p)

        # data acquisition
        self._def_transfer64bit(
            pyspcm.SPCM_BUF_DATA, pyspcm.SPCM_DIR_CARDTOPC, 0, data_pointer, 0, 2 * memsize * numch)
        self.general_command(pyspcm.M2CMD_DATA_STARTDMA |
                             pyspcm.M2CMD_DATA_WAITDMA)

        # convert buffer to numpy array
        data = ct.cast(data_pointer, ct.POINTER(buffer_size))
        output = np.frombuffer(data.contents, dtype=ct.c_int16)

        self._stop_acquisition()

        voltages = self.convert_to_voltage(output, mV_range / 1000)

        return voltages

    def single_software_trigger_acquisition(self, mV_range, memsize, posttrigger_size):
        """ Acquire a single data trace

        Args:
            mV_range
            memsize (int): size of data trace
            posttrigger_size (int): size of data trace after triggering
        Returns:
            voltages (array)
        """
        self.card_mode(pyspcm.SPC_REC_STD_SINGLE)  # single

        self.data_memory_size(memsize)
        self.posttrigger_memory_size(posttrigger_size)
        numch = bin(self.enable_channels()).count("1")

        # start/enable trigger/wait ready
        self.trigger_or_mask(pyspcm.SPC_TMASK_SOFTWARE)  # software trigger
        self.general_command(pyspcm.M2CMD_CARD_START |
                             pyspcm.M2CMD_CARD_ENABLETRIGGER | pyspcm.M2CMD_CARD_WAITREADY)

        # setup software buffer
        buffer_size = ct.c_int16 * memsize * numch
        data_buffer = (buffer_size)()
        data_pointer = ct.cast(data_buffer, ct.c_void_p)

        # data acquisition
        self._def_transfer64bit(
            pyspcm.SPCM_BUF_DATA, pyspcm.SPCM_DIR_CARDTOPC, 0, data_pointer, 0, 2 * memsize * numch)
        self.general_command(pyspcm.M2CMD_DATA_STARTDMA |
                             pyspcm.M2CMD_DATA_WAITDMA)

        # convert buffer to numpy array
        data = ct.cast(data_pointer, ct.POINTER(buffer_size))
        output = np.frombuffer(data.contents, dtype=ct.c_int16)
        self._debug = output
        self._stop_acquisition()

        voltages = self.convert_to_voltage(output, mV_range / 1000)

        return voltages

    def _check_buffers(self):
        """ Check validity of buffers

        See: manual section "Limits of pre trigger, post trigger, memory size"
        """

        pretrigger = self.data_memory_size() - self.posttrigger_memory_size()
        if pretrigger > 2**13:
            raise Exception('value of SPC_PRETRIGGER is invalid')

    def blockavg_hardware_trigger_acquisition(self, mV_range, nr_averages=10,
                                              verbose=0, post_trigger=None):
        """ Acquire data using block averaging and hardware triggering

        To read out multiple channels, use `initialize_channels`

        Args:
            mV_range (float)
            nr_averages (int): number of averages to take
            verbose (int): output level
            post_trigger (None or int): optional size of post_trigger buffer
        Returns:
            voltages (array): if multiple channels are read, then the data is interleaved
        """
        # self.available_card_modes()
        if nr_averages < 2:
            raise Exception('averaging for less than 2 times is not supported')
        self.card_mode(pyspcm.SPC_REC_STD_AVERAGE)  # single
        memsize = self.data_memory_size()
        self.segment_size(memsize)

        if post_trigger is None:
            pre_trigger = min(2**13, memsize / 2)
            post_trigger = memsize - pre_trigger
        else:
            pre_trigger = memsize - post_trigger
        self.posttrigger_memory_size(post_trigger)
        self.pretrigger_memory_size(pre_trigger)

        self._check_buffers()

        self._set_param32bit(pyspcm.SPC_AVERAGES, nr_averages)
        numch = bin(self.enable_channels()).count("1")

        if verbose:
            print('blockavg_hardware_trigger_acquisition: errors %s' %
                  (self.get_error_info32bit(), ))
            print('blockavg_hardware_trigger_acquisition: card_status %s' %
                  (self.card_status(), ))

        self.external_trigger_mode(pyspcm.SPC_TM_POS)
        self.trigger_or_mask(pyspcm.SPC_TMASK_EXT0)
        self.general_command(pyspcm.M2CMD_CARD_START |
                             pyspcm.M2CMD_CARD_ENABLETRIGGER | pyspcm.M2CMD_CARD_WAITREADY)

        # setup software buffer
        sizeof32bit = 4
        buffer_size = ct.c_int32 * memsize * numch
        data_buffer = (buffer_size)()
        data_pointer = ct.cast(data_buffer, ct.c_void_p)

        # data acquisition
        self._def_transfer64bit(
            pyspcm.SPCM_BUF_DATA, pyspcm.SPCM_DIR_CARDTOPC, 0, data_pointer, 0, sizeof32bit * memsize * numch)
        self.general_command(pyspcm.M2CMD_DATA_STARTDMA |
                             pyspcm.M2CMD_DATA_WAITDMA)

        # convert buffer to numpy array
        data = ct.cast(data_pointer, ct.POINTER(buffer_size))
        output = np.frombuffer(data.contents, dtype=ct.c_int32) / nr_averages
        self._debug = output

        self._stop_acquisition()

        voltages = self.convert_to_voltage(output, mV_range / 1000)

        return voltages

    def close(self):
        """Close handle to the card."""
        if self.hCard is not None:
            pyspcm.spcm_vClose(self.hCard)
            self.hCard = None
        super().close()

    def get_card_type(self, verbose=0):
        """Read card type."""
        # read type, function and sn and check for D/A card
        lCardType = pyspcm.int32(0)
        pyspcm.spcm_dwGetParam_i32(
            self.hCard, pyspcm.SPC_PCITYP, pyspcm.byref(lCardType))
        if verbose:
            print('card_type: %s' % szTypeToName(lCardType.value))
        return (lCardType.value)

    # only works if the error was not caused by running the entire program
    # (and therefore making a new M4i object)
    def get_error_info32bit(self):
        """Read an error from the error register."""
        dwErrorReg = pyspcm.uint32(0)
        lErrorValue = pyspcm.int32(0)

        pyspcm.spcm_dwGetErrorInfo_i32(self.hCard, pyspcm.byref(
            dwErrorReg), pyspcm.byref(lErrorValue), None)
        return (dwErrorReg.value, lErrorValue.value)

    def _param64bit(self, param):
        """Read a 64-bit parameter from the device."""
        data = pyspcm.int64(0)
        pyspcm.spcm_dwGetParam_i64(self.hCard, param, pyspcm.byref(data))
        return (data.value)

    def _param32bit(self, param):
        """Read a 32-bit parameter from the device."""
        data = pyspcm.int32(0)
        pyspcm.spcm_dwGetParam_i32(self.hCard, param, pyspcm.byref(data))
        return (data.value)

    def _set_param32bit(self, param, value):
        """ Set a 32-bit parameter on the device."""
        value = int(value)  # convert floating point to int if necessary
        pyspcm.spcm_dwSetParam_i32(self.hCard, param, value)

    def _invalidate_buf(self, buf_type):
        """Invalidate device buffer."""
        pyspcm.spcm_dwInvalidateBuf(self.hCard, buf_type)

    def _def_transfer64bit(self, buffer_type, direction, bytes_till_event, data_pointer, offset, buffer_length):
        """Define a 64-bit transer between the device and the computer."""
        pyspcm.spcm_dwDefTransfer_i64(
            self.hCard, buffer_type, direction, bytes_till_event, data_pointer, offset, buffer_length)

    def get_max_sample_rate(self, verbose=0):
        """Return max sample rate."""
        # read type, function and sn and check for D/A card
        value = self._param32bit(pyspcm.SPC_PCISAMPLERATE)
        if verbose:
            print('max_sample_rate: %s' % (value))
        return value

    def get_card_memory(self, verbose=0):
        data = pyspcm.int64(0)
        pyspcm.spcm_dwGetParam_i64(
            self.hCard, pyspcm.SPC_PCIMEMSIZE, pyspcm.byref(data))
        if verbose:
            print('card_memory: %s' % (data.value))
        return (data.value)
