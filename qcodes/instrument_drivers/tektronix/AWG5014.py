# Tektronix_AWG5014.py class, to perform the communication between the Wrapper
# and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <mcschaafsma@gmail.com>, 2008
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import numpy as np
import struct
from time import sleep, time, localtime
from io import BytesIO
import os
import logging
import array as arr

from qcodes import VisaInstrument, validators as vals


def parsestr(v):
    return v.strip().strip('"')


class Tektronix_AWG5014(VisaInstrument):
    '''
    This is the python driver for the Tektronix AWG5014
    Arbitrary Waveform Generator

    think about:    clock, waveform length

    TODO:
    - Not all functionality is available in the driver
    - There is some double functionality
    - There are some inconsistensies between the name of a parameter and the
      name of the same variable in the tektronix manual

    * Remove test_send??
    * Sequence element (SQEL) parameter functions exist but no corresponding
      parameters


    CHANGES:
    26-11-2008 by Gijs: Copied this plugin from the 520 and added support for
        2 more channels, added set get marker delay functions and increased max
        sampling freq to 1.2 	GS/s
    28-11-2008 ''  '' : Added some functionality to manipulate and manoeuvre
        through the folders on the AWG
    8-8-2015 by Adriaan : Merging the now diverged versions of this driver from
        the Diamond and Transmon groups @ TUD
    7-1-2016 Converted to use with QCodes
    '''
    AWG_FILE_FORMAT_HEAD = {
        'SAMPLING_RATE': 'd',    # d
        'REPETITION_RATE': 'd',    # # NAME?
        'HOLD_REPETITION_RATE': 'h',    # True | False
        'CLOCK_SOURCE': 'h',    # Internal | External
        'REFERENCE_SOURCE': 'h',    # Internal | External
        'EXTERNAL_REFERENCE_TYPE': 'h',    # Fixed | Variable
        'REFERENCE_CLOCK_FREQUENCY_SELECTION': 'h',
        'REFERENCE_MULTIPLIER_RATE': 'h',    #
        'DIVIDER_RATE': 'h',   # 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256
        'TRIGGER_SOURCE': 'h',    # Internal | External
        'INTERNAL_TRIGGER_RATE': 'd',    #
        'TRIGGER_INPUT_IMPEDANCE': 'h',    # 50 ohm | 1 kohm
        'TRIGGER_INPUT_SLOPE': 'h',    # Positive | Negative
        'TRIGGER_INPUT_POLARITY': 'h',    # Positive | Negative
        'TRIGGER_INPUT_THRESHOLD': 'd',    #
        'EVENT_INPUT_IMPEDANCE': 'h',    # 50 ohm | 1 kohm
        'EVENT_INPUT_POLARITY': 'h',    # Positive | Negative
        'EVENT_INPUT_THRESHOLD': 'd',
        'JUMP_TIMING': 'h',    # Sync | Async
        'INTERLEAVE': 'h',    # On |  This setting is stronger than .
        'ZEROING': 'h',    # On | Off
        'COUPLING': 'h',    # The Off | Pair | All setting is weaker than .
        'RUN_MODE': 'h',    # Continuous | Triggered | Gated | Sequence
        'WAIT_VALUE': 'h',    # First | Last
        'RUN_STATE': 'h',    # On | Off
        'INTERLEAVE_ADJ_PHASE': 'd',
        'INTERLEAVE_ADJ_AMPLITUDE': 'd',
    }
    AWG_FILE_FORMAT_CHANNEL = {
        # Include NULL.(Output Waveform Name for Non-Sequence mode)
        'OUTPUT_WAVEFORM_NAME_N': 's',
        'CHANNEL_STATE_N': 'h',  # On | Off
        'ANALOG_DIRECT_OUTPUT_N': 'h',  # On | Off
        'ANALOG_FILTER_N': 'h',  # Enum type.
        'ANALOG_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        # When the Input Method is High/Low, it is skipped.
        'ANALOG_AMPLITUDE_N': 'd',
        # When the Input Method is High/Low, it is skipped.
        'ANALOG_OFFSET_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'ANALOG_HIGH_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'ANALOG_LOW_N': 'd',
        'MARKER1_SKEW_N': 'd',
        'MARKER1_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        # When the Input Method is High/Low, it is skipped.
        'MARKER1_AMPLITUDE_N': 'd',
        # When the Input Method is High/Low, it is skipped.
        'MARKER1_OFFSET_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER1_HIGH_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER1_LOW_N': 'd',
        'MARKER2_SKEW_N': 'd',
        'MARKER2_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        # When the Input Method is High/Low, it is skipped.
        'MARKER2_AMPLITUDE_N': 'd',
        # When the Input Method is High/Low, it is skipped.
        'MARKER2_OFFSET_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER2_HIGH_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER2_LOW_N': 'd',
        'DIGITAL_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        # When the Input Method is High/Low, it is skipped.
        'DIGITAL_AMPLITUDE_N': 'd',
        # When the Input Method is High/Low, it is skipped.
        'DIGITAL_OFFSET_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'DIGITAL_HIGH_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'DIGITAL_LOW_N': 'd',
        'EXTERNAL_ADD_N': 'h',  # AWG5000 only
        'PHASE_DELAY_INPUT_METHOD_N':   'h',  # Phase/DelayInme/DelayInints
        'PHASE_N': 'd',  # When the Input Method is not Phase, it is skipped.
        # When the Input Method is not DelayInTime, it is skipped.
        'DELAY_IN_TIME_N': 'd',
        # When the Input Method is not DelayInPoint, it is skipped.
        'DELAY_IN_POINTS_N': 'd',
        'CHANNEL_SKEW_N': 'd',
        'DC_OUTPUT_LEVEL_N': 'd',  # V
    }

    def __init__(self, name, setup_folder, address, reset=False,
                 clock=1e9, numpoints=1000, timeout=180, **kwargs):
        '''
        Initializes the AWG5014.

        Input:
            name (string)           : name of the instrument
            setup_folder (string)   : folder where externally generate seqs
                                        are stored
            address (string)        : GPIB or ethernet address
            reset (bool)            : resets to default values, default=false
            numpoints (int)         : sets the number of datapoints
            timeout (float)         : visa timeout, in secs. long default (180)
                                        to accommodate large waveforms

        Output:
            None
        '''
        super().__init__(name, address, timeout=timeout, **kwargs)

        self._address = address

        self._values = {}
        self._values['files'] = {}
        self._clock = clock
        self._numpoints = numpoints

        self.add_function('reset', call_cmd='*RST')

        self.add_parameter('state',
                           get_cmd=self.get_state)
        self.add_parameter('run_mode',
                           get_cmd='AWGC:RMOD?',
                           set_cmd='AWGC:RMOD ' + '{}',
                           vals=vals.Enum('CONT', 'TRIG', 'SEQ', 'GAT'))
        # Trigger parameters #
        # ! Warning this is the same as run mode, do not use! exists
        # Solely for legacy purposes
        self.add_parameter('trigger_mode',
                           get_cmd='AWGC:RMOD?',
                           set_cmd='AWGC:RMOD ' + '{}',
                           vals=vals.Enum('CONT', 'TRIG', 'SEQ', 'GAT'))
        self.add_parameter('trigger_impedance',
                           label='Trigger impedance (Ohm)',
                           units='Ohm',
                           get_cmd='TRIG:IMP?',
                           set_cmd='TRIG:IMP ' + '{}',
                           vals=vals.Enum(50, 1000),
                           get_parser=float)
        self.add_parameter('trigger_level',
                           units='V',
                           label='Trigger level (V)',
                           get_cmd='TRIG:LEV?',
                           set_cmd='TRIG:LEV ' + '{:.3f}',
                           vals=vals.Numbers(-5, 5),
                           get_parser=float)
        self.add_parameter('trigger_slope',
                           get_cmd='TRIG:SLOP?',
                           set_cmd='TRIG:SLOP ' + '{}',
                           vals=vals.Enum('POS', 'NEG'))  # ,
                           # get_parser=self.parse_int_pos_neg)
        self.add_parameter('trigger_source',
                           get_cmd='TRIG:source?',
                           set_cmd='TRIG:source ' + '{}',
                           vals=vals.Enum('INT', 'EXT'))
        # Event parameters #
        self.add_parameter('event_polarity',
                           get_cmd='EVEN:POL?',
                           set_cmd='EVEN:POL ' + '{}',
                           vals=vals.Enum('POS', 'NEG'))
        self.add_parameter('event_impedance',
                           label='Event impedance (Ohm)',
                           get_cmd='EVEN:IMP?',
                           set_cmd='EVEN:IMP ' + '{}',
                           vals=vals.Enum(50, 1000),
                           get_parser=float)
        self.add_parameter('event_level',
                           label='Event level (V)',
                           get_cmd='EVEN:LEV?',
                           set_cmd='EVEN:LEV ' + '{:.3f}',
                           vals=vals.Numbers(-5, 5),
                           get_parser=float)
        self.add_parameter('event_jump_timing',
                           get_cmd='EVEN:JTIM?',
                           set_cmd='EVEN:JTIM {}',
                           vals=vals.Enum('SYNC', 'ASYNC'))

        self.add_parameter('clock_freq',
                           label='Clock frequency (Hz)',
                           get_cmd='SOUR:FREQ?',
                           set_cmd='SOUR:FREQ ' + '{}',
                           vals=vals.Numbers(1e6, 1.2e9),
                           get_parser=float)

        self.add_parameter('numpoints',
                           label='Number of datapoints per wave',
                           get_cmd=self._do_get_numpoints,
                           set_cmd=self._do_set_numpoints,
                           vals=vals.Ints(100, int(1e9)))
        self.add_parameter('setup_filename',
                           get_cmd='AWGC:SNAM?')
                           # set_cmd=self.do_set_setup_filename,
                           # vals=vals.Strings())
                           # set function has optional args and therefore
                           # does not work with QCodes

        # Channel parameters #
        for i in range(1, 5):
            amp_cmd = 'SOUR{}:VOLT:LEV:IMM:AMPL'.format(i)
            offset_cmd = 'SOUR{}:VOLT:LEV:IMM:OFFS'.format(i)
            state_cmd = 'OUTPUT{}:STATE'.format(i)
            waveform_cmd = 'SOUR{}:WAV'.format(i)
            # Set channel first to ensure sensible sorting of pars
            self.add_parameter('ch{}_state'.format(i),
                               label='Status channel {}'.format(i),
                               get_cmd=state_cmd + '?',
                               set_cmd=state_cmd + ' {}',
                               vals=vals.Ints(0, 1))
            self.add_parameter('ch{}_amp'.format(i),
                               label='Amplitude channel {} (Vpp)'.format(i),
                               units='Vpp',
                               get_cmd=amp_cmd + '?',
                               set_cmd=amp_cmd + ' {:.6f}',
                               vals=vals.Numbers(0.02, 4.5),
                               get_parser=float)
            self.add_parameter('ch{}_offset'.format(i),
                               label='Offset channel {} (V)'.format(i),
                               units='V',
                               get_cmd=offset_cmd + '?',
                               set_cmd=offset_cmd + ' {:.3f}',
                               vals=vals.Numbers(-.1, .1),
                               get_parser=float)
            self.add_parameter('ch{}_waveform'.format(i),
                               label='Waveform channel {}'.format(i),
                               get_cmd=waveform_cmd + '?',
                               set_cmd=waveform_cmd + ' "{}"',
                               vals=vals.Strings(),
                               get_parser=parsestr)
            # Marker channels
            for j in range(1, 3):
                m_del_cmd = 'SOUR{}:MARK{}:DEL'.format(i, j)
                m_high_cmd = 'SOUR{}:MARK{}:VOLT:LEV:IMM:HIGH'.format(i, j)
                m_low_cmd = 'SOUR{}:MARK{}:VOLT:LEV:IMM:LOW'.format(i, j)

                self.add_parameter(
                    'ch{}_m{}_del'.format(i, j),
                    label='Channel {} Marker {} delay (ns)'.format(i, j),
                    get_cmd=m_del_cmd + '?',
                    set_cmd=m_del_cmd + ' {:.3f}e-9',
                    vals=vals.Numbers(0, 1),
                    get_parser=float)
                self.add_parameter(
                    'ch{}_m{}_high'.format(i, j),
                    label='Channel {} Marker {} high level (V)'.format(i, j),
                    get_cmd=m_high_cmd + '?',
                    set_cmd=m_high_cmd + ' {:.3f}',
                    vals=vals.Numbers(-2.7, 2.7),
                    get_parser=float)
                self.add_parameter(
                    'ch{}_m{}_low'.format(i, j),
                    label='Channel {} Marker {} low level (V)'.format(i, j),
                    get_cmd=m_low_cmd + '?',
                    set_cmd=m_low_cmd + ' {:.3f}',
                    vals=vals.Numbers(-2.7, 2.7),
                    get_parser=float)

        # # Add functions

        # self.add_function('get_state')
        # self.add_function('set_event_jump_timing')
        # self.add_function('get_event_jump_timing')
        # self.add_function('generate_awg_file')
        # self.add_function('send_awg_file')
        # self.add_function('load_awg_file')
        # self.add_function('get_error')
        # self.add_function('pack_waveform')
        # self.add_function('clear_visa')
        # self.add_function('initialize_dc_waveforms')

        # # Setup filepaths
        self.waveform_folder = "Waveforms"
        self._rem_file_path = "Z:\\Waveforms\\"

        # NOTE! this directory has to exist on the AWG!!
        self._setup_folder = setup_folder

        self.goto_root()
        self.change_folder(self.waveform_folder)

        self.set('trigger_impedance', 50)
        if self.get('clock_freq') != 1e9:
            logging.warning('AWG clock freq not set to 1GHz')

        self.connect_message()

    # Functions
    def get_all(self, update=True):
        return self.snapshot(update=update)

    def get_state(self):
        state = self.ask('AWGC:RSTATE?')
        if state.startswith('0'):
            return 'Idle'
        elif state.startswith('1'):
            return 'Waiting for trigger'
        elif state.startswith('2'):
            return 'Running'
        else:
            raise ValueError(__name__ + ' : AWG in undefined state "%s"' %
                             state)

    def start(self):
        '''
        Convenience function, identical to run()
        '''
        return self.run()

    def run(self):
        self.write('AWGC:RUN')
        return self.get_state()

    def stop(self):
        self.write('AWGC:STOP')

    def get_folder_contents(self, print_contents=True):
        if print_contents:
            print('Current folder:', self.get_current_folder_name())
            print(self.ask('MMEM:CAT?')
                  .replace(',"$', '\n$').replace('","', '\n')
                  .replace(',', '\t'))
        return self.ask('mmem:cat?')

    def get_current_folder_name(self):
        return self.ask('mmem:cdir?')

    def set_current_folder_name(self, file_path):
        return self.write('mmem:cdir "%s"' % file_path)

    def change_folder(self, dir):
        return self.write('mmem:cdir "\%s"' % dir)

    def goto_root(self):
        return self.write('mmem:cdir "c:\\.."')

    def create_and_goto_dir(self, dir):
        '''
        Creates (if not yet present) and sets the current directory to <dir>
        and displays the contents

        '''

        dircheck = '%s, DIR' % dir
        if dircheck in self.get_folder_contents():
            self.change_folder(dir)
            logging.debug(__name__ + ' :Directory already exists')
            print('Directory already exists, changed path to %s' % dir)
            print('Contents of folder is %s' % self.ask('mmem:cat?'))
        elif self.get_current_folder_name() == '"\\%s"' % dir:
            print('Directory already set to %s' % dir)
        else:
            self.write('mmem:mdir "\%s"' % dir)
            self.write('mmem:cdir "\%s"' % dir)
            return self.get_folder_contents()

    def all_channels_on(self):
        for i in range(1, 5):
            self.set('ch{}_state'.format(i), 1)

    def all_channels_off(self):
        for i in range(1, 5):
            self.set('ch{}_state'.format(i), 0)

    def clear_waveforms(self):
        '''
        Clears the waveform on all channels.

        Input:
            None
        Output:
            None
        '''
        self.write('SOUR1:FUNC:USER ""')
        self.write('SOUR2:FUNC:USER ""')
        self.write('SOUR3:FUNC:USER ""')
        self.write('SOUR4:FUNC:USER ""')

    def get_sequence_length(self):
        return float(self.ask('SEQuence:LENGth?'))

    def get_refclock(self):
        '''
        Asks AWG whether the 10 MHz reference is set to the
        internal source or an external one.
        Input:
            None

        Output:
            'INT' or 'EXT'
        '''
        return self.ask('AWGC:CLOC:SOUR?')

    def set_refclock_ext(self):
        '''
        Sets the reference clock to internal or external.
        '''
        self.write('AWGC:CLOC:SOUR EXT')

    def set_refclock_int(self):
        '''
        Sets the reference clock to internal or external
        '''
        self.write('AWGC:CLOC:SOUR INT')

    ##############
    # Parameters #
    ##############

    def _do_get_numpoints(self):
        '''
        Returns the number of datapoints in each wave

        Input:
            None

        Output:
            numpoints (int) : Number of datapoints in each wave
        '''
        return self._numpoints

    def _do_set_numpoints(self, numpts):
        '''
        Sets the number of datapoints in each wave.
        This acts on both channels.

        Input:
            numpts (int) : The number of datapoints in each wave

        Output:
            None
        '''
        logging.debug(__name__ + ' : Trying to set numpoints to %s' % numpts)

        warn_string = ' : changing numpoints. This will clear all waveforms!'
        if numpts != self._numpoints:
            logging.warning(__name__ + warn_string)
        print(__name__ + warn_string)
        # Extra print cause logging.warning does not print
        response = input('type "yes" to continue')
        if response == 'yes':
            logging.debug(__name__ + ' : Setting numpoints to %s' % numpts)
            self._numpoints = numpts
            self.clear_waveforms()
        else:
            print('aborted')

    # Sequences section
    def force_trigger_event(self):
        self.write('TRIG:IMM')

    def force_event(self):
        self.write('EVEN:IMM')

    def set_sqel_event_target_index_next(self, element_no):
        self.write('SEQ:ELEM%s:JTARGET:TYPE NEXT' % element_no)

    def set_sqel_event_target_index(self, element_no, index):
        self.write('SEQ:ELEM%s:JTARGET:INDEX %s' % (element_no, index))

    def set_sqel_goto_target_index(self, element_no, goto_to_index_no):
        self.write('SEQ:ELEM%s:GOTO:IND  %s' % (element_no, goto_to_index_no))

    def set_sqel_goto_state(self, element_no, goto_state):
        self.write('SEQuence:ELEMent%s:GOTO:STATe %s' % (
            element_no, int(goto_state)))

    def set_sqel_loopcnt_to_inf(self, element_no, state=True):
        self.write('seq:elem%s:loop:inf %s' % (element_no, int(state)))

    def get_sqel_loopcnt(self, element_no=1):
        return self.ask('SEQ:ELEM%s:LOOP:COUN?' % element_no)

    def set_sqel_loopcnt(self, loopcount, element_no=1):
        self.write('SEQ:ELEM%s:LOOP:COUN %s' % (element_no, loopcount))

    def set_sqel_waveform(self, waveform_name, channel, element_no=1):
        self.write('SEQ:ELEM%s:WAV%s "%s"' % (
            element_no, channel, waveform_name))

    def get_sqel_waveform(self, channel, element_no=1):
        return self.ask('SEQ:ELEM%s:WAV%s?' % (element_no, channel))

    def set_sqel_trigger_wait(self, element_no, state=1):
        self.write('SEQ:ELEM%s:TWA %s' % (element_no, state))
        return self.get_sqel_trigger_wait(element_no)

    def get_sqel_trigger_wait(self, element_no):
        return self.ask('SEQ:ELEM%s:TWA?' % element_no)

    def get_sq_length(self):
        return self.ask('SEQ:LENG?')

    def set_sq_length(self, seq_length):
        self.write('SEQ:LENG %s' % seq_length)

    def set_sqel_event_jump_target_index(self, element_no, jtar_index_no):
        self.write('SEQ:ELEM%s:JTAR:INDex %s' % (element_no, jtar_index_no))

    def set_sqel_event_jump_type(self, element_no, jtar_state):
        self.write('SEQuence:ELEMent%s:JTAR:TYPE %s' %
                   (element_no, jtar_state))

    def get_sq_mode(self):
        return self.ask('AWGC:SEQ:TYPE?')

    def get_sq_position(self):
        return self.ask('AWGC:SEQ:POS?')

    def sq_forced_jump(self, jump_index_no):
        self.write('SEQ:JUMP:IMM %s' % jump_index_no)

    #################################
    # Transmon version file loading #
    #################################

    def load_and_set_sequence(self, wfname_l, nrep_l, wait_l, goto_l,
                              logic_jump_l):
        '''
        sets the AWG in sequence mode and loads waveforms into the sequence.
        wfname_l = list of waveform names [[wf1_ch1,wf2_ch1..],[wf1_ch2,wf2_ch2..],...],
                    waveforms are assumed to be already present and imported in AWG (see send_waveform and
                    import_waveform_file)
        nrep_l = list specifying the number of reps for each seq element
        wait_l = idem for wait_trigger_state
        goto_l = idem for goto_state (goto is the element where it hops to in case the element is finished)
        logic_jump_l = idem for event_jump_to (event or logic jump is the element where it hops in case of an event)

        '''
        self._load_new_style(wfname_l, nrep_l, wait_l, goto_l, logic_jump_l)

    def _load_new_style(self, wfname_l, nrep_l, wait_l, goto_l, logic_jump_l):
        '''
        load sequence not using sequence file
        '''
        self.set_sq_length(0)  # delete prev seq
        # print wfname_l
        len_sq = len(nrep_l)
        self.set_sq_length(len_sq)
        n_ch = len(wfname_l)
        for k in range(len_sq):
            # wfname_l[k]
            # print k
            for n in range(n_ch):
                # print n
                # print wfname_l[n][k]
                if wfname_l[n][k] is not None:

                    self.set_sqel_waveform(wfname_l[n][k], n + 1, k + 1)
            self.set_sqel_trigger_wait(k + 1, int(wait_l[k] != 0))
            self.set_sqel_loopcnt_to_inf(k + 1, False)
            self.set_sqel_loopcnt(nrep_l[k], k + 1)
            qt.msleep()
            if goto_l[k] == 0:
                self.set_sqel_goto_state(k + 1, False)
            else:
                self.set_sqel_goto_state(k + 1, True)
                self.set_sqel_goto_target_index(k + 1, goto_l[k])
            if logic_jump_l[k] == -1:
                self.set_sqel_event_target_index_next(k + 1)
            else:
                self.set_sqel_event_target_index(k + 1, logic_jump_l[k])

    def _load_old_style(self, wfs, rep, wait, goto, logic_jump, filename):
        '''
        Sends a sequence file (for the moment only for ch1
        Inputs (mandatory):

           wfs:  list of filenames

        Output:
            None
        This needs to be written so that awg files are loaded, will be much faster!
        '''
        pass

    ##################################################################

    def import_waveform_file(self, waveform_listname, waveform_filename,
                             type='wfm'):
        return self.write('mmem:imp "%s","%s",%s' % (
            waveform_listname, waveform_filename, type))

    def import_and_load_waveform_file_to_channel(self, channel_no,
                                                 waveform_listname,
                                                 waveform_filename,
                                                 type='wfm'):
        self._import_and_load_waveform_file_to_channel(channel_no,
                                                       waveform_listname,
                                                       waveform_filename,
                                                       type=type)

    def _import_and_load_waveform_file_to_channel(self, channel_no,
                                                  waveform_listname,
                                                  waveform_filename,
                                                  type='wfm'):
        self.write('mmem:imp "%s","%s",%s' % (
            waveform_listname, waveform_filename, type))
        self.write('sour%s:wav "%s"' % (channel_no, waveform_listname))
        i = 0
        while not (self.visa_handle.ask("sour%s:wav?" % channel_no)
                   == '"%s"' % waveform_listname):
            sleep(0.01)
            i = i + 1
        return
    ######################
    # AWG file functions #
    ######################

    def _pack_record(self, name, value, dtype):
        '''
        packs awg_file record into a struct in the folowing way:
            struct.pack(fmtstring, namesize, datasize, name, data)
        where fmtstring = '<IIs"dtype"'

        The file record format is as follows:
        Record Name Size:        (32-bit unsigned integer)
        Record Data Size:        (32-bit unsigned integer)
        Record Name:             (ASCII) (Include NULL.)
        Record Data
        For details see "File and Record Format" in the AWG help

           < denotes little-endian encoding, I and other dtypes are format
           characters denoted in the documentation of the struct package
        '''
        if len(dtype) == 1:
            record_data = struct.pack('<' + dtype, value)
        else:
            if dtype[-1] == 's':
                record_data = value.encode('ASCII')
            else:
                record_data = struct.pack('<' + dtype, *value)

        # the zero byte at the end the record name is the "(Include NULL.)"
        record_name = name.encode('ASCII') + b'\x00'
        record_name_size = len(record_name)
        record_data_size = len(record_data)
        size_struct = struct.pack('<II', record_name_size, record_data_size)
        packed_record = size_struct + record_name + record_data

        return packed_record

    def generate_sequence_cfg(self):
        '''
        This function is used to generate a config file, that is used when
        generating sequence files, from existing settings in the awg.
        Querying the AWG for these settings takes ~0.7 seconds
        '''
        logging.info('Generating sequence_cfg')

        AWG_sequence_cfg = {
            'SAMPLING_RATE': self.get('clock_freq'),
            'CLOCK_SOURCE': (1 if self.ask('AWGC:CLOCK:SOUR?').startswith('INT')
                             else 2),  # Internal | External
            'REFERENCE_SOURCE':   2,  # Internal | External
            'EXTERNAL_REFERENCE_TYPE':   1,  # Fixed | Variable
            'REFERENCE_CLOCK_FREQUENCY_SELECTION': 1,
            # 10 MHz | 20 MHz | 100 MHz
            'TRIGGER_SOURCE':   1 if
            self.get('trigger_source').startswith('EXT') else 2,
            # External | Internal
            'TRIGGER_INPUT_IMPEDANCE': (1 if self.get('trigger_impedance') ==
                                        50. else 2),  # 50 ohm | 1 kohm
            'TRIGGER_INPUT_SLOPE': (1 if self.get('trigger_slope').startswith(
                                    'POS') else 2),  # Positive | Negative
            'TRIGGER_INPUT_POLARITY': (1 if self.ask('TRIG:POL?').startswith(
                                       'POS') else 2),  # Positive | Negative
            'TRIGGER_INPUT_THRESHOLD':  self.get('trigger_level'),  # V
            'EVENT_INPUT_IMPEDANCE':   (1 if self.get('event_impedance') ==
                                        50. else 2),  # 50 ohm | 1 kohm
            'EVENT_INPUT_POLARITY':  (1 if self.get('event_polarity').startswith(
                                      'POS') else 2),  # Positive | Negative
            'EVENT_INPUT_THRESHOLD':   self.get('event_level'),  # V
            'JUMP_TIMING':   (1 if
                              self.get('event_jump_timing').startswith('SYNC')
                              else 2),  # Sync | Async
            'RUN_MODE':   4,  # Continuous | Triggered | Gated | Sequence
            'RUN_STATE':  0,  # On | Off
        }
        return AWG_sequence_cfg

    def generate_awg_file(self,
                          packed_waveforms, wfname_l, nrep, trig_wait,
                          goto_state, jump_to, channel_cfg, sequence_cfg=None):
        '''
        packed_waveforms: dictionary containing packed waveforms with keys
                            wfname_l and delay_labs
        wfname_l: array of waveform names array([[segm1_ch1,segm2_ch1..],
                                                [segm1_ch2,segm2_ch2..],...])
        nrep_l: list of len(segments) specifying the no of
                    reps per segment (0,65536)
        wait_l: list of len(segments) specifying triger wait state (0,1)
        goto_l: list of len(segments) specifying goto state (0, 65536),
                    0 means next)
        logic_jump_l: list of len(segments) specifying logic jump (0 = off)
        channel_cfg: dictionary of valid channel configuration records
        sequence_cfg: dictionary of valid head configuration records
                     (see AWG_FILE_FORMAT_HEAD)
                     When an awg file is uploaded these settings will be set
                     onto the AWG, any paramter not specified will be set to
                     its default value (even overwriting current settings)

        for info on filestructure and valid record names, see AWG Help,
        File and Record Format
        '''
        wfname_l
        timetuple = tuple(np.array(localtime())[[0, 1, 8, 2, 3, 4, 5, 6, 7]])

        # general settings
        head_str = BytesIO()
        bytes_to_write = (self._pack_record('MAGIC', 5000, 'h') +
                          self._pack_record('VERSION', 1, 'h'))
        head_str.write(bytes_to_write)
        # head_str.write(string(bytes_to_write))

        if sequence_cfg is None:
            sequence_cfg = self.generate_sequence_cfg()

        for k in list(sequence_cfg.keys()):
            if k in self.AWG_FILE_FORMAT_HEAD:
                head_str.write(self._pack_record(k, sequence_cfg[k],
                                                 self.AWG_FILE_FORMAT_HEAD[k]))
            else:
                logging.warning('AWG: ' + k +
                                ' not recognized as valid AWG setting')
        # channel settings
        ch_record_str = BytesIO()
        for k in list(channel_cfg.keys()):
            ch_k = k[:-1] + 'N'
            if ch_k in self.AWG_FILE_FORMAT_CHANNEL:
                ch_record_str.write(self._pack_record(k, channel_cfg[k],
                                                      self.AWG_FILE_FORMAT_CHANNEL[ch_k]))
            else:
                logging.warning('AWG: ' + k +
                                ' not recognized as valid AWG channel setting')
        # waveforms
        ii = 21
        wf_record_str = BytesIO()
        wlist = list(packed_waveforms.keys())
        wlist.sort()
        for wf in wlist:
            wfdat = packed_waveforms[wf]
            lenwfdat = len(wfdat)
            # print 'WAVEFORM_NAME_%s: '%ii, wf, 'len: ',len(wfdat)
            wf_record_str.write(
                self._pack_record('WAVEFORM_NAME_%s' % ii, wf + '\x00',
                                  '%ss' % len(wf + '\x00')) +
                self._pack_record('WAVEFORM_TYPE_%s' % ii, 1, 'h') +
                self._pack_record('WAVEFORM_LENGTH_%s' % ii, lenwfdat, 'l') +
                self._pack_record('WAVEFORM_TIMESTAMP_%s' % ii,
                                  timetuple[:-1], '8H') +
                self._pack_record('WAVEFORM_DATA_%s' % ii, wfdat, '%sH'
                                  % lenwfdat))
            ii += 1
        # sequence
        kk = 1
        seq_record_str = BytesIO()
        for segment in wfname_l.transpose():
            seq_record_str.write(
                self._pack_record('SEQUENCE_WAIT_%s' % kk, trig_wait[kk - 1],
                                  'h') +
                self._pack_record('SEQUENCE_LOOP_%s' % kk, int(nrep[kk - 1]),
                                  'l') +
                self._pack_record('SEQUENCE_JUMP_%s' % kk, jump_to[kk - 1],
                                  'h') +
                self._pack_record('SEQUENCE_GOTO_%s' % kk, goto_state[kk - 1],
                                  'h'))
            for wfname in segment:
                if wfname is not None:
                    ch = wfname[-1]
                    # print wfname,'SEQUENCE_WAVEFORM_NAME_CH_'+ch+'_%s'%kk
                    seq_record_str.write(
                        self._pack_record('SEQUENCE_WAVEFORM_NAME_CH_' + ch
                                          + '_%s' % kk, wfname + '\x00',
                                          '%ss' % len(wfname + '\x00')))
            kk += 1

        awg_file = head_str.getvalue() + ch_record_str.getvalue() + \
            wf_record_str.getvalue() + seq_record_str.getvalue()
        return awg_file

    def send_awg_file(self, filename, awg_file, verbose=False):
        if verbose:
            print('Writing to:',
                  self.ask('MMEMory:CDIRectory?').replace('\n', '\ '),
                  filename)
        # Header indicating the name and size of the file being send
        name_str = ('MMEM:DATA "%s",' % filename).encode('ASCII')
        size_str = ('#' + str(len(str(len(awg_file)))) +
                    str(len(awg_file))).encode('ASCII')
        mes = name_str + size_str + awg_file
        self.visa_handle.write_raw(mes)

    def load_awg_file(self, filename):
        s = 'AWGCONTROL:SRESTORE "%s"' % filename
        # print s
        self.visa_handle.write_raw(s)

    def get_error(self):
        # print self.visa_handle.ask('AWGControl:SNAMe?')
        print(self.ask('SYSTEM:ERROR:NEXT?'))
        # self.visa_handle.write('*CLS')

    def pack_waveform(self, wf, m1, m2):
        '''
        packs analog waveform in 14 bit integer, and two bits for m1 and m2
        in a single 16 bit integer
        '''
        wflen = len(wf)
        packed_wf = np.zeros(wflen, dtype=np.uint16)
        packed_wf += np.uint16(np.round(wf * 8191) + 8191 + np.round(16384 * m1) +
                               np.round(32768 * m2))
        if len(np.where(packed_wf == -1)[0]) > 0:
            print(np.where(packed_wf == -1))
        return packed_wf

    # END AWG file functions
    ###########################
    # Waveform file functions #
    ###########################

    def send_waveform(self, w, m1, m2, filename, clock=None):
        '''
        Sends a complete waveform. All parameters need to be specified.
        See also: resend_waveform()

        Input:
            w (float[numpoints]) : waveform
            m1 (int[numpoints])  : marker1
            m2 (int[numpoints])  : marker2
            filename (string)    : filename
            clock (int)          : frequency (Hz)

        Output:
            None
        '''
        # logging.debug(__name__ + ' : Sending waveform %s to instrument' %
        #               filename)
        # Check for errors

        if (not((len(w) == len(m1)) and ((len(m1) == len(m2))))):
            return 'error'

        self._values['files'][filename] = self._file_dict(w, m1, m2, clock)

        m = m1 + np.multiply(m2, 2)
        ws = b''
        # this is probalbly verry slow and memmory consuming!
        for i in range(0, len(w)):
            ws = ws + struct.pack('<fB', w[i], int(np.round(m[i], 0)))

        s1 = b'MMEM:DATA "%s",' % filename
        s3 = b'MAGIC 1000\n'
        s5 = ws
        if clock is not None:
            s6 = b'CLOCK %.10e\n' % clock
        else:
            s6 = b''

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        s4 = s4.encode('UTF-8')
        lenlen = str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))
        s2 = s2.encode('UTF-8')
        mes = s1 + s2 + s3 + s4 + s5 + s6

        self.visa_handle.write_raw(mes)

    def _file_dict(self, w, m1, m2, clock):
        return {
            'w': w,
            'm1': m1,
            'm2': m2,
            'clock_freq': clock,
            'numpoints': len(w)
        }

    def resend_waveform(self, channel, w=[], m1=[], m2=[], clock=[]):
        '''
        Resends the last sent waveform for the designated channel
        Overwrites only the parameters specified

        Input: (mandatory)
            channel (int) : 1 to 4, the number of the designated channel

        Input: (optional)
            w (float[numpoints]) : waveform
            m1 (int[numpoints])  : marker1
            m2 (int[numpoints])  : marker2
            clock (int) : frequency

        Output:
            None
        '''
        filename = self._values['recent_channel_%s' % channel]['filename']
        # logging.debug(__name__ + ' : Resending %s to channel %s' %
        #               (filename, channel))

        if (w == []):
            w = self._values['recent_channel_%s' % channel]['w']
        if (m1 == []):
            m1 = self._values['recent_channel_%s' % channel]['m1']
        if (m2 == []):
            m2 = self._values['recent_channel_%s' % channel]['m2']
        if (clock == []):
            clock = self._values['recent_channel_%s' % channel]['clock_freq']

        # if not ((len(w) == self._numpoints) and (len(m1) == self._numpoints)
        #         and (len(m2) == self._numpoints)):
        #     logging.error(__name__ + ' : one (or more) lengths of waveforms do not match with numpoints')

        self.send_waveform(w, m1, m2, filename, clock)
        self.set_filename(filename, channel)

    def set_filename(self, name, channel):
        '''
        Specifies which file has to be set on which channel
        Make sure the file exists, and the numpoints and clock of the file
        matches the instrument settings.

        If file doesn't exist an error is raised, if the numpoints doesn't match
        the command is neglected

        Input:
            name (string) : filename of uploaded file
            channel (int) : 1 to 4, the number of the designated channel

        Output:
            None
        '''
        # logging.debug(__name__  + ' : Try to set %s on channel %s' %(name, channel))
        exists = False
        if name in self._values['files']:
            exists = True
            # logging.debug(__name__  + ' : File exists in local memory')
            self._values['recent_channel_%s' % channel] = \
                self._values['files'][name]
            self._values['recent_channel_%s' % channel]['filename'] = name
        else:
            # logging.debug(__name__  + ' : File does not exist in memory, \
            # reading from instrument')
            lijst = self.visa_handle.ask('MMEM:CAT? "MAIN"')
            bool = False
            bestand = ""
            for i in range(len(lijst)):
                if (lijst[i] == '"'):
                    bool = True
                elif (lijst[i] == ','):
                    bool = False
                    if (bestand == name):
                        exists = True
                    bestand = ""
                elif bool:
                    bestand = bestand + lijst[i]
        if exists:
            data = self.visa_handle.ask('MMEM:DATA? "%s"' % name)

            # logging.debug(__name__  + ' : File exists on instrument, loading \
            #         into local memory')
            self._import_waveform_file(name, name)
            # string alsvolgt opgebouwd: '#' <lenlen1> <len> 'MAGIC 1000\r\n'
            # '#' <len waveform> 'CLOCK ' <clockvalue>
            len1 = int(data[1])
            len2 = int(data[2:2 + len1])
            i = len1
            tekst = ""
            while (tekst != '#'):
                tekst = data[i]
                i = i + 1
            len3 = int(data[i])
            len4 = int(data[i + 1:i + 1 + len3])
            w = []
            m1 = []
            m2 = []

            for q in range(i + 1 + len3, i + 1 + len3 + len4, 5):
                j = int(q)
                c, d = struct.unpack('<fB', data[j:5 + j])
                w.append(c)
                m2.append(int(d / 2))
                m1.append(d - 2 * int(d / 2))

            clock = float(data[i + 1 + len3 + len4 + 5:len(data)])

            self._values['files'][name] = self._file_dict(w, m1, m2, clock)

            self._values['recent_channel_%s' % channel] = \
                self._values['files'][name]
            self._values['recent_channel_%s' % channel]['filename'] = name
        # else:
            # logging.error(__name__  + ' : Invalid filename specified %s' %name)

        if (self._numpoints == self._values['files'][name]['numpoints']):
            # logging.warning(__name__  + ' : Set file %s on channel %s' % (name, channel))
            self.write('SOUR%s:WAV "%s"' % (channel, name))
        else:
            pass
            # logging.warning(__name__  + ' : Verkeerde lengte %s ipv %s'
            #     %(self._values['files'][name]['numpoints'], self._numpoints))

    def delete_all_waveforms_from_list(self):
        self.write('WLISt:WAVeform:DELete ALL')

    # Ask for string with filenames
    def get_filenames(self):
        # logging.debug(__name__ + ' : Read filenames from instrument')
        return self.ask('MMEM:CAT?')

    def set_DC_out(self, DC_channel_number, Voltage):
        self.write('AWGControl:DC%s:VOLTage:OFFSet %sV' %
                   (DC_channel_number, Voltage))

    def get_DC_out(self, DC_channel_number):
        return self.ask('AWGControl:DC%s:VOLTage:OFFSet?' % DC_channel_number)

    def send_DC_pulse(self, DC_channel_number, Amplitude, length):
        '''
        sends a (slow) pulse on the DC channel specified
        Ampliude: voltage level
        length: seconds
        '''
        restore = self.get_DC_out(DC_channel_number)
        self.set_DC_out(DC_channel_number, Amplitude)
        sleep(length)
        self.set_DC_out(DC_channel_number, restore)

    def set_DC_state(self, state=False):
        self.write('AWGControl:DC:state %s' % (int(state)))

    def get_DC_state(self):
        return self.ask('AWGControl:DC:state?')

    # Send waveform to the device (from transmon driver)

    def upload_awg_file(self, fname, fcontents):
        t0 = time()
        self._rem_file_path
        floc = self._rem_file_path
        f = open(floc + '\\' + fname, 'wb')
        f.write(fcontents)
        f.close()
        t1 = time() - t0
        print('upload time: ', t1)
        self.get_state()
        print('setting time: ', time() - t1 - t0)

    def _set_setup_filename(self, fname):
        folder_name = 'C:/' + self._setup_folder + '/' + fname
        self.set_current_folder_name(folder_name)
        set_folder_name = self.get_current_folder_name()
        if not os.path.split(folder_name)[1] == os.path.split(set_folder_name)[1][:-1]:
            print('Warning, unsuccesfully set AWG file', folder_name)
        print('Current AWG file set to: ', self.get_current_folder_name())
        self.write('AWGC:SRES "%s.awg"' % fname)

    def set_setup_filename(self, fname, force_load=False):
        '''
        sets the .awg file to a .awg file that already exists in the memory of
        the AWG.

        fname (string) : file to be set.
        force_load (bool): if True, sets the file even if it is already loaded

        After setting the file it resets all the instrument settings to what
        it is set in the qtlab memory.
        '''
        cfname = self.get_setup_filename()
        if cfname.find(fname) != -1 and not force_load:
            print('file: %s already loaded' % fname)
        else:
            pars = self.get_parameters()
            self._set_setup_filename(fname)
            # self.visa_handle.ask('*OPC?')
            parkeys = list(pars.keys())
            parkeys.remove('setup_filename')
            parkeys.remove('AWG_model')
            parkeys.remove('numpoints')
            # not reset because this removes all loaded waveforms
            parkeys.remove('trigger_mode')
            # Removed trigger_mode because duplicate of run mode
            comm = False
            for key in parkeys:
                try:
                    # print 'setting: %s' % key
                    exec('self.set_%s(pars[key]["value"])' % key)
                    comm = True
                except Exception as e:
                    print(key + ' not set!')
                    comm = False

                # Sped up by factor 10, VISA protocol should take care of wait
                if comm:
                    self.ask('*OPC?')
        self.get('setup_filename')  # ensures the setup filename gets updated

    def is_awg_ready(self):
        try:
            self.ask('*OPC?')
        # makes the awg read again if there is a timeout
        except Exception as e:
            logging.warning(e)
            logging.warning('AWG is not ready')
            self.visa_handle.read()
        return True

    def initialize_dc_waveforms(self):
        self.set_runmode('CONT')
        self.write('SOUR1:WAV "*DC"')
        self.write('SOUR2:WAV "*DC"')
        self.write('SOUR3:WAV "*DC"')
        self.write('SOUR4:WAV "*DC"')
        self.set_ch1_status('on')
        self.set_ch2_status('on')
        self.set_ch3_status('on')
        self.set_ch4_status('on')

    # QCodes specific parse_functions
    def parse_int_pos_neg(self, val):
        return ['POS', 'NEG'][val]

    def parse_int_int_ext(self, val):
        return ['INT', 'EXT'][val]

    def send_waveform_to_list(self, w, m1, m2, wfmname):
        '''
        Sends a complete waveform directly to the "User defined" waveform list. All parameters need to be specified.
        See also: resend_waveform()

        Input:
            w (float[numpoints]) : waveform (must be a numpy array)
            m1 (int[numpoints])  : marker1  (must be a numpy array)
            m2 (int[numpoints])  : marker2  (must be a numpy array)
            wfmname (string)    : waveform name
            format (string):    'int' or 'real' (int has same awg output precision but much faster to transfer)
        Output:
            None
        '''
        logging.debug(
            __name__ + ' : Sending waveform %s to instrument' % wfmname)
        # Check for errors
        dim = len(w)

        if (not((len(w) == len(m1)) and ((len(m1) == len(m2))))):
            raise Exception('error: sizes of the waveforms do not match')

        self._values['files'][wfmname] = self._file_dict(w, m1, m2, None)

        # if we create a waveform with the same name but different size, it will not get over written
        # Delete the possibly existing file (will do nothing if the file
        # doesn't exist
        s = 'WLIS:WAV:DEL "%s"' % wfmname
        self.write(s)

        print("Sending the waveform %s" % wfmname)

        # create the waveform
        s = 'WLIS:WAV:NEW "%s",%i,INTEGER' % (wfmname, dim)
        self.write(s)
        # Prepare the data block

        number = (2**13 + 2**13 * w + 2**14 *
                  np.array(m1) + 2**15 * np.array(m2))
        number = number.astype('int')
        ws = arr.array('H', number)

        ws = ws.tostring()
        s1 = 'WLIS:WAV:DATA "%s",' % wfmname
        s1 = s1.encode('UTF-8')
        s3 = ws
        s2 = '#' + str(len(str(len(s3)))) + str(len(s3))
        s2 = s2.encode('UTF-8')

        mes = s1 + s2 + s3
        self.visa_handle.write_raw(mes)
