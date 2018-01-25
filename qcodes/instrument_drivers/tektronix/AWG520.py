# Tektronix_AWG520.py class, to perform the communication between the Wrapper and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
# Vishal Ranjan, 2012
# Ron schutjens, 2012
# Adriaan Rol, 2016 Ported to QCodes
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

from typing import List, Union
import time
import logging
import numpy as np
import struct
from qcodes import VisaInstrument, validators as vals


logger = logging.getLogger(__name__)

class Tektronix_AWG520(VisaInstrument):
    '''
    This is the python driver for the Tektronix AWG520
    Arbitrary Waveform Generator

    .. todo::

        1) Get All
        2) Remove test_send??
        3) Add docstrings

    .. todo::

        use inheritance for common use with 520, currently contains
        a lot of repetition
    '''

    def __init__(self, name, address, reset=False, clock=1e9, numpoints=1000,
                 **kw):
        '''
        Initializes the AWG520.

        Args:
            name (string)    : name of the instrument
            address (string) : GPIB address (Note: 520 cannot be controlled
                               via ethernet)
            reset (bool)     : resets to default values, default=false
            numpoints (int)  : sets the number of datapoints

        Output:
            None
        '''
        super().__init__(name, address, **kw)
        self.set_terminator('\n')

        self._address = address
        self._values = {}
        self._values['files'] = {}
        self._clock = clock
        self._numpoints = numpoints
        self._fname = ''

        self.add_function('reset', call_cmd='*RST')
        self.add_parameter('state',
                           get_cmd=self.get_state)

        # Add parameters
        self.add_parameter('trigger_mode',
                           get_cmd='AWGC:RMOD?',
                           set_cmd='AWGC:RMOD ' + '{}',
                           vals=vals.Enum('CONT', 'TRIG', 'ENH', 'GAT'))
        self.add_parameter('trigger_impedance',
                           unit='Ohm',
                           label='Trigger impedance (Ohm)',
                           get_cmd='TRIG:IMP?',
                           set_cmd='TRIG:IMP '+'{}',
                           vals=vals.Enum(50, 1000),
                           get_parser=float)
        self.add_parameter('trigger_level',
                           unit='V',
                           label='Trigger level (V)',
                           get_cmd='TRIG:LEV?',
                           set_cmd='TRIG:LEV '+'{:.3f}',
                           vals=vals.Numbers(-5, 5),
                           get_parser=float)

        self.add_parameter('clock_freq',
                           label='Clock frequency (Hz)',
                           get_cmd='SOUR:FREQ?',
                           set_cmd='SOUR:FREQ '+'{}',
                           vals=vals.Numbers(1e6, 1e9),
                           get_parser=float)
        # Todo check if max freq is 1.2 GHz for the AWG 520 aswell
        self.add_parameter('numpoints',
                           label='Number of datapoints per wave',
                           get_cmd=self._do_get_numpoints,
                           set_cmd=self._do_set_numpoints,
                           vals=vals.Ints(100, int(1e9)))

        for ch in [1, 2]:
            amp_cmd = 'SOUR{}:VOLT:LEV:IMM:AMPL'.format(ch)
            offset_cmd = 'SOUR{}:VOLT:LEV:IMM:OFFS'.format(ch)

            self.add_parameter(
                'ch{}_filename'.format(ch), set_cmd=self._gen_ch_set_func(
                    self._do_set_filename, ch), vals=vals.Anything())
            self.add_parameter('ch{}_amp'.format(ch),
                               label='Amplitude channel {} (V)'.format(ch),
                               unit='V',
                               get_cmd=amp_cmd + '?',
                               set_cmd=amp_cmd + ' {:.6f}',
                               vals=vals.Numbers(0.02, 2.0),
                               get_parser=float)

            self.add_parameter('ch{}_offset'.format(ch),
                               label='Offset channel {} (V)'.format(ch),
                               unit='V',
                               get_cmd=offset_cmd + '?',
                               set_cmd=offset_cmd + ' {:.3f}',
                               vals=vals.Numbers(-1.0, 1.0),
                               get_parser=float)
            self.add_parameter('ch{}_status'.format(ch),
                               get_cmd='OUTP{}?'.format(ch),
                               set_cmd='OUTP{}'.format(ch) + ' {}',
                               vals=vals.Enum('ON', 'OFF'),
                               get_parser=float)

            for j in [1, 2]:
                # TODO: check that 520 does not have marker delay feature
                # m_del_cmd = 'SOUR{}:MARK{}:DEL'.format(ch, j)
                m_high_cmd = 'SOUR{}:MARK{}:VOLT:LEV:IMM:HIGH'.format(ch, j)
                m_low_cmd = 'SOUR{}:MARK{}:VOLT:LEV:IMM:LOW'.format(ch, j)

                self.add_parameter(
                    'ch{}_m{}_high'.format(ch, j),
                    label='Channel {} Marker {} high level (V)'.format(ch, j),
                    get_cmd=m_high_cmd + '?',
                    set_cmd=m_high_cmd + ' {:.3f}',
                    vals=vals.Numbers(-2., 2.),
                    get_parser=float)
                self.add_parameter(
                    'ch{}_m{}_low'.format(ch, j),
                    label='Channel {} Marker {} low level (V)'.format(ch, j),
                    get_cmd=m_low_cmd + '?',
                    set_cmd=m_low_cmd + ' {:.3f}',
                    vals=vals.Numbers(-2., 2.),
                    get_parser=float)

        # Add functions
        if reset:
            self.reset()
        else:
            self.snapshot(update=False)

        self.connect_message()

    # Functions
    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func

    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func

    # get state AWG
    def get_state(self):
        state = self.ask('AWGC:RSTATE?')
        if state.startswith('0'):
            return 'Idle'
        elif state.startswith('1'):
            return 'Waiting for trigger'
        elif state.startswith('2'):
            return 'Running'
        else:
            logger.error('AWG in undefined state')
            return 'error'

    def get_errors(self, max_requests=50):
        errors = []
        for k in range(max_requests):
            error_msg = self.ask('SYSTem:ERRor?')
            if 'No error' in error_msg:
                break
            errors.append(error_msg)
        return errors

    def start(self):
        self.write('AWGC:RUN')
        return

    def stop(self):
        self.write('AWGC:STOP')

    def get_folder_contents(self, return_filesizes=False):
        # Retrieve folder contents as one large string
        contents = self.ask('mmem:cat?')
        # Remove first part of string containing information about directory
        contents = contents.split(',"', 1)[1]

        if contents == ',,"':
            # Empty folder
            return [], []

        # Separate the elements
        contents = contents.split('","')
        # Each element has form
        contents = [elem.split(',') for elem in contents]

        folders = [elem[0] for elem in contents if elem[1] == 'DIR']
        if return_filesizes:
            files = [elem[0::2] for elem in contents if not elem[1]]
        else:
            files = [elem[0] for elem in contents if not elem[1]]

        return files, folders

    def get_current_folder_name(self):
        return self.ask('mmem:cdir?').strip('"')

    def delete_file(self, file):
        return self.write(f'mmem:del "{file}"')

    def set_current_folder_name(self, file_path):
        self.write('mmem:cdir "%s"' % file_path)

    def change_folder(self, dir):
        self.write('mmem:cdir "%s"' % dir)

    def goto_root(self):
        self.write('mmem:cdir')

    def make_directory(self, dir, root):
        """ makes a directory
        if root = True, new dir in main folder
        """
        if root:
            self.goto_root()
            self.write('MMEMory:MDIRectory "{}"'.format(dir))
        else:
            self.write('MMEMory:MDIRectory "{}"'.format(dir))

    def delete_file(self, filename):
        self.write('MMemory:delete "{}"'.format(filename))

    def delete_all_files(self, root=False):
        if root:
            self.goto_root()
        print('Deleting all files')
        files, _ = self.get_folder_contents()
        for elem in files:
            self.delete_file(elem)

    def clear_waveforms(self):
        '''
        Clears the waveform on both channels.

        Input:
            None

        Output:
            None
        '''
        logger.debug('Clear waveforms from channels')
        self.write('SOUR2:FUNC:USER ""')

    def force_trigger(self):
        '''
        forces a trigger event (used for wait_trigger option in sequences)

        Ron
        '''
        return self.write('TRIG:SEQ:IMM')

    def force_logicjump(self):
        '''
        forces a jumplogic event (used as a conditional event during waveform
        executions)

        note: jump_logic events&mode have to be set properly!

        Ron
        '''
        return self.write('AWGC:EVEN:SEQ:IMM')

    def set_jumpmode(self, mode):
        '''
        sets the jump mode for jump logic events, possibilities:
        LOGic,TABle,SOFTware
        give mode as string

        note: jump_logic events&mode have to be set properly!

        Ron
        '''
        return self.write('AWGC:ENH:SEQ:JMOD %s' % mode)

    def get_jumpmode(self, mode):
        '''
        get the jump mode for jump logic events

        Ron
        '''
        return self.ask('AWGC:ENH:SEQ:JMOD?')

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
        logger.debug('Trying to set numpoints to %s' % numpts)
        if numpts != self._numpoints:
            logger.warning('changing numpoints. This will clear all waveforms!')

        response = 'yes'  # raw_input('type "yes" to continue')
        if response is 'yes':
            logger.debug('Setting numpoints to %s' % numpts)
            self._numpoints = numpts
            self.clear_waveforms()
        else:
            print('aborted')

    def set_setup_filename(self, fname, force_reload=False):
        if self._fname == fname and not force_reload:
            print('File %s already loaded in AWG520' % fname)
            return
        else:
            self._fname = fname
            filename = "\%s/%s.seq" % (fname, fname)
            self.set_sequence(filename=filename)
            print('Waiting for AWG to load file "%s"' % fname)
            sleeptime = 0.5
            # while state idle is not possible due to timeout error while loading
            t0 = time.time()
            while(time.time()-t0 < 360):
                try:
                    if self.get_state() == 'Idle':
                        break
                except:
                    time.sleep(sleeptime)
                    print('.')
            self.get_state()
            print('Loading file took %.2fs' % (time.time()-t0))
            return

    def _do_set_filename(self, name, channel):
        '''
        Specifies which file has to be set on which channel
        Make sure the file exists, and the numpoints and clock of the file
        matches the instrument settings.

        If file doesn't exist an error is raised, if the numpoints doesn't match
        the command is neglected

        Input:
            name (string) : filename of uploaded file
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
        '''
        logger.debug('Try to set {} on channel {}'.format(
                      name, channel))
        exists = False
        if name in self._values['files']:
            exists = True
            logger.debug('File exists in loacal memory')
            self._values['recent_channel_%s' % channel] = self._values[
                'files'][name]
            self._values['recent_channel_%s' % channel]['filename'] = name
        else:
            logger.debug('File does not exist in memory, \
            reading from instrument')
            lijst = self.ask('MMEM:CAT? "MAIN"')
            bool = False
            bestand = ""
            for i in range(len(lijst)):
                if (lijst[i] =='"'):
                    bool = True
                elif (lijst[i] == ','):
                    bool = False
                    if (bestand == name):
                        exists = True
                    bestand = ""
                elif bool:
                    bestand = bestand + lijst[i]
        if exists:
            data = self.ask('MMEM:DATA? "%s"' %name)
            logger.debug('File exists on instrument, loading \
            into local memory')
            # string alsvolgt opgebouwd: '#' <lenlen1> <len> 'MAGIC 1000\r\n' '#' <len waveform> 'CLOCK ' <clockvalue>
            len1 = int(data[1])
            len2 = int(data[2:2+len1])
            i = len1
            tekst = ""
            while (tekst !='#'):
                tekst = data[i]
                i = i+1
            len3 = int(data[i])
            len4 = int(data[i+1:i+1+len3])

            w = []
            m1 = []
            m2 = []

            for q in range(i+1+len3, i+1+len3+len4, 5):
                j = int(q)
                c, d = struct.unpack('<fB', data[j:5+j])
                w.append(c)
                m2.append(int(d/2))
                m1.append(d-2*int(d/2))

            clock = float(data[i+1+len3+len4+5:len(data)])

            self._values['files'][name] = {}
            self._values['files'][name]['w'] = w
            self._values['files'][name]['m1'] = m1
            self._values['files'][name]['m2'] = m2
            self._values['files'][name]['clock'] = clock
            self._values['files'][name]['numpoints'] = len(w)

            self._values['recent_channel_%s' %channel] = self._values['files'][name]
            self._values['recent_channel_%s' %channel]['filename'] = name
        else:
            logger.error('Invalid filename specified %s' %name)

        if (self._numpoints==self._values['files'][name]['numpoints']):
            logger.debug('Set file %s on channel %s' % (name, channel))
            self.write('SOUR%s:FUNC:USER "%s","MAIN"' % (channel, name))
            logger.warning('Verkeerde lengte %s ipv %s'
                %(self._values['files'][name]['numpoints'], self._numpoints))

    def send_waveform(self,
                      waveform: List[float],
                      marker1: List[bool],
                      marker2: List[bool],
                      filename: str,
                      clock: float):
        '''
        Sends a complete waveform. All parameters need to be specified.
        In contrast to a pattern, a waveform allows mathematical processing on
        the waveform point, at the cost of slightly slower transferring.

        Args:
            waveform : Waveform array to send
            marker1: Values for marker 1 output (1 is high, 0 is low)
            marker2: Values for marker 2 output (1 is high, 0 is low)
            filename: filename ending with ``.wfm``. If not provided,
                automatically appended.
            clock: frequency (Hz)

        See also:
            `resend_waveform`
            `send_pattern`
        '''
        logger.debug('Sending waveform %s to instrument' % filename)

        if '.' in filename:
            if not filename.split('.')[1] == 'wfm':
                raise SyntaxError('File must end with .wfm')
        else:
            filename += '.wfm'

        if not len(waveform) == len(marker1) == len(marker2):
            raise SyntaxError('Lengths of waveforms and markers arent equal')
        elif len(waveform) % 4:
            raise SyntaxError('Waveform must have even number of points')
        elif len(waveform) < 256:
            raise SyntaxError('Waveform is too short')

        self._values['files'][filename] = {}
        self._values['files'][filename]['w'] = waveform
        self._values['files'][filename]['m1'] = marker1
        self._values['files'][filename]['m2'] = marker2
        self._values['files'][filename]['clock'] = clock
        self._values['files'][filename]['numpoints'] = len(waveform)

        m = marker1 + np.multiply(marker2, 2)

        ws = b''.join(struct.pack('<fB', waveform_elem, int(marker_elem))
                      for waveform_elem, marker_elem in zip(waveform, m))

        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 1000\n'
        s5 = ws
        s6 = 'CLOCK %.10e\n' % clock

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen = str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = str.encode(s1+s2+s3+s4) + s5 + str.encode(s6)
        self.visa_handle.write_raw(mes)

    def send_pattern(self,
                     waveform: List[float],
                     marker1: List[bool],
                     marker2: List[bool],
                     filename: str,
                     clock: float):
        '''
        Sends a pattern file. In contrast to a waveform, a pattern does not
        allow mathematical processing on the waveform points, resulting in a
        slight gain in transfer time. Also has a different byte conversion.

        Args:
            waveform : Waveform array to send
            marker1: Values for marker 1 output (1 is high, 0 is low)
            marker2: Values for marker 2 output (1 is high, 0 is low)
            filename: filename ending with ``.pat``. If not provided,
                automatically appended.
            clock: frequency (Hz)

        See also:
            `resend_waveform`
            `send_waveform`
        '''
        logger.debug('Sending pattern %s to instrument' % filename)

        if '.' in filename:
            if not filename.split('.')[1] == 'pat':
                raise SyntaxError('File must end with .pat')
        else:
            filename += '.pat'

        # Check for errors
        dim = len(waveform)
        if (not((len(waveform) == len(marker1)) and ((len(marker1) == len(marker2))))):
            return 'error'
        self._values['files'][filename]={}
        self._values['files'][filename]['w']=waveform
        self._values['files'][filename]['m1']=marker1
        self._values['files'][filename]['m2']=marker2
        self._values['files'][filename]['clock']=clock
        self._values['files'][filename]['numpoints']=len(waveform)

        m = marker1 + np.multiply(marker2, 2)

        ws = b''.join(struct.pack('<fB', waveform_elem, int(marker_elem))
                      for waveform_elem, marker_elem in zip(waveform, m))

        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 2000\n'
        s5 = ws
        s6 = 'CLOCK %.10e\n' % clock

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen = str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = str.encode(s1+s2+s3+s4) + s5 + str.encode(s6)
        self.visa_handle.write_raw(mes)

    def resend_waveform(self, channel, w=[], m1=[], m2=[], clock=1e9):
        '''
        Resends the last sent waveform for the designated channel
        Overwrites only the parameters specifiedta

        Input: (mandatory)
            channel (int) : 1 or 2, the number of the designated channel

        Input: (optional)
            w (float[numpoints]) : waveform
            m1 (int[numpoints])  : marker1
            m2 (int[numpoints])  : marker2
            clock (int) : frequency

        Output:
            None
        '''
        filename = self._values['recent_channel_%s' %channel]['filename']
        logger.debug('Resending %s to channel %s' % (filename, channel))

        if (w==[]):
            w = self._values['recent_channel_%s' %channel]['w']
        if (m1==[]):
            m1 = self._values['recent_channel_%s' %channel]['m1']
        if (m2==[]):
            m2 = self._values['recent_channel_%s' %channel]['m2']
        if (clock==[]):
            clock = self._values['recent_channel_%s' %channel]['clock']

        if not len(w) == len(m1) == len(m2) == self._numpoints:
            logger.error('one (or more) lengths of waveforms do not match with numpoints')

        self.send_waveform(w, m1, m2, filename, clock)
        self.do_set_filename(filename, channel)

    def send_equation(self, filename, equation, pre_code, points, clock=1e9, ):

        start_msg = f'MMEM:DATA "{filename}",'

        equation_msg = '\n'.join([f'clock={clock}',
                                  f'size={points}',
                                  f'{pre_code}',
                                  f'"{filename}"={equation}'])

        # Information about number of characters
        msg_char_length = str(len(equation_msg))
        length_chars = str(len(msg_char_length))
        length_str = f'#{length_chars}{msg_char_length}'

        self.visa_handle.write_raw(start_msg + length_str + equation_msg)

    def send_sequence(self,
                      filename: str,
                      waveforms: Union[List[str], List[List[str]]],
                      repetitions: List[int] = None,
                      wait_trigger: List[bool] = None,
                      goto_one: List[bool] = None,
                      logic_jump: List[int] = None):
        '''
        Sends a sequence file (for the moment only for ch1)

        Args:
            waveforms:  Waveform filenames to use. Can be either a list of
                filenames, in which case they are output at channel 1, or
                a list with two lists, each containing the filenames for the
                respective channel.
            repetitions: Repetitions for each waveform. 0 is infinite.
            wait_trigger: Whether to wait for a trigger before continuing.
                Default is False for each waveform.
            goto_one: Whether to go back to the first line. If an event occurs,
                logic_jump is used instead. Default is False for each waveform.
            logic_jump: line number for the Logic-Jump.
                0 is Off, –1 is Next, and –2 is Table-Jump. The default is Off.
                Logic jump occurs after an event. Default is zero for each
                waveform.

        TODO:
            Check if wait trigger is for starting current waveform or
            continuing to next
            What does goto do?

        '''
        logger.debug('Sending sequence %s to instrument' % filename)

        if '.' in filename:
            if not filename.split('.')[1] == 'seq':
                raise SyntaxError('File must end with .seq')
        else:
            filename += '.seq'

        if isinstance(waveforms[0], str):
            N_instructions = len(waveforms)
        else:
            N_instructions = len(waveforms[0])

        if repetitions is None:
            repetitions = np.ones(N_instructions)
        if wait_trigger is None:
            wait_trigger = np.zeros(N_instructions)
        if goto_one is None:
            goto_one = np.zeros(N_instructions)
        if logic_jump is None:
            logic_jump = np.zeros(N_instructions)

        # Convert all args to integer arrays
        repetitions = np.array(repetitions, dtype=int)
        wait_trigger = np.array(wait_trigger, dtype=int)
        goto_one = np.array(goto_one, dtype=int)
        logic_jump = np.array(logic_jump, dtype=int)

        assert max(repetitions) <= 65536, "Repetitions must be max 65536"

        s1 = 'MMEM:DATA "%s",' % filename

        if len(np.shape(waveforms)) == 1:
            s3 = 'MAGIC 3001\n'
            instructions = [
                f'"{waveforms[k]}",{repetitions[k]},{wait_trigger[k]},{goto_one[k]},{logic_jump[k]}'
                for k in range(N_instructions)]
            s5 = '\n'.join(instructions)

        else:
            s3 = 'MAGIC 3002\n'
            instructions = [
                f'"{waveforms[0][k]}","{waveforms[1][k]}",{repetitions[k]},{wait_trigger[k]},{goto_one[k]},{logic_jump[k]}'
                for k in range(N_instructions)]
            s5 = '\n'.join(instructions)

        s4 = 'LINES %s\n' % N_instructions
        lenlen=str(len(str(len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s5) + len(s4) + len(s3))

        mes = s1 + s2 + s3 + s4 + s5
        self.write(mes)

    def set_sequence(self,filename, ch=1):
        '''
        loads a sequence file on all channels.
        Waveforms/patterns to be executed on respective channel
        must be defined inside the sequence file itself
        make sure to send all waveforms before setting a seq
        '''
        self.write('SOUR%s:FUNC:USER "%s","MAIN"' % (ch, filename))

    def load_and_set_sequence(self,wfs,rep,wait,goto,logic_jump,filename):
        '''
        Loads and sets the awg sequecne
        '''
        self.send_sequence(wfs,rep,wait,goto,logic_jump,filename)
        self.set_sequence(filename)

    # Unnecessary methods
    def delete_all_waveforms_from_list(self):
        '''
        for compatibillity with awg, is not relevant for AWG520 since it
        has no waveform list
        '''
        pass