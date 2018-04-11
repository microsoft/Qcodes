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


import time
import logging
import numpy as np
import struct
from qcodes import VisaInstrument, validators as vals


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
            self.get_all()
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
        state = self.visa_handle.ask('AWGC:RSTATE?')
        if state.startswith('0'):
            return 'Idle'
        elif state.startswith('1'):
            return 'Waiting for trigger'
        elif state.startswith('2'):
            return 'Running'
        else:
            logging.error(__name__ + ' : AWG in undefined state')
            return 'error'

    def start(self):
        self.visa_handle.write('AWGC:RUN')
        return

    def stop(self):
        self.visa_handle.write('AWGC:STOP')

    def get_folder_contents(self):
        return self.visa_handle.ask('mmem:cat?')

    def get_current_folder_name(self):
        return self.visa_handle.ask('mmem:cdir?')

    def set_current_folder_name(self, file_path):
        self.visa_handle.write('mmem:cdir "%s"' % file_path)

    def change_folder(self, dir):
        self.visa_handle.write('mmem:cdir "%s"' % dir)

    def goto_root(self):
        self.visa_handle.write('mmem:cdir')

    def make_directory(self, dir, root):
        '''
        makes a directory
        if root = True, new dir in main folder
        '''
        if root:
            self.goto_root()
            self.visa_handle.write('MMEMory:MDIRectory "{}"'.format(dir))
        else:
            self.visa_handle.write('MMEMory:MDIRectory "{}"'.format(dir))

    def get_all(self, update=True):
        # TODO: fix bug in snapshot where it tries to get setable only param
        # return self.snapshot(update=update)

        return self.snapshot(update=False)

    def clear_waveforms(self):
        '''
        Clears the waveform on both channels.

        Input:
            None

        Output:
            None
        '''
        logging.debug(__name__ + ' : Clear waveforms from channels')
        self.visa_handle.write('SOUR1:FUNC:USER ""')
        self.visa_handle.write('SOUR2:FUNC:USER ""')

    def force_trigger(self):
        '''
        forces a trigger event (used for wait_trigger option in sequences)

        Ron
        '''
        return self.visa_handle.write('TRIG:SEQ:IMM')

    def force_logicjump(self):
        '''
        forces a jumplogic event (used as a conditional event during waveform
        executions)

        note: jump_logic events&mode have to be set properly!

        Ron
        '''
        return self.visa_handle.write('AWGC:EVEN:SEQ:IMM')

    def set_jumpmode(self, mode):
        '''
        sets the jump mode for jump logic events, possibilities:
        LOGic,TABle,SOFTware
        give mode as string

        note: jump_logic events&mode have to be set properly!

        Ron
        '''
        return self.visa_handle.write('AWGC:ENH:SEQ:JMOD %s' % mode)

    def get_jumpmode(self, mode):
        '''
        get the jump mode for jump logic events

        Ron
        '''
        return self.visa_handle.ask('AWGC:ENH:SEQ:JMOD?')

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
        if numpts != self._numpoints:
            logging.warning(__name__ + ' : changing numpoints. This will clear all waveforms!')

        response = 'yes'  # raw_input('type "yes" to continue')
        if response == 'yes':
            logging.debug(__name__ + ' : Setting numpoints to %s' % numpts)
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
        logging.debug(__name__ + ' : Try to set {} on channel {}'.format(
                      name, channel))
        exists = False
        if name in self._values['files']:
            exists = True
            logging.debug(__name__ + ' : File exists in loacal memory')
            self._values['recent_channel_%s' % channel] = self._values[
                'files'][name]
            self._values['recent_channel_%s' % channel]['filename'] = name
        else:
            logging.debug(__name__ + ' : File does not exist in memory, \
            reading from instrument')
            lijst = self.visa_handle.ask('MMEM:CAT? "MAIN"')
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
            data = self.visa_handle.ask('MMEM:DATA? "%s"' %name)
            logging.debug(__name__  + ' : File exists on instrument, loading \
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
            logging.error(__name__ + ' : Invalid filename specified %s' %name)

        if (self._numpoints==self._values['files'][name]['numpoints']):
            logging.debug(__name__  + ' : Set file %s on channel %s' % (name, channel))
            self.visa_handle.write('SOUR%s:FUNC:USER "%s","MAIN"' % (channel, name))
        else:
            self.visa_handle.write('SOUR%s:FUNC:USER "%s","MAIN"' % (channel, name))
            logging.warning(__name__  + ' : Verkeerde lengte %s ipv %s'
                %(self._values['files'][name]['numpoints'], self._numpoints))


    #  Ask for string with filenames
    def get_filenames(self):
        logging.debug(__name__ + ' : Read filenames from instrument')
        return self.visa_handle.ask('MMEM:CAT? "MAIN"')

    def return_self(self):
        return self
    # Send waveform to the device

    def send_waveform(self, w, m1, m2, filename, clock):
        '''
        Sends a complete waveform. All parameters need to be specified.
        choose a file extension 'wfm' (must end with .pat)
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
        logging.debug(__name__ + ' : Sending waveform %s to instrument' % filename)

        # Check for errors
        dim = len(w)

        if (not((len(w) == len(m1)) and ((len(m1) == len(m2))))):
            return 'error'
        self._values['files'][filename] = {}
        self._values['files'][filename]['w'] = w
        self._values['files'][filename]['m1'] = m1
        self._values['files'][filename]['m2'] = m2
        self._values['files'][filename]['clock'] = clock
        self._values['files'][filename]['numpoints'] = len(w)

        m = m1 + np.multiply(m2, 2)
        ws = ''
        for i in range(0, len(w)):
            ws = ws + struct.pack('<fB', w[i], int(m[i]))
        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 1000\n'
        s5 = ws
        s6 = 'CLOCK %.10e\n' % clock

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen = str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = s1 + s2 + s3 + s4 + s5 + s6
        self.visa_handle.write(mes)

    def send_pattern(self, w, m1, m2, filename, clock):
        '''
        Sends a pattern file.
        similar to waveform except diff file extension
        number of poitns different. diff byte conversion
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
        logging.debug(__name__ + ' : Sending pattern %s to instrument' % filename)

        # Check for errors
        dim = len(w)
        if (not((len(w)==len(m1)) and ((len(m1)==len(m2))))):
            return 'error'
        self._values['files'][filename]={}
        self._values['files'][filename]['w']=w
        self._values['files'][filename]['m1']=m1
        self._values['files'][filename]['m2']=m2
        self._values['files'][filename]['clock']=clock
        self._values['files'][filename]['numpoints']=len(w)

        m = m1 + np.multiply(m2, 2)
        ws = ''
        for i in range(0, len(w)):
            ws = ws + struct.pack('<fB', w[i], int(m[i]))

        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 2000\n'
        s5 = ws
        s6 = 'CLOCK %.10e\n' % clock

        s4 = '#' + str(len(str(len(s5)))) + str(len(s5))
        lenlen=str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3))

        mes = s1 + s2 + s3 + s4 + s5 + s6
        self.visa_handle.write(mes)


    def resend_waveform(self, channel, w=[], m1=[], m2=[], clock=[]):
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
        logging.debug(__name__ + ' : Resending %s to channel %s' % (filename, channel))


        if (w==[]):
            w = self._values['recent_channel_%s' %channel]['w']
        if (m1==[]):
            m1 = self._values['recent_channel_%s' %channel]['m1']
        if (m2==[]):
            m2 = self._values['recent_channel_%s' %channel]['m2']
        if (clock==[]):
            clock = self._values['recent_channel_%s' %channel]['clock']

        if not ( (len(w) == self._numpoints) and (len(m1) == self._numpoints) and (len(m2) == self._numpoints)):
            logging.error(__name__ + ' : one (or more) lengths of waveforms do not match with numpoints')

        self.send_waveform(w, m1, m2, filename, clock)
        self.do_set_filename(filename, channel)

    def delete_all_waveforms_from_list(self):
        '''
        for compatibillity with awg, is not relevant for AWG520 since it
        has no waveform list
        '''
        pass

    def send_sequence(self, wfs, rep, wait, goto, logic_jump, filename):
        '''
        Sends a sequence file (for the moment only for ch1)

        Args:

           wfs:  list of filenames

        Returs:

            None
        '''
        logging.debug(__name__ + ' : Sending sequence %s to instrument' % filename)
        N = str(len(rep))
        try:
            wfs.remove(N*[None])
        except ValueError:
            pass
        s1 = 'MMEM:DATA "%s",' % filename

        if len(np.shape(wfs)) ==1:
            s3 = 'MAGIC 3001\n'
            s5 = ''
            for k in range(len(rep)):
                s5 = s5+ '"%s",%s,%s,%s,%s\n'%(wfs[k],rep[k],wait[k],goto[k],logic_jump[k])

        else:
            s3 = 'MAGIC 3002\n'
            s5 = ''
            for k in range(len(rep)):
                s5 = s5+ '"%s","%s",%s,%s,%s,%s\n'%(wfs[0][k],wfs[1][k],rep[k],wait[k],goto[k],logic_jump[k])

        s4 = 'LINES %s\n'%N
        lenlen=str(len(str(len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s5) + len(s4) + len(s3))


        mes = s1 + s2 + s3 + s4 + s5
        self.visa_handle.write(mes)

    def send_sequence2(self,wfs1,wfs2,rep,wait,goto,logic_jump,filename):
        '''
        Sends a sequence file

        Args:
            wfs1:  list of filenames for ch1 (all must end with .pat)
            wfs2: list of filenames for ch2 (all must end with .pat)
            rep: list
            wait: list
            goto: list
            logic_jump: list
            filename: name of output file (must end with .seq)

        Returns:
            None
        '''
        logging.debug(__name__ + ' : Sending sequence %s to instrument' % filename)


        N = str(len(rep))
        s1 = 'MMEM:DATA "%s",' % filename
        s3 = 'MAGIC 3002\n'
        s4 = 'LINES %s\n'%N
        s5 = ''


        for k in range(len(rep)):
            s5 = s5+ '"%s","%s",%s,%s,%s,%s\n'%(wfs1[k],wfs2[k],rep[k],wait[k],goto[k],logic_jump[k])

        lenlen=str(len(str(len(s5) + len(s4) + len(s3))))
        s2 = '#' + lenlen + str(len(s5) + len(s4) + len(s3))


        mes = s1 + s2 + s3 + s4 + s5
        self.visa_handle.write(mes)

    def set_sequence(self,filename):
        '''
        loads a sequence file on all channels.
        Waveforms/patterns to be executed on respective channel
        must be defined inside the sequence file itself
        make sure to send all waveforms before setting a seq
        '''
        self.visa_handle.write('SOUR%s:FUNC:USER "%s","MAIN"' % (1, filename))

    def load_and_set_sequence(self,wfs,rep,wait,goto,logic_jump,filename):
        '''
        Loads and sets the awg sequecne
        '''
        self.send_sequence(wfs,rep,wait,goto,logic_jump,filename)
        self.set_sequence(filename)

