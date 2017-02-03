# QCoDeS driver for QDac. Based on Alex Johnson's work.

import time
import visa
import logging

from datetime import datetime
from functools import partial
from time import sleep

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)


class QDac(VisaInstrument):
    """
    Driver for the QDev digital-analog converter QDac

    Based on "DAC_commands_v_13.pdf"
    Tested with Software Version: 0.160915

    The driver assumes that the instrument is ALWAYS in verbose mode OFF
    """

    voltage_range_status = {'X 1': 10, 'X 0.1': 1}

    # set nonzero value (seconds) to accept older status when reading settings
    max_status_age = 1

    def __init__(self, name, address, num_chans=48):
        """
        Instantiates the instrument.

        Args:
            name (str): The instrument name used by qcodes
            address (str): The VISA name of the resource
            num_chans (int): Number of channels to assign. Default: 48

        Returns:
            QDac object
        """
        super().__init__(name, address)
        handle = self.visa_handle

        # This is the baud rate on power-up. It can be changed later but
        # you must start out with this value.
        handle.baud_rate = 480600
        handle.parity = visa.constants.Parity(0)
        handle.data_bits = 8
        self.set_terminator('\n')
        # TODO: do we want a method for write termination too?
        handle.write_termination = '\n'
        # TODO: do we need a query delay for robust operation?

        self.num_chans = num_chans

        # default values for all channels
        # DC mode, ampl=1V, offset=0V
        self._fungens = [0]*num_chans
        self._ampls = [1]*num_chans
        self._offsets = [0]*num_chans

        # default values for all function generators
        self._fg_types = ['SINE']*8
        self._fg_freqs = [1000]*8
        self._fg_dutys = [100]*8
        self._fg_nreps = [1]*8

        self.chan_range = range(1, 1 + self.num_chans)
        self.channel_validator = vals.Ints(1, self.num_chans)
        self.fungen_range = range(1, 9)

        for i in self.chan_range:
            stri = str(i)
            self.add_parameter(name='ch{:02}_v'.format(i),
                               label='Channel ' + stri,
                               units='V',
                               # TO-DO: implement max slope for setting
                               set_cmd=partial(self._set_voltage, i),
                               vals=vals.Numbers(-10, 10),
                               get_cmd=partial(self.read_state, i, 'v')
                               )
            self.add_parameter(name='ch{:02}_vrange'.format(i),
                               set_cmd='vol ' + stri + ' {}',
                               get_cmd=partial(self.read_state, i, 'vrange'),
                               vals=vals.Enum(0, 1)
                               )
            self.add_parameter(name='ch{:02}_irange'.format(i),
                               set_cmd='cur ' + stri + ' {}',
                               get_cmd=partial(self.read_state, i, 'irange')
                               )
            self.add_parameter(name='ch{:02}_i'.format(i),
                               label='Current ' + stri,
                               units='muA',
                               get_cmd='get ' + stri,
                               get_parser=self._num_verbose
                               )
            self.add_parameter(name='ch{:02}_fungen'.format(i),
                               label=('The number of the function generator ' +
                                      'to which the channel is bound'),
                               set_cmd=partial(self._set_fungen, i),
                               get_cmd=partial(self._get_fungen, i)
                               )
            self.add_parameter(name='ch{:02}_ampl'.format(i),
                               label=('The channel gain factor used when in ' +
                                      'func. gen. mode.'),
                               units='V',
                               set_cmd=partial(self._set_ampl, i),
                               get_cmd=partial(self._get_ampl, i),
                               vals=vals.Numbers(-10, 10)
                               )
            self.add_parameter(name='ch{:02}_offset'.format(i),
                               label=('The channel offset used when in ' +
                                      'func. gen. mode'),
                               units='V',
                               vals=vals.Numbers(-10, 10),
                               set_cmd=partial(self._set_offset, i),
                               get_cmd=partial(self._get_offset, i))

        for fg in self.fungen_range:
            self.add_parameter('fungen{}_type'.format(fg),
                               label='Type of output waveform',
                               set_cmd=partial(self._fg_setter, fg, 'type'),
                               get_cmd=partial(self._fg_getter, fg, 'type'),
                               vals=vals.Enum('SINE', 'SQUARE', 'RAMP')
                               )
            self.add_parameter('fungen{}_freq'.format(fg),
                               label='Frequency of output waveform',
                               units='Hz',
                               # TO-DO: add reasonable validation
                               # (what is a reasonable max freq?)
                               set_cmd=partial(self._fg_setter, fg, 'freq'),
                               get_cmd=partial(self._fg_getter, fg, 'freq')
                               )
            self.add_parameter('fungen{}_duty'.format(fg),
                               label='Duty cycle of output waveform',
                               units='Percent',
                               set_cmd=partial(self._fg_setter, fg, 'duty'),
                               get_cmd=partial(self._fg_getter, fg, 'duty')
                               )
            self.add_parameter('fungen{}_nreps'.format(fg),
                               label='No. of repetitions of output waveform',
                               vals=vals.Ints(-1, 2**31-1),
                               set_cmd=partial(self._fg_setter, fg, 'nreps'),
                               get_cmd=partial(self._fg_getter, fg, 'nreps')
                               )

        for board in range(6):
            for sensor in range(3):
                label = 'Board {}, Temperature {}'.format(board, sensor)
                self.add_parameter(name='temp{}_{}'.format(board, sensor),
                                   label=label,
                                   units='C',
                                   get_cmd='tem {} {}'.format(board, sensor),
                                   get_parser=self._num_verbose)

        self.add_parameter(name='cal',
                           set_cmd='cal {}',
                           vals=self.channel_validator)
        # TO-DO: maybe it's too dangerous to have this settable.
        # And perhaps ON is a better verbose mode default?
        self.add_parameter(name='verbose',
                           set_cmd='ver {}',
                           val_mapping={True: 1, False: 0})

        # initialise the instrument
        for chan in range(1, self.num_chans+1):
            self.parameters['ch{:02}_fungen'.format(chan)].set(0)
            self.parameters['ch{:02}_vrange'.format(chan)].set(0)
        self.verbose.set(False)
        self.connect_message()
        print('[*] Querying all channels for voltages and currents...')
        self._get_status(readcurrents=True)
        print('[+] Done')

    def run(self, fglist):
        """
        Starts outputting the waveform of the function generators in fglist

        Args:
            fglist (Union[list, int]): The number of the function generators
              to start outputting
        """

        # allow for inputting a single value
        if not isinstance(fglist, list):
            fglist = [fglist]

        # list of channels with function generators bound to them
        chanlist = []
        for fg in fglist:
            for chan in range(1, self.num_chans):
                if self._fungens[chan-1] == fg:
                    chanlist.append(chan)

        typedict = {'SINE': 1, 'SQUARE': 2, 'RAMP': 3}

        for chan in chanlist:
            chanmssg = 'wav {} {} {} {}'.format(chan, self._fungens[chan-1],
                                                self._ampls[chan-1],
                                                self._offsets[chan-1])
            #self.write()

            fungen = self._fungens[chan-1]-1
            typeval = typedict[self._fg_types[fungen]]
            dutyval = self._fg_dutys[fungen]
            # Hz -> ms
            periodval = round(1/self._fg_freqs[fungen]*1e3)
            repval = self._fg_nreps[fungen]
            funmssg = 'fun {} {} {} {} {}'.format(self._fungens[chan-1],
                                                   typeval, periodval,
                                                   dutyval, repval)
            self.write(chanmssg)
            self.write(funmssg)

    #########################
    # Channel gets/sets
    #########################

    def _set_voltage(self, chan, v):
        """
        set_cmd for the chXX_v parameter
        """

        # compensate for the 0.1 multiplier, if it's on
        if self.parameters['ch{:02}_vrange'.format(chan)].get_latest() == 1:
            v = v*10
        # set the mode back to DC in case it had been changed
        self.write('wav {} 0 0 0'.format(chan))
        self.write('set {} {:.6f}'.format(chan, v))

    def _set_fungen(self, chan, num):
        """
        set_cmd for the chXX_fungen parameter.
        """

        # TO-DO: implement support for AWG and pulse generator
        if num not in range(0, 9):
            raise ValueError('Can only bind function generators 1-8' +
                             ' and DC state 0')
        # Python lists are 0-indexed
        self._fungens[chan-1] = num
        # The 'wav' command is 1-indexed
        offset = self.parameters['ch{:02}_v'.format(chan)].get_latest()
        mssg = 'wav {} {} 1 {}'.format(chan, num, offset)
        self.write(mssg)

    def _get_fungen(self, chan):
        """
        get_cmd for the chXX_fungen parameter.
        """
        return self._fungens[chan-1]

    def _set_ampl(self, chan, ampl):
        """
        set_cmd for the chXX_ampl parameter.
        """
        if abs(ampl) + abs(self._offsets[chan-1]) > 10:
            log.warning('Amplitude and offset on channel {:02}'.format(chan) +
                        ' exceed the voltage range. Output signal will clip.')

        self._ampls[chan-1] = ampl

    def _get_ampl(self, chan):
        """
        get_cmd for the chXX_ampl parameter.
        """
        return self._ampls[chan-1]

    def _set_offset(self, chan, offset):
        """
        set_cmd for the chXX_offset parameter.
        """
        if abs(offset) + abs(self._ampls[chan-1]) > 10:
            log.warning('Amplitude and offset on channel {:02}'.format(chan) +
                        ' exceed the voltage range. Output signal will clip.')

        self._offsets[chan-1] = offset

    def _get_offset(self, chan):
        """
        set_cmd for the chXX_offset parameter.
        """
        return self._offsets[chan-1]

    #########################
    # Function generator gets/sets
    #########################

    def _fg_setter(self, fungen, argtype, arg):
        """
        generalised set_cmd for all the fungenXX_ parameters
        """
        argdict = {'type': self._fg_types,
                   'duty': self._fg_dutys,
                   'freq': self._fg_freqs,
                   'nreps': self._fg_nreps}

        argdict[argtype][fungen-1] = arg

    def _fg_getter(self, fungen, argtype):
        """
        generalised get_cmd for all the fungenXX_ parameters
        """
        argdict = {'type': self._fg_types,
                   'duty': self._fg_dutys,
                   'freq': self._fg_freqs,
                   'nreps': self._fg_nreps}

        return argdict[argtype][fungen-1]

    def _num_verbose(self, s):
        '''
        turn a return value from the QDac into a number.
        If the QDac is in verbose mode, this involves stripping off the
        value descriptor.
        '''
        if self.verbose.get_latest():
            s = s.split[': '][-1]
        return float(s)

    def read_state(self, chan, param):
        """
        specific routine for reading items out of status response
        """
        if chan not in self.chan_range:
            raise ValueError('valid channels are {}'.format(self.chan_range))
        valid_params = ('v', 'vrange', 'irange')
        if param not in valid_params:
            raise ValueError(
                'read_state valid params are {}'.format(valid_params))

        self._get_status(readcurrents=False)

        parameter = 'ch{:02}_{}'.format(chan, param)
        value = self.parameters[parameter].get_latest()

        returnmap = {'vrange': {1: '-1 V to 1 V', 10: '-10 V to 10 V'},
                     'irange': {0: '1 muA', 1: '100 muA'}}

        if 'range' in param:
            value = returnmap[param][value]

        return value

    def _get_status(self, readcurrents=False):
        r'''
        Function to query the instrument and get the status of all channels.
        Takes a while to finish.

        The `status` call generates 51 lines of output. Send the command and
        read the first one, which is the software version line
        the full output looks like:
        Software Version: 0.160218\r\n
        Channel\tOut V\t\tVoltage range\tCurrent range\n
        \n
        8\t  0.000000\t\tX 1\t\tpA\n
        7\t  0.000000\t\tX 1\t\tpA\n
        ... (all 48 channels like this in a somewhat peculiar order)
        (no termination afterward besides the \n ending the last channel)
        returns a list of dicts [{v, vrange, irange}]
        NOTE - channels are 1-based, but the return is a list, so of course
        0-based, ie chan1 is out[0]
        '''

        # Status call

        version_line = self.ask('status')
        print('Nasty debug')
        print('--')
        print(version_line)
        if version_line.startswith('Software Version: '):
            self.version = version_line.strip().split(': ')[1]
        else:
            self._wait_and_clear()
            raise ValueError('unrecognized version line: ' + version_line)

        header_line = self.read()
        headers = header_line.lower().strip('\r\n').split('\t')
        expected_headers = ['channel', 'out v', '', 'voltage range',
                            'current range']
        if headers != expected_headers:
            raise ValueError('unrecognized header line: ' + header_line)

        chans = [{} for i in self.chan_range]
        chans_left = set(self.chan_range)
        while chans_left:
            line = self.read().strip()
            if not line:
                continue
            chanstr, v, _, vrange, _, irange = line.split('\t')
            chan = int(chanstr)
            chanstr = '{:02}'.format(chan)

            irange_trans = {'hi cur': 1, 'lo cur': 0}

            vals_dict = {
                'v': ('ch{}_v', float(v)),
                'vrange': ('ch{}_vrange',
                           self.voltage_range_status[vrange.strip()]),
                'irange': ('ch{}_irange', irange_trans[irange])
            }

            chans[chan - 1] = vals_dict
            for param in vals_dict:
                parameter = vals_dict[param][0].format(chanstr)
                value = vals_dict[param][1]
                self.parameters[parameter]._save_val(value)

            chans_left.remove(chan)

        if readcurrents:
            for chan in range(1, self.num_chans+1):
                paramname = 'ch{:02}_i'.format(chan)
                param = self.parameters[paramname]
                param._save_val(param.get())

        self._status = chans
        self._status_ts = datetime.now()
        return chans

    def write(self, cmd):
        """
        QDac always returns something even from set commands, even when
        verbose mode is off, so we'll override write to take this out
        if you want to use this response, we put it in self._write_response
        (but only for the very last write call)
        """
        nr_bytes_written, ret_code = self.visa_handle.write(cmd)
        self.check_error(ret_code)
        self._write_response = self.visa_handle.read()

    def read(self):
        return self.visa_handle.read()

    def _wait_and_clear(self, delay=0.5):
        time.sleep(delay)
        self.visa_handle.clear()

    def connect_message(self):
        """
        Override of the standard Instrument class connect_message.
        Usually, the response to *IDN? is printed. Here, the
        software version is printed.
        """
        self.visa_handle.write('status')

        print('Connected to QDac on', self._address+',',
              self.visa_handle.read())

        # take care of the rest of the output
        for ii in range(50):
            self.visa_handle.read()

    def print_overview(self, update_currents=False):
        """
        Pretty-prints the status of the QDac
        """

        self._get_status(readcurrents=update_currents)

        paramstoget = [['i', 'v'], ['irange', 'vrange']]
        printdict = {'i': 'Current', 'v': 'Voltage', 'vrange': 'Voltage range',
                     'irange': 'Current range'}

        returnmap = {'vrange': {1: '-1 V to 1 V', 10: '-10 V to 10 V'},
                     'irange': {0: '1 muA', 1: '100 muA'}}

        # Print the channels
        for ii in range(self.num_chans):
            line = 'Channel {} \n'.format(ii+1)
            line += '    '
            for pp in paramstoget[0]:
                paramname = 'ch{:02}_'.format(ii+1) + pp
                param = self.parameters[paramname]
                line += printdict[pp]
                line += ': {}'.format(param.get_latest())
                line += ' ({})'.format(param.units)
                line += '. '
            line += '\n    '
            for pp in paramstoget[1]:
                paramname = 'ch{:02}_'.format(ii+1) + pp
                param = self.parameters[paramname]
                line += printdict[pp]
                value = param.get_latest()
                line += ': {}'.format(returnmap[pp][value])
                line += '. '
            print(line)
