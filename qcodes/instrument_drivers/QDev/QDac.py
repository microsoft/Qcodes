# QCoDeS driver for QDac. Based on Alex Johnson's work.

import time
import visa
import logging
import numpy as np

from datetime import datetime
from functools import partial
from operator import xor
from collections import OrderedDict

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals
from qcodes.utils.deprecate import deprecate


log = logging.getLogger(__name__)


class QDac(VisaInstrument):
    """
    Driver for the QDev digital-analog converter QDac

    Based on "DAC_commands_v_13.pdf"
    Tested with Software Version: 0.170202

    The driver assumes that the instrument is ALWAYS in verbose mode OFF
    """

    voltage_range_status = {'X 1': 10, 'X 0.1': 1}

    # set nonzero value (seconds) to accept older status when reading settings
    max_status_age = 1

    @deprecate(
        reason=('the non-channelized version of the QDac'
                'is no longer supported'),
        alternative='qcodes.instrument_drivers.qdev.qdach_channels.QDac'
    )
    def __init__(self, name, address, num_chans=48, update_currents=True):
        """
        Instantiates the instrument.

        Args:
            name (str): The instrument name used by qcodes
            address (str): The VISA name of the resource
            num_chans (int): Number of channels to assign. Default: 48
            update_currents (bool): Whether to query all channels for their
                current current value on startup. Default: True.

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
        self._write_response = ''

        # The following bool is used in self.write
        self.debugmode = False

        if self._get_firmware_version() < 0.170202:
            raise RuntimeError('''
                               Obsolete QDAC Software version detected.
                               QCoDeS only supports version 0.170202 or newer.
                               Contact rikke.lutge@nbi.ku.dk for an update.
                               ''')
        self._update_currents = update_currents
        self.num_chans = num_chans

        # Assigned slopes. Entries will eventually be [chan, slope]
        self._slopes = []
        # Function generators (used in _set_voltage)
        self._fgs = set(range(1, 9))
        self._assigned_fgs = {}  # {chan: fg}
        # Sync channels
        self._syncoutputs = []  # Entries: [chan, syncchannel]

        self.chan_range = range(1, 1 + self.num_chans)
        self.channel_validator = vals.Ints(1, self.num_chans)
        self._params_to_skip_update = []
        for i in self.chan_range:
            self._params_to_skip_update.append('ch{:02}_v'.format(i))
            self._params_to_skip_update.append('ch{:02}_i'.format(i))
            self._params_to_skip_update.append('ch{:02}_vrange'.format(i))
            self._params_to_skip_update.append('ch{:02}_irange'.format(i))
            stri = str(i)
            self.add_parameter(name='ch{:02}_v'.format(i),
                               label='Channel ' + stri,
                               unit='V',
                               set_cmd=partial(self._set_voltage, i),
                               vals=vals.Numbers(-10, 10),
                               get_cmd=partial(self.read_state, i, 'v')
                               )
            self.add_parameter(name='ch{:02}_vrange'.format(i),
                               set_cmd=partial(self._set_vrange, i),
                               get_cmd=partial(self.read_state, i, 'vrange'),
                               vals=vals.Enum(0, 1)
                               )
            self.add_parameter(name='ch{:02}_irange'.format(i),
                               set_cmd='cur ' + stri + ' {}',
                               get_cmd=partial(self.read_state, i, 'irange')
                               )
            self.add_parameter(name='ch{:02}_i'.format(i),
                               label='Current ' + stri,
                               unit='A',
                               get_cmd='get ' + stri,
                               get_parser=self._current_parser
                               )
            self.add_parameter(name='ch{:02}_slope'.format(i),
                               label='Maximum voltage slope',
                               unit='V/s',
                               set_cmd=partial(self._setslope, i),
                               get_cmd=partial(self._getslope, i),
                               vals=vals.MultiType(vals.Enum('Inf'),
                                                   vals.Numbers(1e-3, 100))
                               )
            self.add_parameter(name='ch{:02}_sync'.format(i),
                               label='Channel {} sync output',
                               set_cmd=partial(self._setsync, i),
                               get_cmd=partial(self._getsync, i),
                               vals=vals.Ints(0, 5)
                               )

            self.add_parameter(name='ch{:02}_sync_delay'.format(i),
                               label='Channel {} sync pulse delay'.format(i),
                               unit='s',
                               get_cmd=None, set_cmd=None,
                               initial_value=0)

            self.add_parameter(name='ch{:02}_sync_duration'.format(i),
                               label='Channel {} sync pulse duration'.format(i),
                               unit='s',
                               get_cmd=None, set_cmd=None,
                               initial_value=0.01)

        for board in range(6):
            for sensor in range(3):
                label = 'Board {}, Temperature {}'.format(board, sensor)
                self.add_parameter(name='temp{}_{}'.format(board, sensor),
                                   label=label,
                                   unit='C',
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

        self.add_parameter(name='fast_voltage_set',
                           label='fast voltage set',
                           get_cmd=None, set_cmd=None,
                           vals=vals.Bool(),
                           initial_value=False,
                           docstring=""""Toggles if DC voltage set should unset any ramp attached to this channel.
                                     If you enable this you should ensure thay any function generator is unset
                                     from the channel before setting voltage""")
        # Initialise the instrument, all channels DC (unbind func. generators)
        for chan in self.chan_range:
            # Note: this call does NOT change the voltage on the channel
            self.write('wav {} 0 1 0'.format(chan))

        self.verbose.set(False)
        self.connect_message()
        log.info('[*] Querying all channels for voltages and currents...')
        self._get_status(readcurrents=update_currents)
        log.info('[+] Done')

    def snapshot_base(self, update=False, params_to_skip_update=None):
        # call get_status here if updates are requested
        # this is much faster than updating the individual channels
        update_currents = self._update_currents and update
        if update:
            self._get_status(readcurrents=update_currents)
        if params_to_skip_update is None:
            params_to_skip_update = self._params_to_skip_update
        supfun = super().snapshot_base
        snap = supfun(update=update,
                      params_to_skip_update=params_to_skip_update)
        return snap

    #########################
    # Channel gets/sets
    #########################

    def _set_voltage(self, chan, v_set):
        """
        set_cmd for the chXX_v parameter

        If a finite slope has been assigned, we assign a function generator to
        ramp the voltage.
        """
        # validation
        atten = self.parameters['ch{:02}_vrange'.format(chan)].get_latest()
        attendict = {0: 10, 1: 1, 10: 10}
        if abs(v_set) > attendict[atten]:
            v_set = np.sign(v_set)*attendict[atten]
            log.warning('Requested voltage outside reachable range.' +
                        ' Setting voltage on channel ' +
                        '{} to {} V'.format(chan, v_set))

        slopechans = [sl[0] for sl in self._slopes]
        if chan in slopechans:
            slope = [sl[1] for sl in self._slopes if sl[0] == chan][0]
            # find and assign fg
            fg = min(self._fgs.difference(set(self._assigned_fgs.values())))
            self._assigned_fgs[chan] = fg
            # We need .get and not get_latest in case a ramp was interrupted
            v_start = self.parameters['ch{:02}_v'.format(chan)].get()
            mytime = abs(v_set-v_start)/slope
            log.info('Slope: {}, time: {}'.format(slope, mytime))
            # Attenuation compensation and syncing
            # happen inside _rampvoltage
            self._rampvoltage(chan, fg, v_start, v_set, mytime)
        else:
            # compensate for the 0.1 multiplier, if it's on
            if self.parameters['ch{:02}_vrange'.format(chan)].get_latest() == 1:
                v_set = v_set*10
            # set the mode back to DC in case it had been changed
            if not self.fast_voltage_set():
                self.write('wav {} 0 0 0'.format(chan))
            self.write('set {} {:.6f}'.format(chan, v_set))

    def _set_vrange(self, chan, switchint):
        """
        set_cmd for the chXX_vrange parameter

        The switchint is an integer. 1 means attenuation ON.

        Since the vrange is actually a 20 dB attenuator (amplitude factor 0.1)
        immediately applied to the channel output, we must update the voltage
        parameter accordingly
        """

        tdict = {'-10 V to 10 V': 0,
                 '-1 V to 1 V': 1,
                 10: 0,
                 0: 0,
                 1: 1}

        old = tdict[self.parameters['ch{:02}_vrange'.format(chan)].get_latest()]

        self.write('vol {} {}'.format(chan, switchint))

        if xor(old, switchint):
            voltageparam = self.parameters['ch{:02}_v'.format(chan)]
            oldvoltage = voltageparam.get_latest()
            newvoltage = {0: 10, 1: 0.1}[switchint]*oldvoltage
            voltageparam.cache.set(newvoltage)

    def _num_verbose(self, s):
        """
        turn a return value from the QDac into a number.
        If the QDac is in verbose mode, this involves stripping off the
        value descriptor.
        """
        if self.verbose.get_latest():
            s = s.split[': '][-1]
        return float(s)

    def _current_parser(self, s):
        """
        parser for chXX_i parameter
        """
        return 1e-6*self._num_verbose(s)

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

        returnmap = {'vrange': {1: 1, 10: 0},
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

            # The following dict must be ordered to ensure that vrange comes
            # before v when iterating through it
            vals_dict = OrderedDict()
            vals_dict.update({'vrange': ('ch{}_vrange',
                              self.voltage_range_status[vrange.strip()])})
            vals_dict.update({'irange': ('ch{}_irange', irange_trans[irange])})
            vals_dict.update({'v': ('ch{}_v', float(v))})

            chans[chan - 1] = vals_dict
            for param in vals_dict:
                parameter = vals_dict[param][0].format(chanstr)
                value = vals_dict[param][1]
                if param == 'vrange':
                    attenuation = 0.1*value
                if param == 'v':
                    value *= attenuation
                self.parameters[parameter].cache.set(value)
            chans_left.remove(chan)

        if readcurrents:
            for chan in range(1, self.num_chans+1):
                paramname = 'ch{:02}_i'.format(chan)
                param = self.parameters[paramname]
                _ = param.get()

        self._status = chans
        self._status_ts = datetime.now()
        return chans

    def _setsync(self, chan, sync):
        """
        set_cmd for the chXX_sync parameter.

        Args:
            chan (int): The channel number (1-48)
            sync (int): The associated sync output. 0 means 'unassign'
        """

        if chan not in range(1, 49):
            raise ValueError('Channel number must be 1-48.')

        if sync == 0:
            # try to remove the sync from internal bookkeeping
            try:
                sc = self._syncoutputs
                to_remove = [sc.index(syn) for syn in sc if syn[0] == chan][0]
                self._syncoutputs.remove(sc[to_remove])
            except IndexError:
                pass
            # free the previously assigned sync
            oldsync = self.parameters['ch{:02d}_sync'.format(chan)].get_latest()
            if oldsync is not None:
                self.write('syn {} 0 0 0'.format(oldsync))
            return

        if sync in [syn[1] for syn in self._syncoutputs]:
            oldchan = [syn[0] for syn in self._syncoutputs if syn[1] == sync][0]
            self._syncoutputs.remove([oldchan, sync])

        if chan in [syn[0] for syn in self._syncoutputs]:
            oldsyn = [syn[1] for syn in self._syncoutputs if syn[0] == chan][0]
            self._syncoutputs[self._syncoutputs.index([chan, oldsyn])] = [chan,
                                                                          sync]
            return

        self._syncoutputs.append([chan, sync])
        return

    def _getsync(self, chan):
        """
        get_cmd of the chXX_sync parameter
        """
        if chan in [syn[0] for syn in self._syncoutputs]:
            sync = [syn[1] for syn in self._syncoutputs if syn[0]==chan][0]
            return sync
        else:
            return 0

    def _setslope(self, chan, slope):
        """
        set_cmd for the chXX_slope parameter, the maximum slope of a channel.

        Args:
            chan (int): The channel number (1-48)
            slope (Union[float, str]): The slope in V/s. Write 'Inf' to allow
              arbitraryly small rise times.
        """
        if chan not in range(1, 49):
            raise ValueError('Channel number must be 1-48.')

        if slope == 'Inf':
            self.write('wav {} 0 0 0'.format(chan))

            # Now clear the assigned slope and function generator (if possible)
            try:
                self._assigned_fgs.pop(chan)
            except KeyError:
                pass
            # Remove a sync output, if one was assigned
            syncchans = [syn[0] for syn in self._syncoutputs]
            if chan in syncchans:
                self.parameters['ch{:02d}_sync'.format(chan)].set(0)
            try:
                sls = self._slopes
                to_remove = [sls.index(sl) for sl in sls if sl[0] == chan][0]
                self._slopes.remove(sls[to_remove])
                return
            # If the value was already 'Inf', the channel was not
            # in the list and nothing happens
            except IndexError:
                return

        if chan in [sl[0] for sl in self._slopes]:
            oldslope = [sl[1] for sl in self._slopes if sl[0] == chan][0]
            self._slopes[self._slopes.index([chan, oldslope])] = [chan, slope]
            return

        if len(self._slopes) >= 8:
            rampchans = ', '.join([str(c[0]) for c in self._slopes])
            raise ValueError('Can not assign finite slope to more than ' +
                             "8 channels. Assign 'Inf' to at least one of " +
                             'the following channels: {}'.format(rampchans))

        self._slopes.append([chan, slope])
        return

    def _getslope(self, chan):
        """
        get_cmd of the chXX_slope parameter
        """
        if chan in [sl[0] for sl in self._slopes]:
            slope = [sl[1] for sl in self._slopes if sl[0]==chan][0]
            return slope
        else:
            return 'Inf'

    def printslopes(self):
        """
        Print the finite slopes assigned to channels
        """
        for sl in self._slopes:
            print('Channel {}, slope: {} (V/s)'.format(sl[0], sl[1]))

    def _rampvoltage(self, chan, fg, v_start, setvoltage, ramptime):
        """
        Smoothly ramp the voltage of a channel by the means of a function
        generator. Helper function used by _set_voltage.

        Args:
            chan (int): The channel number (counting from 1)
            fg (int): The function generator (counting from 1)
            setvoltage (float): The voltage to ramp to
            ramptime (float): The ramp time in seconds.
        """

        # Crazy stuff happens if the period is too small, e.g. the channel
        # can jump to its max voltage
        if ramptime <= 0.002:
            ramptime = 0
            log.warning('Cancelled a ramp with a ramptime of '
                        '{} s'.format(ramptime) + '. Voltage not changed.')

        offset = v_start
        amplitude = setvoltage-v_start
        if self.parameters['ch{:02}_vrange'.format(chan)].get_latest() == 1:
            offset *= 10
            amplitude *= 10

        chanmssg = 'wav {} {} {} {}'.format(chan, fg,
                                            amplitude,
                                            offset)

        if chan in [syn[0] for syn in self._syncoutputs]:
            #syncing = True
            sync = [syn[1] for syn in self._syncoutputs if syn[0] == chan][0]
            chstr = 'ch{:02}_sync'.format(chan)
            sync_duration = 1000*self.parameters[chstr+'_duration'].get()
            sync_delay = 1000*self.parameters[chstr+'_delay'].get()
            self.write('syn {} {} {} {}'.format(sync, fg,
                                                sync_delay,
                                                sync_duration))
        else:
            syncing = False

        typedict = {'SINE': 1, 'SQUARE': 2, 'RAMP': 3}

        typeval = typedict['RAMP']
        dutyval = 100
        # s -> ms
        periodval = ramptime*1e3
        repval = 1
        funmssg = 'fun {} {} {} {} {}'.format(fg,
                                              typeval, periodval,
                                              dutyval, repval)
        self.write(chanmssg)
        self.write(funmssg)


    def write(self, cmd):
        """
        QDac always returns something even from set commands, even when
        verbose mode is off, so we'll override write to take this out
        if you want to use this response, we put it in self._write_response
        (but only for the very last write call)

        Note that this procedure makes it very cumbersome to handle the returned
        messages from concatenated commands, e.g. 'wav 1 1 1 0;fun 2 1 100 1 1'
        Please don't use concatenated commands

        TODO (WilliamHPNielsen): add automatic de-concatenation of commands.
        """
        if self.debugmode:
            log.info('Sending command string: {}'.format(cmd))

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
        Usually, the response to `*IDN?` is printed. Here, the
        software version is printed.
        """
        self.visa_handle.write('status')

        log.info('Connected to QDac on {}, {}'.format(self._address,
                                                      self.visa_handle.read()))

        # take care of the rest of the output
        for ii in range(50):
            self.visa_handle.read()

    def _get_firmware_version(self):
        self.write('status')
        FW_str = self._write_response
        FW_version = float(FW_str.replace('Software Version: ', ''))
        for ii in range(50):
            self.read()
        return FW_version

    def print_overview(self, update_currents=False):
        """
        Pretty-prints the status of the QDac
        """

        self._get_status(readcurrents=update_currents)

        paramstoget = [['i', 'v'], ['irange', 'vrange']]
        printdict = {'i': 'Current', 'v': 'Voltage', 'vrange': 'Voltage range',
                     'irange': 'Current range'}

        returnmap = {'vrange': {1: '-1 V to 1 V', 10: '-10 V to 10 V'},
                     'irange': {0: '0 to 1 muA', 1: '0 to 100 muA'}}

        # Print the channels
        for ii in range(self.num_chans):
            line = 'Channel {} \n'.format(ii+1)
            line += '    '
            for pp in paramstoget[0]:
                paramname = 'ch{:02}_'.format(ii+1) + pp
                param = self.parameters[paramname]
                line += printdict[pp]
                line += ': {}'.format(param.get_latest())
                line += ' ({})'.format(param.unit)
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
