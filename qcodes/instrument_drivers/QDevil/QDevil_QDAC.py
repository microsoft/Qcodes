# QCoDeS driver for the QDevil QDAC using channels
# Adapted by QDevil from "qdev\QDac_channels.py" in
# the instrument drivers package
# Version 2.0 QDevil 2019-11-19

import time
import visa
import logging

from datetime import datetime
from functools import partial
from operator import xor
from collections import OrderedDict

from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.instrument.channel import MultiChannelInstrumentParameter
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)

mode_dict = {0: "V-high / I-high", 1: "V-high / I-low", 2: "V-low / I-low"}


class Mode:
    vhigh_ihigh = 0
    vhigh_ilow = 1
    vlow_ilow = 2


class Waveform:
    # Enum-like class defining the built-in waveform types
    sine = 1
    square = 2
    triangle = 3
    staircase = 4
    all = [sine, square, triangle, staircase]


class Generator():
    fg = 0
    t_end = 9.9e9

    def __init__(self, generator_number):
        self.fg = generator_number


class QDacChannel(InstrumentChannel):
    """
    A single output channel of the QDac.

    Exposes chan.v, chan.i, chan.mode, chan.slope,
            chan.sync, chan.sync_delay, chan.sync_duration
    Always set v to zero before changing range
    """

    _CHANNEL_VALIDATION = vals.Numbers(1, 48)

    def __init__(self, parent, name, channum):
        """
        Args:
            parent (Instrument): The instrument to which the channel belongs.
            name (str): The name of the channel
            channum (int): The number of the channel (1-48)
        """
        super().__init__(parent, name)

        # Validate the channel
        self._CHANNEL_VALIDATION.validate(channum)

        # Add the parameters
        self.add_parameter(name='v',
                           label='Channel {} voltage'.format(channum),
                           unit='V',
                           set_cmd=partial(self._parent._set_voltage, channum),
                           get_cmd='set {}'.format(channum),
                           get_parser=float,
                           # Initial values. Updated on init and during
                           # operation:
                           vals=vals.Numbers(-9.99, 9.99)
                           )

        self.add_parameter(name='mode',
                           label='Channel {} mode.'.format(channum),
                           set_cmd=partial(self._parent._set_mode, channum),
                           vals=vals.Enum(0, 1, 2)
                           )

        self.add_parameter(name='i',
                           label='Channel {} current'.format(channum),
                           get_cmd='get {}'.format(channum),
                           unit='A',
                           get_parser=self._parent._current_parser
                           )

        self.add_parameter(name='slope',
                           label='Channel {} slope'.format(channum),
                           unit='V/s',
                           set_cmd=partial(self._parent._setslope, channum),
                           get_cmd=partial(self._parent._getslope, channum),
                           vals=vals.MultiType(vals.Enum('Inf'),
                                               vals.Numbers(1e-3, 1000))
                           )

        self.add_parameter(name='sync',
                           label='Channel {} sync output'.format(channum),
                           set_cmd=partial(self._parent._setsync, channum),
                           get_cmd=partial(self._parent._getsync, channum),
                           vals=vals.Ints(0, 5)
                           )

        self.add_parameter(name='sync_delay',
                           label='Channel {} sync pulse delay'.format(channum),
                           unit='s',
                           get_cmd=None, set_cmd=None,
                           initial_value=0
                           )

        self.add_parameter(
                        name='sync_duration',
                        label='Channel {} sync pulse duration'.format(channum),
                        unit='s',
                        get_cmd=None, set_cmd=None,
                        initial_value=0.01
                        )

    def snapshot_base(self, update=False, params_to_skip_update=None):
        update_currents = self._parent._update_currents and update
        if update and not self._parent._get_status_performed:
            self._parent._get_status(readcurrents=update_currents)
        # call get_status rather than getting the status individually for
        # each parameter. This is only done if _get_status_performed is False
        # this is used to signal that the parent has already called it and
        # no need to repeat.
        if params_to_skip_update is None:
            params_to_skip_update = ('v', 'i', 'mode')
        snap = super().snapshot_base(
                            update=update,
                            params_to_skip_update=params_to_skip_update)
        return snap


class QDacMultiChannelParameter(MultiChannelInstrumentParameter):
    """
    The class to be returned by __getattr__ of the ChannelList. Here customised
    for fast multi-readout of voltages.
    """
    def __init__(self, channels, param_name, *args, **kwargs):
        super().__init__(channels, param_name, *args, **kwargs)

    def get_raw(self):
        """
        Return a tuple containing the data from each of the channels in the
        list.
        """
        # For voltages, we can do something slightly faster than the naive
        # approach

        if self._param_name == 'v':
            qdac = self._channels[0]._parent
            qdac._get_status(readcurrents=False)
            output = tuple(chan.parameters[self._param_name].get_latest()
                           for chan in self._channels)
        else:
            output = tuple(chan.parameters[self._param_name].get()
                           for chan in self._channels)

        return output


class QDac(VisaInstrument):
    """
    Channelised driver for the QDevil QDAC voltage source
    Exposes channels, temperature sensors and calibration output,
    and 'ramp_voltages' for mutil channel ramping.
    Tested with Firmware Version: 1.07

    The driver assumes that the instrument is ALWAYS in verbose mode OFF
    and sets this as part of the initialization, so please do not change this.
    """

    # set nonzero value (seconds) to accept older status when reading settings
    max_status_age = 1

    def __init__(self,
                 name,
                 address,
                 update_currents=False,
                 **kwargs):
        """
        Instantiates the instrument.

        Args:
            name (str): The instrument name used by qcodes
            address (str): The VISA name of the resource
            update_currents (bool): Whether to query all channels for their
                current sensor value on startup, which takes about 0.5 sec
                per channel. Default: False.

        Returns:
            QDac object
        """

        super().__init__(name, address, **kwargs)
        handle = self.visa_handle
        self._get_status_performed = False

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
        firmware_version = self._get_firmware_version()
        if firmware_version < 1.07:
            log.warning("Firmware version: {}".format(firmware_version))
            raise RuntimeError('''
                No QDevil QDAC detected or the firmware version is obsolete.
                This driver only supports version 1.07 or newer. Please
                contact info@qdevil.com for a firmware update.
                ''')

        self.num_chans = self._get_number_of_channels()
        num_boards = int(self.num_chans/8)
        self._output_n_lines = self.num_chans + 2

        # Assigned slopes. Entries will eventually be {chan: slope}
        self._slopes = {}
        # Function generators and triggers (used in ramping)
        self._fgs = set(range(1, 9))
        self._assigned_fgs = {}  # {chan: fg}
        self._trigs = set(range(1, 10))
        self._assigned_triggers = {}  # {fg: trigger}
        # Sync channels
        self._syncoutputs = {}  # {chan: syncoutput}

        self._chan_range = range(1, 1 + self.num_chans)
        self.channel_validator = vals.Ints(1, self.num_chans)

        channels = ChannelList(self, "Channels", QDacChannel,
                               snapshotable=False,
                               multichan_paramclass=QDacMultiChannelParameter)

        for i in self._chan_range:
            channel = QDacChannel(self, 'chan{:02}'.format(i), i)
            channels.append(channel)
            self.add_submodule('ch{:02}'.format(i), channel)
        channels.lock()
        self.add_submodule('channels', channels)

        for board in range(num_boards):
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

        self.add_parameter(name='verbose',
                           label=label,
                           val_mapping={True: 1, False: 0})

        # Initialise the instrument, all channels DC (unbind func. generators)
        for chan in self._chan_range:
            # Note: this call may change the voltage on the channel
            self.write('wav {} 0 1 0'.format(chan))

        # Get all calibrated min/max output values, before switching to
        # verbose off mode
        self.vranges = {}
        for chan in self._chan_range:
            self.vranges.update(
                {chan: {0: self._get_minmax_outputvoltage(chan, 0),
                        1: self._get_minmax_outputvoltage(chan, 1)}})

        self.write('ver 0')
        self.verbose._save_val(False)
        self.connect_message()
        log.info('[*] Querying all channels for voltages and currents...')
        self._get_status(readcurrents=update_currents)
        self._update_currents = update_currents
        # Set "v" parameter limits to actual calibrated limits
        self._set_vvals_to_current_range()
        log.info('[+] Done')

    def snapshot_base(self, update=False, params_to_skip_update=None):
        update_currents = self._update_currents and update
        if update:
            self._get_status(readcurrents=update_currents)
        self._get_status_performed = True
        # call get_status rather than getting the status individually for
        # each parameter. We set _get_status_performed to True
        # to indicate that each update channel does not need to call this
        # function as opposed to when snapshot is called on an individual
        # channel
        snap = super().snapshot_base(
                                update=update,
                                params_to_skip_update=params_to_skip_update)
        self._get_status_performed = False
        return snap

    #########################
    # Channel gets/sets
    #########################

    def _set_voltage(self, chan, v_set):
        """
        set_cmd for the chXX_v parameter

        Args:
            chan (int): The 1-indexed channel number
            v_set (float): The target voltage

        If a finite slope has been assigned, a function generator will
        ramp the voltage.
        """

        slope = self._slopes.get(chan, None)
        if slope:
            # We need .get and not get_latest in case a ramp was interrupted
            v_start = self.channels[chan-1].v.get()
            duration = abs(v_set-v_start)/slope
            log.info('Slope: {}, time: {}'.format(slope, duration))
            # SYNCing happens inside ramp_voltages
            self.ramp_voltages([chan], [v_start], [v_set], duration)
        else:
            self.write('wav {} 0 0 0;set {} {:.6f}'.format(chan, chan, v_set))

    # Helper for _set_mode
    def _clipto(self, value, min, max, errmsg=""):
        if value > max:
            log.warning(errmsg)
            return max
        elif value < min:
            log.warning(errmsg)
            return min
        else:
            return value

    # Helper for _set_mode. It is not possible ot say if the channel is
    # connected to a generator, so we need to ask.
    def _wav_or_set_msg(self, chan, new_voltage):
        self.write('wav {}'.format(chan))
        FW_str = self._write_response
        gen, _, _ = FW_str.split(',')
        if int(gen) > 0:
            # The amplitude must be set to zero to avoid potential overflow.
            # Assuming that voltage range is not changed during a ramp
            return 'wav {} {} {:.6f} {:.6f}'\
                    .format(chan, int(gen), 0, new_voltage)
        else:
            return 'set {} {:.6f}'.format(chan, new_voltage)

    def _set_mode(self, chan, new_mode):
        """
        set_cmd for the QDAC's mode (combined voltage and current sense range)
        """
        # Mode 0: (10V, 100uA), Mode 1: (10V, 1uA), Mode 2: (1V, 1uA)
        ivrangedict = {
                        0: {"vol": 0, "cur": 1},
                        1: {"vol": 0, "cur": 0},
                        2: {"vol": 1, "cur": 0}}
        # switchint =  (0,1,2,
        #               3,4,5,
        #               6,7,8)

        old_mode = self.channels[chan-1].mode.get_latest()
        new_vrange = ivrangedict[new_mode]["vol"]
        new_irange = ivrangedict[new_mode]["cur"]
        switchint = int(3*old_mode+new_mode)
        message = ''

        if old_mode == new_mode:
            return

        # If the voltage range is going to change we have to take care of
        # setting the voltage after the switch, and therefore read them first
        # Only the current range has to change
        if switchint in [1, 3]:
            message = 'cur {} {}'.format(chan, new_irange)
        # The voltage range has to change
        else:  # switchint in [2,5,6,7] (7 is max)
            if switchint == 2:
                message = 'cur {} {};'.format(chan, new_irange)
            old_voltage = self.channels[chan-1].v.get()
            new_voltage = self._clipto(
                    # Actually, for 6,7 new_voltage = old_voltage, always.
                    old_voltage, self.vranges[chan][new_vrange]['Min'],
                    self.vranges[chan][new_vrange]['Max'],
                    "Voltage is outside the bounds of the new voltage range"
                    " and is therefore clipped.")
            message += 'vol {} {};'.format(chan, new_vrange)
            message += self._wav_or_set_msg(chan, new_voltage)
            if switchint == 6:
                message += ';cur {} {}'.format(chan, new_irange)
            self.channels[chan-1].v.vals = vals.Numbers(
                    self.vranges[chan][new_vrange]['Min'],
                    self.vranges[chan][new_vrange]['Max'])
            self.channels[chan-1].v._save_val(new_voltage)
        self.write(message)

    def _vrange(self, range):
        if range == Mode.vlow_ilow:
            return 1
        else:
            return 0

    def _irange(self, range):
        if range == Mode.vhigh_ihigh:
            return 1
        else:
            return 0

    def _set_vvals_to_current_range(self):
        """
        Command for setting all 'v' limits ('vals') of all channels to the
        actual calibrated output limits for the range each individual channel
        is currently in.
        """
        for chan in range(1, self.num_chans+1):
            vrange = self._vrange(self.channels[chan-1].mode())
            self.channels[chan-1].v.vals = vals.Numbers(
                        self.vranges[chan][vrange]['Min'],
                        self.vranges[chan][vrange]['Max'])

    def _num_verbose(self, s):
        """
        Turns a return value from the QDac into a number.
        If the QDac is in verbose mode, this involves stripping off the
        value descriptor.
        """
        if self.verbose.get_latest():
            s = s.split[': '][-1]
        return float(s)

    def _current_parser(self, s):
        """
        Parser for chXX_i parameter (converts from uA to A)
        """
        return 1e-6*self._num_verbose(s)

    def _get_status(self, readcurrents=False):
        """
        Function to query the instrument and get the status of all channels.
        Takes a while to finish.

        The `status` call generates 27 or 51 lines of output. Send the command
        and read the first one, which is the software version line
        the full output looks like:
        Software Version: 1.07\r\n
        Channel\tOut V\t\tVoltage range\tCurrent range\n
        \n
        8\t  0.000000\t\tX 1\t\tpA\n
        7\t  0.000000\t\tX 1\t\tpA\n
        ... (all 24/48 channels like this)
        (no termination afterward besides the \n ending the last channel)
        """
        irange_trans = {'hi cur': 1, 'lo cur': 0}
        vrange_trans = {'X 1': 0, 'X 0.1': 1}
        vi_range_dict = {0: {0: 1, 1: 0}, 1: {0: 2, 1: 3}}
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

        chans_left = set(self._chan_range)
        while chans_left:
            line = self.read().strip()
            if not line:
                continue
            chanstr, v, _, vrange, _, irange = line.split('\t')
            chan = int(chanstr)
            vrange_int = int(vrange_trans[vrange.strip()])
            irange_int = int(irange_trans[irange.strip()])
            mode = vi_range_dict[vrange_int][irange_int]
            self.channels[chan-1].mode._save_val(mode)
            self.channels[chan-1].v._save_val(float(v))
            chans_left.remove(chan)

        if readcurrents:
            for chan in self._chan_range:
                self.channels[chan-1].i.get()

    def _setsync(self, chan, sync):
        """
        set_cmd for the chXX_sync parameter.

        Args:
            chan (int): The channel number (1-48 or 1-24)
            sync (int): The associated sync output (1-3 on 24 ch units
            or 1-5 on 48 ch units). 0 means 'unassign'
        """

        if chan not in range(1, self.num_chans+1):
            raise ValueError(
                    'Channel number must be 1-{}.'.format(self.num_chans))

        if sync == 0:
            oldsync = self.channels[chan-1].sync.get_latest()
            # try to remove the sync from internal bookkeeping
            self._syncoutputs.pop(chan, None)
            # free the previously assigned sync
            if oldsync is not None:
                self.write('syn {} 0 0 0'.format(oldsync))
            return

        if sync in self._syncoutputs.values():
            oldchan = [ch for ch, sy in self._syncoutputs.items()
                       if sy == sync]
            self._syncoutputs.pop(oldchan[0], None)
        self._syncoutputs[chan] = sync
        return

    def _getsync(self, chan):
        """
        get_cmd of the chXX_sync parameter
        """
        return self._syncoutputs.get(chan, 0)

    def _setslope(self, chan, slope):
        """
        set_cmd for the chXX_slope parameter, the maximum slope of a channel.
        With a finite slope the channel will be ramped using a generator.

        Args:
            chan (int): The channel number (1-24 or 1-48)
            slope (Union[float, str]): The slope in V/s.
            Write 'Inf' to release the channelas slope channel and to release
            the associated function generator. The output rise time will now
            only depend on the analog electronics.
        """
        if chan not in range(1, self.num_chans+1):
            raise ValueError(
                        'Channel number must be 1-{}.'.format(self.num_chans))

        if slope == 'Inf':
            # Set the channel in DC mode
            v_set = self.channels[chan-1].v.get()
            self.write('set {} {:.6f};wav {} 0 0 0'.format(chan, v_set, chan))

            # Now release the function generator and fg trigger (if possible)
            try:
                fg = self._assigned_fgs[chan]
                self._assigned_fgs[chan].t_end = 0
                self._assigned_triggers.pop(fg)
            except KeyError:
                pass

            # Remove a sync output, if one was assigned
            if chan in self._syncoutputs:
                self.channels[chan-1].sync.set(0)
            # Now clear the assigned slope
            self._slopes.pop(chan, None)
        else:
            self._slopes[chan] = slope
        return

    def _getslope(self, chan):
        """
        get_cmd of the chXX_slope parameter
        """
        return self._slopes.get(chan, 'Inf')

    def print_slopes(self):
        """
        Print the finite slopes assigned to channels, sorted by channel number
        """
        for chan, slope in sorted(self._slopes.items()):
            print('Channel {}, slope: {} (V/s)'.format(chan, slope))

    def _get_minmax_outputvoltage(self, channel, vrange_int):
        """
        Returns a dictionary of the calibrated Min and Max output
        voltages of 'channel' for the voltage given range (0,1) given by
        'vrange_int'
        """
        # For firmware 1.07 verbose mode and nn verbose mode give verbose
        # result, So this is designed for verbose mode
        if channel not in range(1, self.num_chans+1):
            raise ValueError(
                        'Channel number must be 1-{}.'.format(self.num_chans))
        if vrange_int not in range(0, 2):
            raise ValueError('Range must be 0 or 1.')

        self.write('rang {} {}'.format(channel, vrange_int))
        FW_str = self._write_response
        return {'Min': float(FW_str.split('MIN:')[1].split('MAX')[0].strip()),
                'Max': float(FW_str.split('MAX:')[1].strip())}

    def write(self, cmd):
        """
        QDac always returns something even from set commands, even when
        verbose mode is off, so we'll override write to take this out
        if you want to use this response, we put it in self._write_response
        (but only for the very last write call)

        In this method we expect to read one termination char per command. As
        commands are concatenated by `;` we count the number of concatenated
        commands as count(';') + 1 e.g. 'wav 1 1 1 0;fun 2 1 100 1 1' is two
        commands. Note that only the response of the last command will be
        available in `_write_response`
        """

        log.debug("Writing to instrument {}: {}".format(self.name, cmd))
        _, ret_code = self.visa_handle.write(cmd)
        self.check_error(ret_code)
        for _ in range(cmd.count(';')+1):
            self._write_response = self.visa_handle.read()

    def read(self):
        return self.visa_handle.read()

    def _wait_and_clear(self, delay=0.5):
        time.sleep(delay)
        self.visa_handle.clear()

    def connect_message(self, idn_param='IDN', begin_time=None):
        """
        Override of the standard Instrument class connect_message.
        Usually, the response to `*IDN?` is printed. Here, the
        software version is printed.
        """
        self.visa_handle.write('version')
        log.info('Connected to QDAC on {}, {}'.format(
                                    self._address, self.visa_handle.read()))

    def _get_firmware_version(self):
        """
        Check if the "version" command reponds. If so we probbaly have a QDevil
        QDAC, and the version number is returned. Otherwise 0.0 is returned.
        """
        self.write('version')
        FW_str = self._write_response
        if ((not ("Unrecognized command" in FW_str))
                and ("Software Version: " in FW_str)):
            FW_version = float(
                self._write_response.replace("Software Version: ", ""))
        else:
            FW_version = 0.0
        return FW_version

    def _get_number_of_channels(self):
        """
        Returns the number of channels for the instrument
        """
        self.write('boardNum')
        FW_str = self._write_response
        return 8*int(FW_str.strip("numberOfBoards:"))

    def print_overview(self, update_currents=False):
        """
        Pretty-prints the status of the QDac
        """

        self._get_status(readcurrents=update_currents)

        paramstoget = [['v', 'i'], ['mode']]  # Second item to be translated
        printdict = {'v': 'Voltage', 'i': 'Current', 'mode': 'Mode'}
        returnmap = {'mode': mode_dict}

        # Print the channels
        for ii in range(self.num_chans):
            line = 'Channel {} \n'.format(ii+1)
            line += '    '
            for pp in paramstoget[0]:
                param = getattr(self.channels[ii], pp)
                line += printdict[pp]
                line += ': {}'.format(param.get_latest())
                line += ' ({})'.format(param.unit)
                line += '. '
            line += '\n    '
            for pp in paramstoget[1]:
                param = getattr(self.channels[ii], pp)
                line += printdict[pp]
                value = param.get_latest()
                line += ': {}'.format(returnmap[pp][value])
                line += '. '
            print(line)

    def _get_functiongenerator(self, chan):
        """
        Function for getting a free generator (of 8 available) for a channel.
        Used as helper function for ramp_voltages, but may also be used if the
        user wants to use a function generator for something else.
        If there are no free generators this function will wait for up to
        FGS_TIMEOUT for one to be ready.

        To mark a function generator as available for others set
        self._assigned_fgs[chan].t_end = 0

        Args:
            chan: (1..24/48) the channel for which a function generator is
                  requested.
        """
        FGS_TIMEOUT = 2  # Max time to wait for next available generator

        if len(self._assigned_fgs) < 8:
            fg = min(self._fgs.difference(
                        set([g.fg for g in self._assigned_fgs.values()])))
            self._assigned_fgs[chan] = Generator(fg)
        else:
            # If no available fgs, see if one is soon to be ready
            time_now = time.time()
            available_fgs_chans = []
            fgs_t_end_ok = [g.t_end for chan, g
                            in self._assigned_fgs.items()
                            if g.t_end < time_now+FGS_TIMEOUT]
            if len(fgs_t_end_ok) > 0:
                first_ready_t = min(fgs_t_end_ok)
                available_fgs_chans = [chan for chan, g
                                       in self._assigned_fgs.items()
                                       if g.t_end == first_ready_t]
                if first_ready_t > time_now:
                    log.warning('''
                    Trying to ramp more channels than there are generators.\n
                    Waiting for ramp generator to be released''')
                    time.sleep(first_ready_t - time_now)

            if len(available_fgs_chans) > 0:
                oldchan = available_fgs_chans[0]
                fg = self._assigned_fgs[oldchan].fg
                self._assigned_fgs.pop(oldchan)
                self._assigned_fgs[chan] = Generator(fg)
                # Set the old channel in DC mode
                v_set = self.channels[oldchan-1].v.get_latest()
                self.write('set {} {:.6f};wav {} 0 0 0'
                           .format(oldchan, v_set, oldchan))
            else:
                raise RuntimeError('''
                Trying to ramp more channels than there are generators
                available. Please insert delays allowing channels to finish
                ramping before trying to ramp other channels, or reduce the
                number of ramped channels. Or increase FGS_TIMEOUT.''')
        return fg

    def ramp_voltages(self, channellist, v_startlist, v_endlist, ramptime):
        """
        Function for smoothly ramping one channel or more channels
        simultaneously (max. 8). This is a shallow interface to
        ramp_voltages_2D. Function generators and triggers are
        are assigned automatically.

        Args:
            channellist:    List (int) of channels to be ramped (1 indexed)\n
            v_startlist:    List (int) of voltages to ramp from.
                            MAY BE EMPTY. But if provided, time is saved by
                            NOT reading the present values from the instrument.

            v_endlist:      List (int) of voltages to ramp to.\n
            ramptime:       Total ramp time in seconds (min. 0.002). Has
                            to be an integer number of 0.001 secs).\n
        Returns:
            Estimated time of the excecution of the 2D scan.

        NOTE: This function returns as the ramps are started. So you need
        to wait for 'ramptime' until measuring....
        """

        if ramptime < 0.002:
            log.warning('Ramp time too short: {:.3f} s. Ramp time set to 2 ms.'
                        .format(ramptime))
            ramptime = 0.002
        steps = int(ramptime*1000)
        return self.ramp_voltages_2D(
                            slow_chans=[], slow_vstart=[], slow_vend=[],
                            fast_chans=channellist, fast_vstart=v_startlist,
                            fast_vend=v_endlist, step_length=0.001,
                            slow_steps=1, fast_steps=steps)

    def ramp_voltages_2D(self, slow_chans, slow_vstart, slow_vend,
                         fast_chans, fast_vstart, fast_vend,
                         step_length, slow_steps, fast_steps):
        """
        Function for smoothly ramping two channel groups simultaneously with
        one slow (x) and one fast (y) group. used by 'ramp_voltages' where x is
        empty. Function generators and triggers are assigned automatically.

        Args:
            slow_chans:   List (int) of channels to be ramped (1 indexed) in
                          the slow-group\n
            slow_vstart:  List (int) of voltages to ramp from in the
                          slow-group.
                          MAY BE EMPTY. But if provided, time is saved by NOT
                          reading the present values from the instrument.\n
            slow_vend:    list (int) of voltages to ramp to in the slow-group.

            slow_steps:   (int) number of steps in the x direction.\n
            fast_chans:   List (int) of channels to be ramped (1 indexed) in
                          the fast-group.\n
            fast_vstart:  List (int) of voltages to ramp from in the
                          fast-group.
                          MAY BE EMPTY. But if provided, time is saved by NOT
                          reading the present values from the instrument.\n
            fast_vend:    list (int) of voltages to ramp to in the fast-group.

            fast_steps:   (int) number of steps in the fast direction.\n
            step_length:  (float) Time spent at each step in seconds
                          (min. 0.001) multiple of 1 ms.\n
        Returns:
            Estimated time of the excecution of the 2D scan.\n
        NOTE: This function returns as the ramps are started.
        """
        channellist = [*slow_chans, *fast_chans]
        v_endlist = [*slow_vend, *fast_vend]
        v_startlist = [*slow_vstart, *fast_vstart]
        step_length_ms = int(step_length*1000)

        if step_length < 0.001:
            log.warning('step_length too short: {:.3f} s. \nstep_length set to'
                        .format(step_length_ms) + ' minimum (1ms).')
            step_length_ms = 1

        if any([ch in fast_chans for ch in slow_chans]):
            raise ValueError(
                    'Channel cannot be in both slow_chans and fast_chans!')

        no_channels = len(channellist)
        if no_channels != len(v_endlist):
            raise ValueError(
                    'Number of channels and number of voltages inconsistent!')

        for chan in channellist:
            if chan not in range(1, self.num_chans+1):
                raise ValueError(
                        'Channel number must be 1-{}.'.format(self.num_chans))
            if not (chan in self._assigned_fgs):
                self._get_functiongenerator(chan)

        # Voltage validation
        for i in range(no_channels):
            self.channels[channellist[i]-1].v.validate(v_endlist[i])
        if v_startlist:
            for i in range(no_channels):
                self.channels[channellist[i]-1].v.validate(v_startlist[i])

        # Get start voltages if not provided
        if not slow_vstart:
            slow_vstart = [self.channels[ch-1].v.get() for ch in slow_chans]

        if not fast_vstart:
            fast_vstart = [self.channels[ch-1].v.get() for ch in fast_chans]

        v_startlist = [*slow_vstart, *fast_vstart]
        if no_channels != len(v_startlist):
            raise ValueError(
                'Number of start voltages do not match number of channels!')

        # Find trigger not aleady uses (avoid starting other
        # channels/function generators)
        if no_channels == 1:
            trigger = 0
        else:
            trigger = int(min(self._trigs.difference(
                                    set(self._assigned_triggers.values()))))

        # Make sure any sync outputs are configured
        for chan in channellist:
            if chan in self._syncoutputs:
                sync = self._syncoutputs[chan]
                sync_duration = 1000*self.channels[chan-1].sync_duration.get()
                sync_delay = 1000*self.channels[chan-1].sync_delay.get()
                self.write('syn {} {} {} {}'.format(
                                            sync, self._assigned_fgs[chan].fg,
                                            sync_delay, sync_duration))

        # Now program the channel amplitudes and function generators
        msg = ''
        for i in range(no_channels):
            amplitude = v_endlist[i]-v_startlist[i]
            ch = channellist[i]
            fg = self._assigned_fgs[ch].fg
            if trigger > 0:  # Trigger 0 is not a trigger
                self._assigned_triggers[fg] = trigger
            msg += 'wav {} {} {} {}'.format(ch, fg, amplitude, v_startlist[i])
            # using staircase = function 4
            nsteps = slow_steps if ch in slow_chans else fast_steps
            repetitions = slow_steps if ch in fast_chans else 1

            delay = step_length_ms \
                if ch in fast_chans else fast_steps*step_length_ms
            msg += ';fun {} {} {} {} {} {};'.format(
                        fg, Waveform.staircase, delay, int(nsteps),
                        repetitions, trigger)
            # Update latest values to ramp end values
            # (actually not necessary when called from _set_voltage)
            self.channels[ch-1].v._save_val(v_endlist[i])
        self.write(msg[:-1])  # last semicolon is stripped

        # Fire trigger to start generators simultaneously, saving communication
        # time by not using triggers for single channel ramping
        if trigger > 0:
            self.write('trig {}'.format(trigger))

        # Update fgs dict so that we know when the ramp is supposed to end
        time_ramp = slow_steps * fast_steps * step_length_ms / 1000
        time_end = time_ramp + time.time()
        for i in range(no_channels):
            self._assigned_fgs[chan].t_end = time_end
        return time_ramp
