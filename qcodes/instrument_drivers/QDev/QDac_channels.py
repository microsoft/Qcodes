# QCoDeS driver for QDac using channels

import logging
import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyvisa as visa
from pyvisa.resources.serial import SerialInstrument

from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import (
    ChannelList,
    InstrumentChannel,
    MultiChannelInstrumentParameter,
)
from qcodes.instrument.parameter import ParamRawDataType
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)


class QDacChannel(InstrumentChannel):
    """
    A single output channel of the QDac.

    Exposes chan.v, chan.vrange, chan.slope, chan.i, chan.irange
    """

    _CHANNEL_VALIDATION = vals.Numbers(1, 48)

    def __init__(self, parent: Instrument, name: str, channum: int):
        """
        Args:
            parent: The instrument to which the channel is
                attached.
            name: The name of the channel
            channum: The number of the channel in question (1-48)
        """
        super().__init__(parent, name)

        # Validate the channel
        self._CHANNEL_VALIDATION.validate(channum)

        # Add the parameters

        self.add_parameter('v',
                           label=f'Channel {channum} voltage',
                           unit='V',
                           set_cmd=partial(self._parent._set_voltage, channum),
                           get_cmd=partial(self._parent._get_voltage, channum),
                           get_parser=float,
                           vals=vals.Numbers(-10, 10)
                           )

        self.add_parameter('vrange',
                           label=f'Channel {channum} atten.',
                           set_cmd=partial(self._parent._set_vrange, channum),
                           get_cmd=partial(self._parent._get_vrange, channum),
                           vals=vals.Enum(0, 1)
                           )

        self.add_parameter('i',
                           label=f'Channel {channum} current',
                           get_cmd=f'get {channum}',
                           unit='A',
                           get_parser=self._parent._current_parser
                           )

        self.add_parameter('irange',
                           label=f'Channel {channum} irange',
                           set_cmd=f'cur {channum} {{}}',
                           get_cmd=f'cur {channum}',
                           get_parser=int
                           )

        self.add_parameter('slope',
                           label=f'Channel {channum} slope',
                           unit='V/s',
                           set_cmd=partial(self._parent._setslope, channum),
                           get_cmd=partial(self._parent._getslope, channum),
                           vals=vals.MultiType(vals.Enum('Inf'),
                                               vals.Numbers(1e-3, 100))
                           )

        self.add_parameter('sync',
                           label=f'Channel {channum} sync output',
                           set_cmd=partial(self._parent._setsync, channum),
                           get_cmd=partial(self._parent._getsync, channum),
                           vals=vals.Ints(0, 5)
                           )

        self.add_parameter(name='sync_delay',
                           label=f'Channel {channum} sync pulse delay',
                           unit='s',
                           get_cmd=None, set_cmd=None,
                           initial_value=0
                           )

        self.add_parameter(name='sync_duration',
                           label=f'Channel {channum} sync pulse duration',
                           unit='s',
                           get_cmd=None, set_cmd=None,
                           initial_value=0.01
                           )

    def snapshot_base(
            self,
            update: Optional[bool] = False,
            params_to_skip_update: Optional[Sequence[str]] = None
    ) -> Dict[Any, Any]:
        update_currents = self._parent._update_currents and update
        if update and not self._parent._get_status_performed:
            self._parent._update_cache(readcurrents=update_currents)
        # call get_status rather than getting the status individually for
        # each parameter. This is only done if _get_status_performed is False
        # this is used to signal that the parent has already called it and
        # no need to repeat.
        if params_to_skip_update is None:
            params_to_skip_update = ('v', 'i', 'irange', 'vrange')
        snap = super().snapshot_base(update=update,
                                     params_to_skip_update=params_to_skip_update)
        return snap


class QDacMultiChannelParameter(MultiChannelInstrumentParameter):
    """
    The class to be returned by __getattr__ of the ChannelList. Here customised
    for fast multi-readout of voltages.
    """
    def __init__(
            self,
            channels: Sequence[InstrumentChannel],
            param_name: str,
            *args: Any,
            **kwargs: Any):
        super().__init__(channels, param_name, *args, **kwargs)

    def get_raw(self) -> Tuple[ParamRawDataType, ...]:
        """
        Return a tuple containing the data from each of the channels in the
        list.
        """
        # For voltages, we can do something slightly faster than the naive
        # approach

        if self._param_name == 'v':
            qdac = self._channels[0]._parent
            qdac._update_cache(readcurrents=False)
            output = tuple(chan.parameters[self._param_name].get_latest()
                           for chan in self._channels)
        else:
            output = tuple(chan.parameters[self._param_name].get()
                           for chan in self._channels)

        return output


class QDac(VisaInstrument):
    """
    Channelised driver for the QDev digital-analog converter QDac

    Based on "DAC_commands_v_13.pdf"
    Tested with Software Version: 0.170202

    The driver assumes that the instrument is ALWAYS in verbose mode OFF
    """


    # set nonzero value (seconds) to accept older status when reading settings
    max_status_age = 1

    def __init__(self,
                 name: str,
                 address: str,
                 num_chans: int = 48,
                 update_currents: bool = True,
                 **kwargs: Any):
        """
        Instantiates the instrument.

        Args:
            name: The instrument name used by qcodes
            address: The VISA name of the resource
            num_chans: Number of channels to assign. Default: 48
            update_currents: Whether to query all channels for their
                current current value on startup. Default: True.

        Returns:
            QDac object
        """
        super().__init__(name, address, **kwargs)
        self._output_n_lines = 50
        handle = self.visa_handle
        assert isinstance(handle, SerialInstrument)
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

        if self._get_firmware_version() < 0.170202:
            raise RuntimeError('''
                               Obsolete QDAC Software version detected.
                               QCoDeS only supports version 0.170202 or newer.
                               Contact rikke.lutge@nbi.ku.dk for an update.
                               ''')

        self.num_chans = num_chans

        # Assigned slopes. Entries will eventually be [chan, slope]
        self._slopes: List[Tuple[int, Union[str, float]]] = []
        # Function generators (used in _set_voltage)
        self._fgs = set(range(1, 9))
        self._assigned_fgs: Dict[int, int] = {}  # {chan: fg}
        # Sync channels
        self._syncoutputs: List[Tuple[int, int]] = []  # Entries: [chan, syncchannel]

        self.chan_range = range(1, 1 + self.num_chans)
        self.channel_validator = vals.Ints(1, self.num_chans)

        channels = ChannelList(self, "Channels", QDacChannel,
                               snapshotable=False,
                               multichan_paramclass=QDacMultiChannelParameter)

        for i in self.chan_range:
            channel = QDacChannel(self, f'chan{i:02}', i)
            channels.append(channel)
            # Should raise valueerror if name is invalid (silently fails now)
            self.add_submodule(f"ch{i:02}", channel)
        self.add_submodule("channels", channels.to_channel_tuple())

        for board in range(6):
            for sensor in range(3):
                label = f'Board {board}, Temperature {sensor}'
                self.add_parameter(name=f'temp{board}_{sensor}',
                                   label=label,
                                   unit='C',
                                   get_cmd=f'tem {board} {sensor}',
                                   get_parser=self._num_verbose)

        self.add_parameter(name='cal',
                           set_cmd='cal {}',
                           vals=self.channel_validator)
        # TO-DO: maybe it's too dangerous to have this settable.
        # And perhaps ON is a better verbose mode default?
        self.add_parameter(name='verbose',
                           set_cmd='ver {}',
                           val_mapping={True: 1, False: 0})

        # Initialise the instrument, all channels DC (unbind func. generators)
        for chan in self.chan_range:
            # Note: this call does NOT change the voltage on the channel
            self.write(f'wav {chan} 0 1 0')

        self.verbose.set(False)
        self.connect_message()
        log.info('[*] Querying all channels for voltages and currents...')
        self._update_cache(readcurrents=update_currents)
        self._update_currents = update_currents
        log.info('[+] Done')

    def snapshot_base(
            self,
            update: Optional[bool] = False,
            params_to_skip_update: Optional[Sequence[str]] = None
    ) -> Dict[Any, Any]:
        update_currents = self._update_currents and update is True
        if update:
            self._update_cache(readcurrents=update_currents)
            self._get_status_performed = True
        # call get_status rather than getting the status individually for
        # each parameter. We set _get_status_performed to True
        # to indicate that each update channel does not need to call this
        # function as opposed to when snapshot is called on an individual
        # channel
        snap = super().snapshot_base(update=update,
                                     params_to_skip_update=params_to_skip_update)
        self._get_status_performed = False
        return snap

    #########################
    # Channel gets/sets
    #########################

    def _set_voltage(
            self,
            chan: int,
            v_set: float) -> None:
        """
        set_cmd for the chXX_v parameter

        Args:
            chan: The 1-indexed channel number
            v_set: The target voltage

        If a finite slope has been assigned, we assign a function generator to
        ramp the voltage.
        """
        channel = self.channels[chan-1]

        slopechans = [sl[0] for sl in self._slopes]
        if chan in slopechans:
            slope = [sl[1] for sl in self._slopes if sl[0] == chan][0]
            # find and assign fg
            fg = min(self._fgs.difference(set(self._assigned_fgs.values())))
            self._assigned_fgs[chan] = fg
            # We need .get and not get_latest in case a ramp was interrupted
            v_start = channel.v.get()
            time = abs(v_set-v_start)/slope
            log.info(f'Slope: {slope}, time: {time}')
            # Attenuation compensation and syncing
            # happen inside _rampvoltage
            self._rampvoltage(chan, fg, v_start, v_set, time)
        else:
            v_dac = QDac._get_v_dac_from_v_exp(channel, v_set)
            # set the mode back to DC in case it had been changed
            # and then set the voltage
            self.write(f'wav {chan} 0 0 0;set {chan} {v_dac:.6f}')

    def _get_voltage(self, chan: int) -> float:
        """
        get_cmd for the chXX_v parameter

        Args:
            chan: The 1-indexed channel number
        """
        self._update_cache(readcurrents=False)
        return self.channels[chan - 1].v.cache()


    # In order to get conversions right let us define a vocabulary:
    # v_exp: is the voltage including the attenuation.This is the value
    # we want to store in the cache and the value we interact with as a
    # qcodes user.
    # v_dac: this is the voltage generated by the dac and handled by the VISA
    # commands.
    # Then we have the general relationship`v_exp = v_dac * attenuation`,
    @staticmethod
    def _get_attenuation(channel: QDacChannel) -> float:
        return 0.1 if channel.vrange.cache() == 1 else 1.0

    @staticmethod
    def _get_v_dac_from_v_exp(channel: QDacChannel, v_exp: float) -> float:
        return v_exp / QDac._get_attenuation(channel)

    @staticmethod
    def _get_v_exp_from_v_dac(channel: QDacChannel, v_dac: float) -> float:
        return v_dac * QDac._get_attenuation(channel)

    def _set_vrange(self, chan: int, switchint: int) -> None:
        """
        set_cmd for the chXX_vrange parameter

        The switchint is an integer. 1 means attenuation ON.

        Since the vrange is actually a 20 dB attenuator (amplitude factor 0.1)
        immediately applied to the channel output, we must update the voltage
        parameter accordingly
        """

        self.write(f'vol {chan} {switchint}')

        # setting v_range preserves v_dac but changes v_exp, see comment above
        # for definitions.
        channel = self.channels[chan-1]
        if channel.vrange.cache() != switchint:
            v_dac = QDac._get_v_dac_from_v_exp(channel, channel.v.cache())
            channel.vrange.cache.set(switchint)
            self._update_v_validator(channel, switchint)
            channel.v.cache.set(QDac._get_v_exp_from_v_dac(channel, v_dac))

    def _get_vrange(self, chan: int) -> float:
        """
        get_cmd for the chXX_vrange parameter

        Args:
            chan: The 1-indexed channel number
        """
        self._update_cache(readcurrents=False)
        return self.channels[chan - 1].vrange.cache()

    def _num_verbose(self, s: str) -> float:
        """
        turn a return value from the QDac into a number.
        If the QDac is in verbose mode, this involves stripping off the
        value descriptor.
        """
        if self.verbose.get_latest():
            s = s.split(': ')[-1]
        return float(s)

    def _current_parser(self, s: str) -> float:
        """
        parser for chXX_i parameter
        """
        return 1e-6*self._num_verbose(s)

    def _update_cache(self, readcurrents: bool = False) -> None:
        r"""
        Function to query the instrument and get the status of all channels,
        e.g. voltage (``v``), voltage range (``vrange``), and current range (``irange``)
        parameters of all the channels.
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
        """

        def validate_version(version_line: str) -> None:
            if version_line.startswith('Software Version: '):
                self.version = version_line.strip().split(': ')[1]
            else:
                self._wait_and_clear()
                raise ValueError('unrecognized version line: ' + version_line)

        def validate_header(header_line: str) -> None:
            headers = header_line.lower().strip('\r\n').split('\t')
            expected_headers = ['channel', 'out v', '', 'voltage range',
                                'current range']
            if headers != expected_headers:
                raise ValueError('unrecognized header line: ' + header_line)

        def parse_line(line: str) -> Tuple[int, int, int, float]:
            i_range_trans = {'hi cur': 1, 'lo cur': 0}
            v_range_trans = {'X 1': 0, 'X 0.1': 1}

            chan_str, v_str, _, v_range_str, _, i_range_str = line.split('\t')
            chan = int(chan_str)
            v_dac = float(v_str)
            v_range = v_range_trans[v_range_str.strip()]
            i_range = i_range_trans[i_range_str.strip()]
            return chan, i_range, v_range, v_dac

        validate_version(self.ask('status'))
        validate_header(self.read())

        chans_left = set(self.chan_range)
        while chans_left:
            line = self.read().strip()
            if not line:
                continue
            chan, i_range, v_range, v_dac = parse_line(line)

            channel = self.channels[chan - 1]
            channel.vrange.cache.set(v_range)
            self._update_v_validator(channel, v_range)
            channel.irange.cache.set(i_range)
            channel.v.cache.set(QDac._get_v_exp_from_v_dac(channel, v_dac))

            chans_left.remove(chan)

        if readcurrents:
            self._read_currents()

    def _read_currents(self) -> None:
        for chan in range(1, self.num_chans + 1):
            param = self.channels[chan - 1].i
            _ = param.get()

    @staticmethod
    def _update_v_validator(channel: QDacChannel, v_range: int) -> None:
        range = (-10.01, 10.01) if v_range == 0 else (-1.001, 1.001)
        channel.v.vals = vals.Numbers(*range)

    def _setsync(self, chan: int, sync: int) -> None:
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
            oldsync = self.channels[chan-1].sync.get_latest()
            if oldsync is not None:
                self.write(f'syn {oldsync} 0 0 0')
            return

        if sync in [syn[1] for syn in self._syncoutputs]:
            oldchan = [syn[0] for syn in self._syncoutputs if syn[1] == sync][0]
            self._syncoutputs.remove((oldchan, sync))

        if chan in [syn[0] for syn in self._syncoutputs]:
            oldsyn = [syn[1] for syn in self._syncoutputs if syn[0] == chan][0]
            self._syncoutputs[
                self._syncoutputs.index((chan, oldsyn))
            ] = (chan, sync)
            return

        self._syncoutputs.append((chan, sync))
        return

    def _getsync(self, chan: int) -> float:
        """
        get_cmd of the chXX_sync parameter
        """
        if chan in [syn[0] for syn in self._syncoutputs]:
            sync = [syn[1] for syn in self._syncoutputs if syn[0] == chan][0]
            return sync
        else:
            return 0

    def _setslope(self, chan: int, slope: Union[float, str]) -> None:
        """
        set_cmd for the chXX_slope parameter, the maximum slope of a channel.

        Args:
            chan: The channel number (1-48)
            slope: The slope in V/s. Write 'Inf' to allow
              arbitrary small rise times.
        """
        if chan not in range(1, 49):
            raise ValueError('Channel number must be 1-48.')

        if slope == 'Inf':
            self.write(f'wav {chan} 0 0 0')

            # Now clear the assigned slope and function generator (if possible)
            try:
                self._assigned_fgs.pop(chan)
            except KeyError:
                pass
            # Remove a sync output, if one was assigned
            syncchans = [syn[0] for syn in self._syncoutputs]
            if chan in syncchans:
                self.channels[chan-1].sync.set(0)
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
            self._slopes[self._slopes.index((chan, oldslope))] = (chan, slope)
            return

        if len(self._slopes) >= 8:
            rampchans = ", ".join(str(c[0]) for c in self._slopes)
            raise ValueError(
                "Can not assign finite slope to more than "
                + "8 channels. Assign 'Inf' to at least one of "
                + f"the following channels: {rampchans}"
            )

        self._slopes.append((chan, slope))
        return

    def _getslope(self, chan: int) -> Union[str, float]:
        """
        get_cmd of the chXX_slope parameter
        """
        if chan in [sl[0] for sl in self._slopes]:
            slope = [sl[1] for sl in self._slopes if sl[0] == chan][0]
            return slope
        else:
            return 'Inf'

    def printslopes(self) -> None:
        """
        Print the finite slopes assigned to channels
        """
        for sl in self._slopes:
            print(f"Channel {sl[0]}, slope: {sl[1]} (V/s)")

    def _rampvoltage(
            self,
            chan: int,
            fg: int,
            v_start: float,
            setvoltage: float,
            ramptime: float
    ) -> None:
        """
        Smoothly ramp the voltage of a channel by the means of a function
        generator. Helper function used by _set_voltage.

        Args:
            chan: The channel number (counting from 1)
            fg: The function generator (counting from 1)
            setvoltage: The voltage to ramp to
            ramptime: The ramp time in seconds.
        """

        # Crazy stuff happens if the period is too small, e.g. the channel
        # can jump to its max voltage
        if ramptime <= 0.002:
            ramptime = 0
            log.warning('Cancelled a ramp with a ramptime of '
                        '{} s'.format(ramptime) + '. Voltage not changed.')

        offset = v_start
        amplitude = setvoltage-v_start
        if self.channels[chan-1].vrange.get_latest() == 1:
            offset *= 10
            amplitude *= 10

        chanmssg = 'wav {} {} {} {}'.format(chan, fg,
                                            amplitude,
                                            offset)

        if chan in [syn[0] for syn in self._syncoutputs]:
            sync = [syn[1] for syn in self._syncoutputs if syn[0] == chan][0]
            sync_duration = 1000*self.channels[chan-1].sync_duration.get()
            sync_delay = 1000*self.channels[chan-1].sync_delay.get()
            self.write('syn {} {} {} {}'.format(sync, fg,
                                                sync_delay,
                                                sync_duration))

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

    def write(self, cmd: str) -> None:
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

        log.debug(f"Writing to instrument {self.name}: {cmd}")

        self.visa_handle.write(cmd)
        for _ in range(cmd.count(';')+1):
            self._write_response = self.visa_handle.read()

    def read(self) -> str:
        return self.visa_handle.read()

    def _wait_and_clear(self, delay: float = 0.5) -> None:
        time.sleep(delay)
        self.visa_handle.clear()

    def connect_message(self,
                        idn_part: str = "IDN",
                        being_time: Optional[float] = None) -> None:
        """
        Override of the standard Instrument class connect_message.
        Usually, the response to `*IDN?` is printed. Here, the
        software version is printed.
        """
        self.visa_handle.write('status')

        log.info('Connected to QDac on {}, {}'.format(self._address,
                                                      self.visa_handle.read()))

        # take care of the rest of the output
        for _ in range(self._output_n_lines):
            self.visa_handle.read()

    def _get_firmware_version(self) -> float:
        self.write('status')
        FW_str = self._write_response
        FW_version = float(FW_str.replace('Software Version: ', ''))
        for _ in range(self._output_n_lines):
            self.read()
        return FW_version

    def print_overview(self, update_currents: bool = False) -> None:
        """
        Pretty-prints the status of the QDac
        """

        self._update_cache(readcurrents=update_currents)

        paramstoget = [['i', 'v'], ['irange', 'vrange']]
        printdict = {'i': 'Current', 'v': 'Voltage', 'vrange': 'Voltage range',
                     'irange': 'Current range'}

        returnmap = {'vrange': {1: '-1 V to 1 V', 0: '-10 V to 10 V'},
                     'irange': {0: '0 to 1 muA', 1: '0 to 100 muA'}}

        # Print the channels
        for ii in range(self.num_chans):
            line = f"Channel {ii+1} \n"
            line += "    "
            for pp in paramstoget[0]:
                param = getattr(self.channels[ii], pp)
                line += printdict[pp]
                line += f': {param.get_latest()}'
                line += f' ({param.unit})'
                line += '. '
            line += '\n    '
            for pp in paramstoget[1]:
                param = getattr(self.channels[ii], pp)
                line += printdict[pp]
                value = param.get_latest()
                line += f": {returnmap[pp][value]}"
                line += ". "
            print(line)
