# QCoDeS driver for the QDevil QDAC using channels
# Adapted by QDevil from "qdev\QDac_channels.py" in
# the instrument drivers package
# Version 2.1 QDevil 2020-02-10

from typing import Optional, Sequence, Dict, Any, Tuple, Union
import time
import pyvisa as visa
from pyvisa.resources.serial import SerialInstrument
import logging

from functools import partial
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.instrument.channel import MultiChannelInstrumentParameter
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals
from qcodes.instrument.parameter import ParamRawDataType
from enum import Enum
from collections import namedtuple

LOG = logging.getLogger(__name__)


_ModeTuple = namedtuple('Mode', 'v i')


class Mode(Enum):
    """
    Enum type use as the mode parameter for channels
    defining the combined voltage and current range.

    get_label() returns a text representation of the mode.
    """
    vhigh_ihigh = _ModeTuple(v=0, i=1)
    vhigh_ilow = _ModeTuple(v=0, i=0)
    vlow_ilow = _ModeTuple(v=1, i=0)

    def get_label(self) -> str:
        _MODE_LABELS = {
                "vhigh_ihigh": "V range high / I range high",
                "vhigh_ilow": "V range high / I range low",
                "vlow_ilow": "V range low / I range low"}
        return _MODE_LABELS[self.name]


class Waveform:
    # Enum-like class defining the built-in waveform types
    sine = 1
    square = 2
    triangle = 3
    staircase = 4
    all_waveforms = [sine, square, triangle, staircase]


class Generator:
    #  Class used in the internal book keeping of generators
    def __init__(self, generator_number: int):
        self.fg = generator_number
        self.t_end = 9.9e9


class QDacChannel(InstrumentChannel):
    """
    A single output channel of the QDac.

    Exposes chan.v, chan.i, chan.mode, chan.slope,
    chan.sync, chan.sync_delay, chan.sync_duration.\n
    NB: Set v to zero before changing mode if the
    mode_force lfag is False (default).
    """

    def __init__(self, parent: "QDac", name: str, channum: int):
        """
        Args:
            parent: The instrument to which the channel belongs.
            name: The name of the channel
            channum: The number of the channel (1-24 or 1-48)
        """
        super().__init__(parent, name)

        # Add the parameters
        self.add_parameter(name='v',
                           label=f'Channel {channum} voltage',
                           unit='V',
                           set_cmd=partial(self._parent._set_voltage, channum),
                           get_cmd=f'set {channum}',
                           get_parser=float,
                           # Initial range. Updated on init and during
                           # operation:
                           vals=vals.Numbers(-9.99, 9.99)
                           )

        self.add_parameter(name='mode',
                           label=f'Channel {channum} mode.',
                           set_cmd=partial(self._parent._set_mode, channum),
                           get_cmd=None,
                           vals=vals.Enum(*list(Mode))
                           )

        self.add_parameter(name='i',
                           label=f'Channel {channum} current',
                           get_cmd=f'get {channum}',
                           unit='A',
                           get_parser=self._parent._current_parser
                           )

        self.add_parameter(name='slope',
                           label=f'Channel {channum} slope',
                           unit='V/s',
                           set_cmd=partial(self._parent._setslope, channum),
                           get_cmd=partial(self._parent._getslope, channum),
                           vals=vals.MultiType(vals.Enum('Inf'),
                                               vals.Numbers(1e-3, 10000))
                           )

        self.add_parameter(name='sync',
                           label=f'Channel {channum} sync output',
                           set_cmd=partial(self._parent._setsync, channum),
                           get_cmd=partial(self._parent._getsync, channum),
                           vals=vals.Ints(0, 4)  # Updated at qdac init
                           )

        self.add_parameter(name='sync_delay',
                           label=f'Channel {channum} sync pulse delay',
                           unit='s',
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers(0, 10000),
                           initial_value=0
                           )

        self.add_parameter(
                        name='sync_duration',
                        label=f'Channel {channum} sync pulse duration',
                        unit='s',
                        get_cmd=None, set_cmd=None,
                        vals=vals.Numbers(0.001, 10000),
                        initial_value=0.01
                        )

    def snapshot_base(
            self,
            update: Optional[bool] = False,
            params_to_skip_update: Optional[Sequence[str]] = None) -> Dict:
        update_currents = self._parent._update_currents and update
        if update and not self._parent._get_status_performed:
            self._parent._update_cache(update_currents=update_currents)
        # call update_cache rather than getting the status individually for
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
    def __init__(self, channels: Sequence[InstrumentChannel],
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
        # approach by asking the instrument for a channel overview.

        if self._param_name == 'v':
            qdac = self._channels[0]._parent
            qdac._update_cache(update_currents=False)
            output = tuple(chan.parameters[self._param_name].cache()
                           for chan in self._channels)
        else:
            output = tuple(chan.parameters[self._param_name].get()
                           for chan in self._channels)

        return output


class QDac(VisaInstrument):
    """
    Channelised driver for the QDevil QDAC voltage source.

    Exposes channels, temperature sensors and calibration output,
    and 'ramp_voltages' + 'ramp_voltages_2d' for multi channel ramping.

    In addition a 'mode_force' flag (default False) is exposed.
    'mode_force' (=True) is used to enable voltage range switching, via
    the channel 'mode' parameter, even at non-zero output voltages.

    Tested with Firmware Version: 1.07

    The driver assumes that the instrument is ALWAYS in verbose mode OFF
    and sets this as part of the initialization, so please do not change this.
    """

    # set nonzero value (seconds) to accept older status when reading settings
    max_status_age = 1

    def __init__(self,
                 name: str,
                 address: str,
                 update_currents: bool = False,
                 **kwargs: Any):
        """
        Instantiates the instrument.

        Args:
            name: The instrument name used by qcodes
            address: The VISA name of the resource
            update_currents: Whether to query all channels for their
                current sensor value on startup, which takes about 0.5 sec
                per channel. Default: False.

        Returns:
            QDac object
        """

        super().__init__(name, address, **kwargs)
        handle = self.visa_handle
        self._get_status_performed = False

        assert isinstance(handle, SerialInstrument)
        # Communication setup + firmware check
        handle.baud_rate = 480600
        handle.parity = visa.constants.Parity(0)
        handle.data_bits = 8
        self.set_terminator('\n')
        handle.write_termination = '\n'
        self._write_response = ''
        firmware_version = self._get_firmware_version()
        if firmware_version < 1.07:
            LOG.warning(f"Firmware version: {firmware_version}")
            raise RuntimeError('''
                No QDevil QDAC detected or the firmware version is obsolete.
                This driver only supports version 1.07 or newer. Please
                contact info@qdevil.com for a firmware update.
                ''')

        # Initialse basic information and internal book keeping
        self.num_chans = self._get_number_of_channels()
        num_boards = int(self.num_chans/8)
        self._output_n_lines = self.num_chans + 2
        self._chan_range = range(1, 1 + self.num_chans)
        self.channel_validator = vals.Ints(1, self.num_chans)
        self._reset_bookkeeping()

        # Add channels (and channel parameters)
        channels = ChannelList(self, "Channels", QDacChannel,
                               snapshotable=False,
                               multichan_paramclass=QDacMultiChannelParameter)

        for i in self._chan_range:
            channel = QDacChannel(self, f'chan{i:02}', i)
            channels.append(channel)
            self.add_submodule(f'ch{i:02}', channel)
        channels.lock()
        self.add_submodule('channels', channels)

        # Updatechannel  sync port validator according to number of boards
        self._num_syns = max(num_boards-1, 1)
        for chan in self._chan_range:
            self.channels[chan-1].sync.vals = vals.Ints(0, self._num_syns)

        # Add non-channel parameters
        for board in range(num_boards):
            for sensor in range(3):
                label = f'Board {board}, Temperature {sensor}'
                self.add_parameter(name=f'temp{board}_{sensor}',
                                   label=label,
                                   unit='C',
                                   get_cmd=f'tem {board} {sensor}',
                                   get_parser=self._num_verbose)

        self.add_parameter(name='cal',
                           set_cmd='cal {}',
                           vals=vals.Ints(0, self.num_chans))

        self.add_parameter(name='mode_force',
                           label='Mode force',
                           get_cmd=None, set_cmd=None,
                           vals=vals.Bool(),
                           initial_value=False)

        # Due to a firmware bug in 1.07 voltage ranges are always reported
        # vebosely. So for future compatibility we set verbose True
        self.write('ver 1')
        self._update_voltage_ranges()
        # The driver require verbose mode off except for the above command
        self.write('ver 0')
        self._verbose = False  # Just so that the code can check the state
        self.connect_message()
        LOG.info('[*] Querying all channels for voltages and currents...')
        self._update_cache(update_currents=update_currents)
        self._update_currents = update_currents
        self._load_state()
        LOG.info('[+] Done')

    def _reset_bookkeeping(self) -> None:
        """
        Resets all internal variables used for ramping and
        synchronization outputs.
        """
        # Assigned slopes. Entries will eventually be {chan: slope}
        self._slopes:  Dict[int, Union[str, float]] = {}
        # Function generators and triggers (used in ramping)
        self._fgs = set(range(1, 9))
        self._assigned_fgs: Dict[int, Generator] = {}  # {chan: fg}
        self._trigs = set(range(1, 10))
        self._assigned_triggers: Dict[int, int] = {}  # {fg: trigger}
        # Sync channels
        self._syncoutputs: Dict[int, int] = {}  # {chan: syncoutput}

    def _load_state(self) -> None:
        """
        Used as part of initiaisation. DON'T use _load_state() separately.\n
        Updates internal book keeping of running function generators.
        used triggers and active sync outputs.\n
        Slopes can not be read/updated as it is not possible to
        say if a generator is running because a slope has been assigned
        or because it is being ramped direcly (by e.g. ramp_voltages_2d()).
        """
        # Assumes that all variables and virtual
        # parameters have been initialised (and read)

        self.write('ver 0')  # Just to be on the safe side

        self._reset_bookkeeping()
        for ch_idx in range(self.num_chans):
            chan = ch_idx + 1
            # Check if the channels are being ramped
            # It is not possible to find out if it has a slope assigned
            # as it may be ramped explicitely by the user
            # We assume that generators are running, but we cannot know
            self.write(f'wav {chan}')
            fg_str, amplitude_str, offset_str = self._write_response.split(',')
            amplitude = float(amplitude_str)
            offset = float(offset_str)
            fg = int(fg_str)
            if fg in range(1, 9):
                voltage = self.channels[ch_idx].v.get()
                time_now = time.time()
                self.write(f'fun {fg}')
                response = self._write_response.split(',')
                waveform = int(response[0])
                # Probably this driver is involved if a stair case is assigned
                if waveform == Waveform.staircase:
                    if len(response) == 6:
                        step_length_ms, no_steps, rep, rep_remain_str, trigger \
                            = response[1:6]
                        rep_remain = int(rep_remain_str)
                    else:
                        step_length_ms, no_steps, rep, trigger = response[1:5]
                        rep_remain = int(rep)
                    ramp_time = 0.001 * float(step_length_ms) * int(no_steps)
                    ramp_remain = 0
                    if (amplitude != 0):
                        ramp_remain = (amplitude+offset-voltage)/amplitude
                    if int(rep) == -1:
                        time_end = time_now + 315360000
                    else:
                        time_end = (ramp_remain + max(0, rep_remain-1)) \
                                   * ramp_time + time_now + 0.001
                else:
                    if waveform == Waveform.sine:
                        period_ms, rep, rep_remain_str, trigger = response[1:5]
                    else:
                        period_ms, _, rep, rep_remain_str, trigger = response[1:6]
                    if int(rep) == -1:
                        time_end = time_now + 315360000  # 10 years from now
                    else:  # +1 is just a safe guard
                        time_end = time_now + 0.001 \
                                   * (int(rep_remain_str)+1) * float(period_ms)

                self._assigned_fgs[chan] = Generator(fg)
                self._assigned_fgs[chan].t_end = time_end
                if trigger != 0:
                    self._assigned_triggers[fg] = int(trigger)
                for syn in range(1, self._num_syns+1):
                    self.write(f'syn {syn}')
                    syn_fg, delay_ms, duration_ms = \
                        self._write_response.split(',')
                    if int(syn_fg) == fg:
                        self.channels[ch_idx].sync.cache.set(syn)
                        self.channels[ch_idx].sync_delay(float(delay_ms)/1000)
                        self.channels[ch_idx].sync_duration(
                            float(duration_ms)/1000)

    def reset(self, update_currents: bool = False) -> None:
        """
        Resets the instrument setting all channels to zero output voltage
        and all parameters to their default values, including removing any
        assigned sync putputs, function generators, triggers etc.
        """
        # In case the QDAC has been switched off/on
        # clear the io buffer and set verbose False
        self.device_clear()
        self.write('ver 0')

        self.cal(0)
        # Resetting all slopes first will cause v.set() disconnect generators
        self.channels[0:self.num_chans].slope('Inf')
        self.channels[0:self.num_chans].v(0)
        self.channels[0:self.num_chans].mode(Mode.vhigh_ihigh)
        self.channels[0:self.num_chans].sync(0)
        self.channels[0:self.num_chans].sync_delay(0)
        self.channels[0:self.num_chans].sync_duration(0.01)

        if update_currents:
            self.channels[0:self.num_chans].i.get()
        self.mode_force(False)
        self._reset_bookkeeping()

    def snapshot_base(self,
                      update: Optional[bool] = False,
                      params_to_skip_update: Optional[Sequence[str]] = None) -> Dict:
        update_currents = self._update_currents and update is True
        if update:
            self._update_cache(update_currents=update_currents)
            self._get_status_performed = True
        # call _update_cache rather than getting the status individually for
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

    def _set_voltage(self, chan: int, v_set: float) -> None:
        """
        set_cmd for the chXX_v parameter

        Args:
            chan: The 1-indexed channel number
            v_set: The target voltage

        If a finite slope has been assigned, a function generator will
        ramp the voltage.
        """

        slope = self._slopes.get(chan, None)
        if slope:
            # We need .get and not cache/get_latest in case a ramp
            # was interrupted
            v_start = self.channels[chan-1].v.get()
            duration = abs(v_set-v_start)/slope
            LOG.info(f'Slope: {slope}, time: {duration}')
            # SYNCing happens inside ramp_voltages
            self.ramp_voltages([chan], [v_start], [v_set], duration)
        else:  # Should not be necessary to wav here.
            self.write('wav {ch} 0 0 0;set {ch} {voltage:.6f}'
                       .format(ch=chan, voltage=v_set))

    def _set_mode(self, chan: int, new_mode: Mode) -> None:
        """
        set_cmd for the QDAC's mode (combined voltage and current sense range).
        It is not possible to switch from voltage range without setting the
        the volage to zero first or set the global mode_force parameter True.
        """
        def _clipto(value: float, min_: float, max_: float) -> float:
            errmsg = ("Voltage is outside the bounds of the new voltage range"
                      " and is therefore clipped.")
            if value > max_:
                LOG.warning(errmsg)
                return max_
            elif value < min_:
                LOG.warning(errmsg)
                return min_
            else:
                return value

        # It is not possible ot say if the channel is connected to
        # a generator, so we need to ask.
        def wav_or_set_msg(chan: int, new_voltage: float) -> str:
            self.write(f'wav {chan}')
            fw_str = self._write_response
            gen, _, _ = fw_str.split(',')
            if int(gen) > 0:
                # The amplitude must be set to zero to avoid potential overflow
                # Assuming that voltage range is not changed during a ramp
                return 'wav {} {} {:.6f} {:.6f}'\
                        .format(chan, int(gen), 0, new_voltage)
            else:
                return f'set {chan} {new_voltage:.6f}'

        old_mode = self.channels[chan-1].mode.cache()
        new_vrange = new_mode.value.v
        old_vrange = old_mode.value.v
        new_irange = new_mode.value.i
        old_irange = old_mode.value.i
        message = ''
        max_zero_voltage = {0: 20e-6, 1: 3e-6}
        NON_ZERO_VOLTAGE_MSG = (
                'Please set the voltage to zero before changing the voltage'
                ' range in order to avoid jumps or spikes.'
                ' Or set mode_force=True to allow voltage range change for'
                ' non-zero voltages.')

        if old_mode == new_mode:
            return

        # If the voltage range is going to change we have to take care of
        # setting the voltage after the switch, and therefore read it first
        # We also need to make sure than only one of the voltage/current
        # relays is on at a time (otherwise the firmware will enforce it).

        if (new_irange != old_irange) and (new_vrange == old_vrange == 0):
            # Only the current sensor relay has to switch:
            message += f'cur {chan} {new_irange}'
        # The voltage relay (also) has to switch:
        else:
            # Current sensor relay on->off before voltage relay off->on:
            if new_irange < old_irange and new_vrange > old_vrange:
                message += f'cur {chan} {new_irange};'
            old_voltage = self.channels[chan-1].v.get()
            # Check if voltage is non-zero and mode_force is off
            if ((self.mode_force() is False) and
                    (abs(old_voltage) > max_zero_voltage[old_vrange])):
                raise ValueError(NON_ZERO_VOLTAGE_MSG)
            new_voltage = _clipto(
                    old_voltage, self.vranges[chan][new_vrange]['Min'],
                    self.vranges[chan][new_vrange]['Max'])
            message += f'vol {chan} {new_vrange};'
            message += wav_or_set_msg(chan, new_voltage)
            # Current sensor relay off->on after voltage relay on->off:
            if new_irange > old_irange and new_vrange < old_vrange:
                message += f';cur {chan} {new_irange}'
            self.channels[chan-1].v.vals = self._v_vals(chan, new_vrange)
            self.channels[chan-1].v.cache.set(new_voltage)

        self.write(message)

    def _v_vals(self, chan: int, vrange_int: int) -> vals.Numbers:
        """
        Returns the validator for the specified voltage range.
        """
        return vals.Numbers(self.vranges[chan][vrange_int]['Min'],
                            self.vranges[chan][vrange_int]['Max'])

    def _update_v_validators(self) -> None:
        """
        Command for setting all 'v' limits ('vals') of all channels to the
        actual calibrated output limits for the range each individual channel
        is currently in.
        """
        for chan in range(1, self.num_chans+1):
            vrange = self.channels[chan-1].mode.value.v
            self.channels[chan-1].v.vals = self._v_vals(chan, vrange)

    def _num_verbose(self, s: str) -> float:
        """
        Turns a return value from the QDac into a number.
        If the QDac is in verbose mode, this involves stripping off the
        value descriptor.
        """
        if self._verbose:
            s = s.split(': ')[-1]
        return float(s)

    def _current_parser(self, s: str) -> float:
        """
        Parser for chXX_i parameter (converts from uA to A)
        """
        return 1e-6*self._num_verbose(s)

    def _update_cache(self, update_currents: bool = False) -> None:
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

        # Status call, check the
        version_line = self.ask('status')
        if version_line.startswith('Software Version: '):
            self.version = version_line.strip().split(': ')[1]
        else:
            self._wait_and_clear()
            raise ValueError('unrecognized version line: ' + version_line)

        # Check header line
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
            mode = Mode((vrange_int, irange_int))
            self.channels[chan-1].mode.cache.set(mode)
            self.channels[chan-1].v.cache.set(float(v))
            self.channels[chan-1].v.vals = self._v_vals(chan, vrange_int)
            chans_left.remove(chan)

        if update_currents:
            for chan in self._chan_range:
                self.channels[chan-1].i.get()

    def _setsync(self, chan: int, sync: int) -> None:
        """
        set_cmd for the chXX_sync parameter.

        Args:
            chan: The channel number (1-48 or 1-24)
            sync: The associated sync output (1-3 on 24 ch units
            or 1-5 on 48 ch units). 0 means 'unassign'
        """

        if chan not in range(1, self.num_chans+1):
            raise ValueError(
                    f'Channel number must be 1-{self.num_chans}.')

        if sync == 0:
            oldsync = self.channels[chan-1].sync.cache()
            # try to remove the sync from internal bookkeeping
            self._syncoutputs.pop(chan, None)
            # free the previously assigned sync
            if oldsync is not None:
                self.write(f'syn {oldsync} 0 0 0')
            return

        # Make sure to clear hardware an _syncoutpus appropriately
        if chan in self._syncoutputs:
            # Changing SYNC port for a channel
            oldsync = self.channels[chan-1].sync.cache()
            if sync != oldsync:
                self.write(f'syn {oldsync} 0 0 0')
        elif sync in self._syncoutputs.values():
            # Assigning an already used SYNC port to a different channel
            oldchan = [ch for ch, sy in self._syncoutputs.items()
                       if sy == sync]
            self._syncoutputs.pop(oldchan[0], None)
            self.write(f'syn {sync} 0 0 0')

        self._syncoutputs[chan] = sync
        return

    def _getsync(self, chan: int) -> int:
        """
        get_cmd of the chXX_sync parameter
        """
        return self._syncoutputs.get(chan, 0)

    def print_syncs(self) -> None:
        """
        Print assigned SYNC ports, sorted by channel number
        """
        for chan, sync in sorted(self._syncoutputs.items()):
            print(f'Channel {chan}, SYNC: {sync} (V/s)')

    def _setslope(self, chan: int, slope: Union[float, str]) -> None:
        """
        set_cmd for the chXX_slope parameter, the maximum slope of a channel.
        With a finite slope the channel will be ramped using a generator.

        Args:
            chan: The channel number (1-24 or 1-48)
            slope: The slope in V/s.
            Write 'Inf' to release the channelas slope channel and to release
            the associated function generator. The output rise time will now
            only depend on the analog electronics.
        """
        if chan not in range(1, self.num_chans+1):
            raise ValueError(
                        f'Channel number must be 1-{self.num_chans}.')

        if slope == 'Inf':
            # Set the channel in DC mode
            v_set = self.channels[chan-1].v.get()
            self.write('set {ch} {voltage:.6f};wav {ch} 0 0 0'
                       .format(ch=chan, voltage=v_set))

            # Now release the function generator and fg trigger (if possible)
            try:
                fg = self._assigned_fgs[chan]
                self._assigned_fgs[chan].t_end = 0
                self._assigned_triggers.pop(fg.fg)
            except KeyError:
                pass

            # Remove a sync output, if one was assigned
            if chan in self._syncoutputs:
                self.channels[chan-1].sync.set(0)
            # Now clear the assigned slope
            self._slopes.pop(chan, None)
        else:
            self._slopes[chan] = slope

    def _getslope(self, chan: int) -> Union[str, float]:
        """
        get_cmd of the chXX_slope parameter
        """
        return self._slopes.get(chan, 'Inf')

    def print_slopes(self) -> None:
        """
        Print the finite slopes assigned to channels, sorted by channel number
        """
        for chan, slope in sorted(self._slopes.items()):
            print(f'Channel {chan}, slope: {slope} (V/s)')

    def _get_minmax_outputvoltage(self, channel: int, vrange_int: int) -> Dict[str, float]:
        """
        Returns a dictionary of the calibrated Min and Max output
        voltages of 'channel' for the voltage given range (0,1) given by
        'vrange_int'
        """
        # For firmware 1.07 verbose mode and nn verbose mode give verbose
        # result, So this is designed for verbose mode
        if channel not in range(1, self.num_chans+1):
            raise ValueError(
                        f'Channel number must be 1-{self.num_chans}.')
        if vrange_int not in range(0, 2):
            raise ValueError('Range must be 0 or 1.')

        self.write(f'rang {channel} {vrange_int}')
        fw_str = self._write_response
        return {'Min': float(fw_str.split('MIN:')[1].split('MAX')[0].strip()),
                'Max': float(fw_str.split('MAX:')[1].strip())}

    def _update_voltage_ranges(self) -> None:
        # Get all calibrated min/max output values, requires verbose on
        # in firmware version 1.07
        self.write('ver 1')
        self.vranges = {}
        for chan in self._chan_range:
            self.vranges.update(
                {chan: {0: self._get_minmax_outputvoltage(chan, 0),
                        1: self._get_minmax_outputvoltage(chan, 1)}})
        self.write('ver 0')

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

        LOG.debug(f"Writing to instrument {self.name}: {cmd}")
        self.visa_handle.write(cmd)
        for _ in range(cmd.count(';')+1):
            self._write_response = self.visa_handle.read()

    def read(self) -> str:
        return self.visa_handle.read()

    def _wait_and_clear(self, delay: float = 0.5) -> None:
        time.sleep(delay)
        self.visa_handle.clear()

    def connect_message(self,
                        idn_param: str = 'IDN',
                        begin_time: Optional[float] = None) -> None:
        """
        Override of the standard Instrument class connect_message.
        Usually, the response to `*IDN?` is printed. Here, the
        software version is printed.
        """
        self.visa_handle.write('version')
        LOG.info('Connected to QDAC on {}, {}'.format(
                                    self._address, self.visa_handle.read()))

    def _get_firmware_version(self) -> float:
        """
        Check if the "version" command reponds. If so we probbaly have a QDevil
        QDAC, and the version number is returned. Otherwise 0.0 is returned.
        """
        self.write('version')
        fw_str = self._write_response
        if ((not ("Unrecognized command" in fw_str))
                and ("Software Version: " in fw_str)):
            fw_version = float(
                self._write_response.replace("Software Version: ", ""))
        else:
            fw_version = 0.0
        return fw_version

    def _get_number_of_channels(self) -> int:
        """
        Returns the number of channels for the instrument
        """
        self.write('boardNum')
        fw_str = self._write_response
        return 8*int(fw_str.strip("numberOfBoards:"))

    def print_overview(self, update_currents: bool =  False) -> None:
        """
        Pretty-prints the status of the QDac
        """

        self._update_cache(update_currents=update_currents)

        for ii in range(self.num_chans):
            line = 'Channel {} \n'.format(ii+1)
            line += '    Voltage: {} ({}).\n'.format(
                    self.channels[ii].v.cache(), self.channels[ii].v.unit)
            line += '    Current: {} ({}).\n'.format(
                    self.channels[ii].i.cache.get(get_if_invalid=False),
                                                self.channels[ii].i.unit)
            line += '    Mode: {}.\n'.format(
                    self.channels[ii].mode.cache().get_label())
            line += '    Slope: {} ({}).\n'.format(
                    self.channels[ii].slope.cache(),
                    self.channels[ii].slope.unit)
            if self.channels[ii].sync.cache() > 0:
                line += '    Sync Out: {}, Delay: {} ({}), '\
                        'Duration: {} ({}).\n'.format(
                            self.channels[ii].sync.cache(),
                            self.channels[ii].sync_delay.cache(),
                            self.channels[ii].sync_delay.unit,
                            self.channels[ii].sync_duration.cache(),
                            self.channels[ii].sync_duration.unit,
                        )

            print(line)

    def _get_functiongenerator(self, chan: int) -> int:
        """
        Function for getting a free generator (of 8 available) for a channel.
        Used as helper function for ramp_voltages, but may also be used if the
        user wants to use a function generator for something else.
        If there are no free generators this function will wait for up to
        fgs_timeout for one to be ready.

        To mark a function generator as available for others set
        self._assigned_fgs[chan].t_end = 0

        Args:
            chan: (1..24/48) the channel for which a function generator is
                  requested.
        """
        fgs_timeout = 2  # Max time to wait for next available generator

        if len(self._assigned_fgs) < 8:
            fg = min(self._fgs.difference(
                        {g.fg for g in self._assigned_fgs.values()}))
            self._assigned_fgs[chan] = Generator(fg)
        else:
            # If no available fgs, see if one is soon to be ready
            # Nte, this does not handle if teh user has assigned the
            # same fg to multiple channels cheating the driver
            time_now = time.time()
            available_fgs_chans = []
            fgs_t_end_ok = [g.t_end for chan, g
                            in self._assigned_fgs.items()
                            if g.t_end < time_now+fgs_timeout]
            if len(fgs_t_end_ok) > 0:
                first_ready_t = min(fgs_t_end_ok)
                available_fgs_chans = [chan for chan, g
                                       in self._assigned_fgs.items()
                                       if g.t_end == first_ready_t]
                if first_ready_t > time_now:
                    LOG.warning('''
                    Trying to ramp more channels than there are generators.\n
                    Waiting for ramp generator to be released''')
                    time.sleep(first_ready_t - time_now)

            if len(available_fgs_chans) > 0:
                oldchan = available_fgs_chans[0]
                fg = self._assigned_fgs[oldchan].fg
                self._assigned_fgs.pop(oldchan)
                self._assigned_fgs[chan] = Generator(fg)
                # Set the old channel in DC mode
                v_set = self.channels[oldchan-1].v.cache()
                self.write('set {ch} {voltage:.6f};wav {ch} 0 0 0'
                           .format(ch=oldchan, voltage=v_set))
            else:
                raise RuntimeError('''
                Trying to ramp more channels than there are generators
                available. Please insert delays allowing channels to finish
                ramping before trying to ramp other channels, or reduce the
                number of ramped channels. Or increase fgs_timeout.''')
        return fg

    def ramp_voltages(
            self,
            channellist: Sequence[int],
            v_startlist: Sequence[float],
            v_endlist: Sequence[float],
            ramptime: float) -> float:
        """
        Function for smoothly ramping one channel or more channels
        simultaneously (max. 8). This is a shallow interface to
        ramp_voltages_2d. Function generators and triggers are
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
            LOG.warning('Ramp time too short: {:.3f} s. Ramp time set to 2 ms.'
                        .format(ramptime))
            ramptime = 0.002
        steps = int(ramptime*1000)
        return self.ramp_voltages_2d(
                            slow_chans=[], slow_vstart=[], slow_vend=[],
                            fast_chans=channellist, fast_vstart=v_startlist,
                            fast_vend=v_endlist, step_length=0.001,
                            slow_steps=1, fast_steps=steps)

    def ramp_voltages_2d(
            self,
            slow_chans: Sequence[int],
            slow_vstart: Sequence[float],
            slow_vend: Sequence[float],
            fast_chans: Sequence[int],
            fast_vstart: Sequence[float],
            fast_vend: Sequence[float],
            step_length: float,
            slow_steps: int,
            fast_steps: int) -> float:
        """
        Function for smoothly ramping two channel groups simultaneously with
        one slow (x) and one fast (y) group. used by 'ramp_voltages' where x is
        empty. Function generators and triggers are assigned automatically.

        Args:
            slow_chans:   List of channels to be ramped (1 indexed) in
                          the slow-group\n
            slow_vstart:  List of voltages to ramp from in the
                          slow-group.
                          MAY BE EMPTY. But if provided, time is saved by NOT
                          reading the present values from the instrument.\n
            slow_vend:    list of voltages to ramp to in the slow-group.

            fast_chans:   List of channels to be ramped (1 indexed) in
                          the fast-group.\n
            fast_vstart:  List of voltages to ramp from in the
                          fast-group.
                          MAY BE EMPTY. But if provided, time is saved by NOT
                          reading the present values from the instrument.\n
            fast_vend:    list of voltages to ramp to in the fast-group.

            step_length:  Time spent at each step in seconds
                          (min. 0.001) multiple of 1 ms.\n
            slow_steps:   number of steps in the slow direction.\n
            fast_steps:   number of steps in the fast direction.\n

        Returns:
            Estimated time of the excecution of the 2D scan.\n
        NOTE: This function returns as the ramps are started.
        """
        channellist = [*slow_chans, *fast_chans]
        v_endlist = [*slow_vend, *fast_vend]
        v_startlist = [*slow_vstart, *fast_vstart]
        step_length_ms = int(step_length*1000)

        if step_length < 0.001:
            LOG.warning('step_length too short: {:.3f} s. \nstep_length set to'
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
                        f'Channel number must be 1-{self.num_chans}.')
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
                sync_duration = int(
                                1000*self.channels[chan-1].sync_duration.get())
                sync_delay = int(1000*self.channels[chan-1].sync_delay.get())
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
            self.channels[ch-1].v.cache.set(v_endlist[i])
        self.write(msg[:-1])  # last semicolon is stripped

        # Fire trigger to start generators simultaneously, saving communication
        # time by not using triggers for single channel ramping
        if trigger > 0:
            self.write(f'trig {trigger}')

        # Update fgs dict so that we know when the ramp is supposed to end
        time_ramp = slow_steps * fast_steps * step_length_ms / 1000
        time_end = time_ramp + time.time()
        for chan in channellist:
            self._assigned_fgs[chan].t_end = time_end
        return time_ramp
