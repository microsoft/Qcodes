import logging
from functools import partial
from typing import Optional

from qcodes import VisaInstrument, Instrument
from qcodes import ChannelList, InstrumentChannel
from qcodes.utils import validators as vals
import numpy as np
from qcodes import MultiParameter, ArrayParameter

log = logging.getLogger(__name__)


class FrequencySweepMagPhase(MultiParameter):
    """
    Sweep that return magnitude and phase.
    """

    def __init__(self, name: str, instrument: Instrument,
                 start: float, stop: float, npts: int, channel: int) -> None:
        super().__init__(name, names=("", ""), shapes=((), ()))
        self._instrument = instrument
        self.set_sweep(start, stop, npts)
        self._channel = channel
        self.names = ('magnitude',
                      'phase')
        self.labels = (f'{instrument.short_name} magnitude',
                       f'{instrument.short_name} phase')
        self.units = ('', 'rad')
        self.setpoint_units = (('Hz',), ('Hz',))
        self.setpoint_labels = (
            (f'{instrument.short_name} frequency',),
            (f'{instrument.short_name} frequency',)
        )
        self.setpoint_names = (
            (f'{instrument.short_name}_frequency',),
            (f'{instrument.short_name}_frequency',)
        )

    def set_sweep(self, start: float, stop: float, npts: int) -> None:
        # Needed to update config of the software parameter on sweep change
        # frequency setpoints tuple as needs to be hashable for look up.
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = ((f,), (f,))
        self.shapes = ((npts,), (npts,))

    def get_raw(self):
        old_format = self._instrument.format()
        self._instrument.format('Complex')
        data = self._instrument._get_sweep_data(force_polar=True)
        self._instrument.format(old_format)
        return abs(data), np.angle(data)


class FrequencySweep(ArrayParameter):
    """
    Hardware controlled parameter class for Rohde Schwarz ZNB trace.

    Instrument returns an array of transmission or reflection data depending
    on the active measurement.

    Args:
        name: parameter name
        instrument: instrument the parameter belongs to
        start: starting frequency of sweep
        stop: ending frequency of sweep
        npts: number of points in frequency sweep

    Methods:
          set_sweep(start, stop, npts): sets the shapes and
              setpoint arrays of the parameter to correspond with the sweep
          get(): executes a sweep and returns magnitude and phase arrays

    """

    def __init__(self, name: str, instrument: Instrument,
                 start: float, stop: float, npts: int, channel: int) -> None:
        super().__init__(name, shape=(npts,),
                         instrument=instrument,
                         unit='dB',
                         label=f'{instrument.short_name} magnitude',
                         setpoint_units=('Hz',),
                         setpoint_labels=(f'{instrument.short_name}'
                                          ' frequency',),
                         setpoint_names=(f'{instrument.short_name}_frequency',))
        self.set_sweep(start, stop, npts)
        self._channel = channel

    def set_sweep(self, start: float, stop: float, npts: int) -> None:
        # Needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up.
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = (f,)
        self.shape = (npts,)

    def get_raw(self):
        data = self._instrument._get_sweep_data()
        if self._instrument.format() in ['Polar', 'Complex',
                                         'Smith', 'Inverse Smith']:
            log.warning("QCoDeS Dataset does not currently support Complex "
                        "values. Will discard the imaginary part. In order to "
                        "acquire phase and amplitude use the "
                        "FrequencySweepMagPhase parameter.")
        return data


class ZNBChannel(InstrumentChannel):

    def __init__(self, parent: 'ZNB', name: str, channel: int,
                 vna_parameter: Optional[str] = None,
                 existing_trace_to_bind_to: Optional[str] = None) -> None:
        """
        Args:
            parent: Instrument that this channel is bound to.
            name: Name to use for this channel.
            channel: channel on the VNA to use
            vna_parameter: Name of parameter on the vna that this should
                measure such as S12. If left empty this will fall back to
                `name`.
            existing_trace_to_bind_to: Name of an existing trace on the VNA.
                If supplied try to bind to an existing trace with this name
                rather than creating a new trace.

        """
        n = channel
        self._instrument_channel = channel
        # Additional wait when adjusting instrument timeout to sweep time.
        self._additional_wait = 1

        if vna_parameter is None:
            vna_parameter = name
        self._vna_parameter = vna_parameter
        super().__init__(parent, name)

        if existing_trace_to_bind_to is None:
            self._tracename = f"Trc{channel}"
        else:
            traces = self._parent.ask(f"CONFigure:TRACe:CATalog?")
            if existing_trace_to_bind_to not in traces:
                raise RuntimeError(f"Trying to bind to"
                                   f" {existing_trace_to_bind_to} "
                                   f"which is not in {traces}")
            self._tracename = existing_trace_to_bind_to

        # map hardware channel to measurement
        # hardware channels are mapped one to one to QCoDeS channels
        # we are not using sub traces within channels.
        if existing_trace_to_bind_to is None:
            self.write(f"CALC{self._instrument_channel}:PAR:SDEF"
                       f" '{self._tracename}', '{self._vna_parameter}'")

        # Source power is dependent on model, but not well documented.
        # Here we assume -60 dBm for ZNB20, the others are set,
        # due to lack of knowledge, to -80 dBm as of before the edit.
        full_modelname = self._parent.get_idn()['model']
        if full_modelname is not None:
            model = full_modelname.split('-')[0]
        else:
            raise RuntimeError("Could not determine ZNB model")
        mSourcePower = {'ZNB4': -80, 'ZNB8': -80, 'ZNB20': -60}
        if model not in mSourcePower.keys():
            raise RuntimeError(f"Unsupported ZNB model: {model}")
        self._min_source_power: float
        self._min_source_power = mSourcePower[model]

        self.add_parameter(name='vna_parameter',
                           label='VNA parameter',
                           get_cmd=f"CALC{self._instrument_channel}:"
                                   f"PAR:MEAS? '{self._tracename}'",
                           get_parser=self._strip)
        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd=f'SOUR{n}:POW?',
                           set_cmd=f'SOUR{n}:POW {{:.4f}}',
                           get_parser=float,
                           vals=vals.Numbers(self._min_source_power, 25))
        # there is an 'increased bandwidth option' (p. 4 of manual) that does
        # not get taken into account here
        self.add_parameter(name='bandwidth',
                           label='Bandwidth',
                           unit='Hz',
                           get_cmd=f'SENS{n}:BAND?',
                           set_cmd=f'SENS{n}:BAND {{:.4f}}',
                           get_parser=int,
                           vals=vals.Enum(
                               *np.append(10 ** 6,
                                          np.kron([1, 1.5, 2, 3, 5, 7],
                                                  10 ** np.arange(6))))
                           )
        self.add_parameter(name='avg',
                           label='Averages',
                           unit='',
                           get_cmd=f'SENS{n}:AVER:COUN?',
                           set_cmd=f'SENS{n}:AVER:COUN {{:.4f}}',
                           get_parser=int,
                           vals=vals.Ints(1, 5000))
        self.add_parameter(name='start',
                           get_cmd=f'SENS{n}:FREQ:START?',
                           set_cmd=self._set_start,
                           get_parser=float,
                           vals=vals.Numbers(self._parent._min_freq,
                                             self._parent._max_freq - 10))
        self.add_parameter(name='stop',
                           get_cmd=f'SENS{n}:FREQ:STOP?',
                           set_cmd=self._set_stop,
                           get_parser=float,
                           vals=vals.Numbers(self._parent._min_freq + 1,
                                             self._parent._max_freq))
        self.add_parameter(name='center',
                           get_cmd=f'SENS{n}:FREQ:CENT?',
                           set_cmd=self._set_center,
                           get_parser=float,
                           vals=vals.Numbers(self._parent._min_freq + 0.5,
                                             self._parent._max_freq - 10))
        self.add_parameter(name='span',
                           get_cmd=f'SENS{n}:FREQ:SPAN?',
                           set_cmd=self._set_span,
                           get_parser=float,
                           vals=vals.Numbers(1, self._parent._max_freq -
                                             self._parent._min_freq))
        self.add_parameter(name='npts',
                           get_cmd=f'SENS{n}:SWE:POIN?',
                           set_cmd=self._set_npts,
                           get_parser=int)
        self.add_parameter(name='status',
                           get_cmd=f'CONF:CHAN{n}:MEAS?',
                           set_cmd=f'CONF:CHAN{n}:MEAS {{}}',
                           get_parser=int)
        self.add_parameter(name='format',
                           get_cmd=partial(self._get_format,
                                           tracename=self._tracename),
                           set_cmd=self._set_format,
                           val_mapping={'dB': 'MLOG\n',
                                        'Linear Magnitude': 'MLIN\n',
                                        'Phase': 'PHAS\n',
                                        'Unwr Phase': 'UPH\n',
                                        'Polar': 'POL\n',
                                        'Smith': 'SMIT\n',
                                        'Inverse Smith': 'ISM\n',
                                        'SWR': 'SWR\n',
                                        'Real': 'REAL\n',
                                        'Imaginary': 'IMAG\n',
                                        'Delay': "GDEL\n",
                                        'Complex': "COMP\n"
                                        })

        self.add_parameter(name='trace_mag_phase',
                           start=self.start(),
                           stop=self.stop(),
                           npts=self.npts(),
                           channel=n,
                           parameter_class=FrequencySweepMagPhase)
        self.add_parameter(name='trace',
                           start=self.start(),
                           stop=self.stop(),
                           npts=self.npts(),
                           channel=n,
                           parameter_class=FrequencySweep)
        self.add_parameter(name='electrical_delay',
                           label='Electrical delay',
                           get_cmd=f'SENS{n}:CORR:EDEL2:TIME?',
                           set_cmd=f'SENS{n}:CORR:EDEL2:TIME {{}}',
                           get_parser=float,
                           unit='s')
        self.add_parameter(name='sweep_time',
                           label='Sweep time',
                           get_cmd=f'SENS{n}:SWE:TIME?',
                           get_parser=float,
                           unit='s')
        self.add_function('set_electrical_delay_auto',
                          call_cmd=f'SENS{n}:CORR:EDEL:AUTO ONCE')

        self.add_function('autoscale',
                          call_cmd='DISPlay:TRACe1:Y:SCALe:AUTO ONCE, '
                                   f"{self._tracename}")

    def _get_format(self, tracename: str) -> str:
        n = self._instrument_channel
        self.write(f"CALC{n}:PAR:SEL '{tracename}'")
        return self.ask(f"CALC{n}:FORM?")

    def _set_format(self, val: str) -> None:
        unit_mapping = {'MLOG\n': 'dB',
                        'MLIN\n': '',
                        'PHAS\n': 'rad',
                        'UPH\n': 'rad',
                        'POL\n': '',
                        'SMIT\n': '',
                        'ISM\n': '',
                        'SWR\n': 'U',
                        'REAL\n': 'U',
                        'IMAG\n': 'U',
                        'GDEL\n': 'S',
                        'COMP\n': ''}
        label_mapping = {'MLOG\n': 'Magnitude',
                         'MLIN\n': 'Magnitude',
                         'PHAS\n': 'Phase',
                         'UPH\n': 'Unwrapped phase',
                         'POL\n': 'Complex Magnitude',
                         'SMIT\n': 'Complex Magnitude',
                         'ISM\n': 'Complex Magnitude',
                         'SWR\n': 'Standing Wave Ratio',
                         'REAL\n': 'Real Magnitude',
                         'IMAG\n': 'Imaginary Magnitude',
                         'GDEL\n': 'Delay',
                         'COMP\n': 'Complex Magnitude'}
        channel = self._instrument_channel
        self.write(f"CALC{channel}:PAR:SEL '{self._tracename}'")
        self.write(f"CALC{channel}:FORM {val}")
        self.trace.unit = unit_mapping[val]
        self.trace.label = f"{self.short_name} {label_mapping[val]}"

    @staticmethod
    def _strip(var: str) -> str:
        """Strip newline and quotes from instrument reply."""
        return var.rstrip()[1:-1]

    def _set_start(self, val: float):
        channel = self._instrument_channel
        self.write(f'SENS{channel}:FREQ:START {val:.7f}')
        stop = self.stop()
        if val >= stop:
            raise ValueError(
                "Stop frequency must be larger than start frequency.")
        # we get start as the vna may not be able to set it to the
        # exact value provided.
        start = self.start()
        if val != start:
            log.warning(
                f"Could not set start to {val} setting it to {start}")
        self.update_traces()

    def _set_stop(self, val: float):
        channel = self._instrument_channel
        start = self.start()
        if val <= start:
            raise ValueError(
                "Stop frequency must be larger than start frequency.")
        self.write(f'SENS{channel}:FREQ:STOP {val:.7f}')
        # We get stop as the vna may not be able to set it to the
        # exact value provided.
        stop = self.stop()
        if val != stop:
            log.warning(
                f"Could not set stop to {val} setting it to {stop}")
        self.update_traces()

    def _set_npts(self, val: int):
        channel = self._instrument_channel
        self.write(f'SENS{channel}:SWE:POIN {val:.7f}')
        self.update_traces()

    def _set_span(self, val: float):
        channel = self._instrument_channel
        self.write(f'SENS{channel}:FREQ:SPAN {val:.7f}')
        self.update_traces()

    def _set_center(self, val: float):
        channel = self._instrument_channel
        self.write(f'SENS{channel}:FREQ:CENT {val:.7f}')
        self.update_traces()

    def update_traces(self):
        """ updates start, stop and npts of all trace parameters"""
        start = self.start()
        stop = self.stop()
        npts = self.npts()
        for _, parameter in self.parameters.items():
            if isinstance(parameter, (ArrayParameter, MultiParameter)):
                try:
                    parameter.set_sweep(start, stop, npts)
                except AttributeError:
                    pass

    def _get_sweep_data(self, force_polar: bool = False):

        if not self._parent.rf_power():
            log.warning("RF output is off when getting sweep data")
        # It is possible that the instrument and QCoDeS disagree about
        # which parameter is measured on this channel.
        instrument_parameter = self.vna_parameter()
        if instrument_parameter != self._vna_parameter:
            raise RuntimeError("Invalid parameter. Tried to measure "
                               f"{self._vna_parameter} "
                               f"got {instrument_parameter}")
        self.write(f'SENS{self._instrument_channel}:AVER:STAT ON')
        self.write(f'SENS{self._instrument_channel}:AVER:CLE')

        # preserve original state of the znb
        with self.status.set_to(1):
            self.root_instrument.cont_meas_off()
            try:
                # if force polar is set, the SDAT data format will be used. Here
                # the data will be transferred as a complex number independent
                # of the set format in the instrument.
                if force_polar:
                    data_format_command = 'SDAT'
                else:
                    data_format_command = 'FDAT'
                timeout = self.sweep_time() + self._additional_wait
                with self.root_instrument.timeout.set_to(timeout):
                    # instrument averages over its last 'avg' number of sweeps
                    # need to ensure averaged result is returned
                    for avgcount in range(self.avg()):
                        self.write(f'INIT{self._instrument_channel}:IMM; *WAI')
                    self.write(f"CALC{self._instrument_channel}:PAR:SEL "
                               f"'{self._tracename}'")
                    data_str = self.ask(
                        f'CALC{self._instrument_channel}:DATA?'
                        f' {data_format_command}')
                data = np.array(data_str.rstrip().split(',')).astype('float64')
                if self.format() in ['Polar', 'Complex',
                                     'Smith', 'Inverse Smith']:
                    data = data[0::2] + 1j * data[1::2]
            finally:
                self.root_instrument.cont_meas_on()
        return data


class ZNB(VisaInstrument):
    """
    QCoDeS driver for the Rohde & Schwarz ZNB8 and ZNB20
    virtual network analyser. It can probably be extended to ZNB4 and 40
    without too much work.

    Requires FrequencySweep parameter for taking a trace

    Args:
        name: instrument name
        address: Address of instrument probably in format
            'TCPIP0::192.168.15.100::inst0::INSTR'
        init_s_params: Automatically setup channels for all S parameters on the
            VNA.
        reset_channels: If True any channels defined on the VNA at the time
            of initialization are reset and removed.
        **kwargs: passed to base class

    TODO:
    - check initialisation settings and test functions
    """

    CHANNEL_CLASS = ZNBChannel

    def __init__(self, name: str, address: str, init_s_params: bool = True,
                 reset_channels: bool = True, **kwargs) -> None:

        super().__init__(name=name, address=address, **kwargs)

        # TODO(JHN) I could not find a way to get max and min freq from
        # the API, if that is possible replace below with that
        # See page 1025 in the manual. 7.3.15.10 for details of max/min freq
        # no attempt to support ZNB40, not clear without one how the format
        # is due to variants
        fullmodel = self.get_idn()['model']
        if fullmodel is not None:
            model = fullmodel.split('-')[0]
        else:
            raise RuntimeError("Could not determine ZNB model")
        # format seems to be ZNB8-4Port
        m_frequency = {'ZNB4': (9e3, 4.5e9), 'ZNB8': (9e3, 8.5e9),
                       'ZNB20': (100e3, 20e9)}
        if model not in m_frequency.keys():
            raise RuntimeError(f"Unsupported ZNB model {model}")
        self._min_freq: float
        self._max_freq: float
        self._min_freq, self._max_freq = m_frequency[model]

        self.add_parameter(name='num_ports',
                           get_cmd='INST:PORT:COUN?',
                           get_parser=int)
        num_ports = self.num_ports()

        self.add_parameter(name='rf_power',
                           get_cmd='OUTP1?',
                           set_cmd='OUTP1 {}',
                           val_mapping={True: '1\n', False: '0\n'})
        self.add_function('reset', call_cmd='*RST')
        self.add_function('tooltip_on', call_cmd='SYST:ERR:DISP ON')
        self.add_function('tooltip_off', call_cmd='SYST:ERR:DISP OFF')
        self.add_function('cont_meas_on', call_cmd='INIT:CONT:ALL ON')
        self.add_function('cont_meas_off', call_cmd='INIT:CONT:ALL OFF')
        self.add_function('update_display_once', call_cmd='SYST:DISP:UPD ONCE')
        self.add_function('update_display_on', call_cmd='SYST:DISP:UPD ON')
        self.add_function('update_display_off', call_cmd='SYST:DISP:UPD OFF')
        self.add_function('display_sij_split',
                          call_cmd=f'DISP:LAY GRID;:DISP:LAY:GRID'
                                   f' {num_ports},{num_ports}')
        self.add_function('display_single_window',
                          call_cmd='DISP:LAY GRID;:DISP:LAY:GRID 1,1')
        self.add_function('display_dual_window',
                          call_cmd='DISP:LAY GRID;:DISP:LAY:GRID 2,1')
        self.add_function('rf_off', call_cmd='OUTP1 OFF')
        self.add_function('rf_on', call_cmd='OUTP1 ON')
        if reset_channels:
            self.reset()
            self.clear_channels()
        channels = ChannelList(self, "VNAChannels", self.CHANNEL_CLASS,
                               snapshotable=True)
        self.add_submodule("channels", channels)
        if init_s_params:
            for i in range(1, num_ports + 1):
                for j in range(1, num_ports + 1):
                    ch_name = 'S' + str(i) + str(j)
                    self.add_channel(ch_name)
            self.channels.lock()
            self.display_sij_split()
            self.channels.autoscale()

        self.update_display_on()
        if reset_channels:
            self.rf_off()
        self.connect_message()

    def display_grid(self, rows: int, cols: int):
        """
        Display a grid of channels rows by columns.
        """
        self.write(f'DISP:LAY GRID;:DISP:LAY:GRID {rows},{cols}')

    def add_channel(self, channel_name: str, **kwargs):
        i_channel = len(self.channels) + 1
        channel = self.CHANNEL_CLASS(self, channel_name, i_channel, **kwargs)
        self.channels.append(channel)
        if i_channel == 1:
            self.display_single_window()
        if i_channel == 2:
            self.display_dual_window()
        # shortcut
        setattr(self, channel_name, channel)
        # initialising channel
        self.write(f'SENS{i_channel}:SWE:TYPE LIN')
        self.write(f'SENS{i_channel}:SWE:TIME:AUTO ON')
        self.write(f'TRIG{i_channel}:SEQ:SOUR IMM')
        self.write(f'SENS{i_channel}:AVER:STAT ON')

    def clear_channels(self):
        """
        Remove all channels from the instrument and channel list and
        unlock the channel list.
        """
        self.write('CALCulate:PARameter:DELete:ALL')
        for submodule in self.submodules.values():
            if isinstance(submodule, ChannelList):
                submodule._channels = []
                submodule._channel_mapping = {}
                submodule._locked = False
