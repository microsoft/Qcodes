import logging
from functools import partial
from typing import Any, Union

import numpy as np

import qcodes.utils.validators as vals
from qcodes.instrument.parameter import ArrayParameter, ParamRawDataType
from qcodes.instrument.visa import VisaInstrument

log = logging.getLogger(__name__)

_unit_map = {'Log mag': 'dB',
             'Phase': 'degree',
             'Delay': "",
             'Smith chart': 'dim. less',
             'Polar': 'dim. less',
             'Lin mag': 'dim. less',
             'Real': "",
             'Imaginary': "",
             'SWR': 'dim. less'}


def HPIntParser(value: str) -> int:
    """
    Small custom parser for ints

    Args:
        value: the VISA return string using exponential notation
    """
    return int(float(value))


class TraceNotReady(Exception):
    pass


class HP8753DTrace(ArrayParameter):
    """
    Class to hold a the trace from the HP8753D

    Although the trace can have two values per frequency, this
    class only returns the first value
    """

    def __init__(
            self,
            name: str,
            instrument: "HP8753D"
    ):
        super().__init__(name=name,
                         shape=(1,),  # is overwritten by prepare_trace
                         label='',  # is overwritten by prepare_trace
                         unit='',  # is overwritten by prepare_trace
                         setpoint_names=('Frequency',),
                         setpoint_labels=('Frequency',),
                         setpoint_units=('Hz',),
                         snapshot_get=False,
                         instrument=instrument
                         )

    def prepare_trace(self) -> None:
        """
        Update setpoints, units and labels
        """

        # we don't trust users to keep their fingers off the front panel,
        # so we query the instrument for all values
        assert isinstance(self.instrument, HP8753D)
        fstart = self.instrument.start_freq()
        fstop = self.instrument.stop_freq()
        npts = self.instrument.trace_points()

        sps = np.linspace(fstart, fstop, npts)
        self.setpoints = (tuple(sps),)
        self.shape = (len(sps),)

        self.label = self.instrument.s_parameter()
        self.unit = _unit_map[self.instrument.display_format()]

        self.instrument._traceready = True

    def get_raw(self) -> ParamRawDataType:
        """
        Return the trace
        """

        inst = self.instrument
        assert isinstance(inst, HP8753D)
        if not inst._traceready:
            raise TraceNotReady('Trace not ready. Please run prepare_trace.')

        inst.write('FORM2')  # 32-bit floating point numbers
        inst.write('OUTPFORM')
        inst.visa_handle.read_termination = ''
        raw_resp = inst.visa_handle.read_raw()
        inst.visa_handle.read_termination = '\n'

        first_points = b''
        # 4 bytes header, 4 bytes per point, value1's and value2's
        # are intertwined like: val1_001, val2_001, val1_002, val2_002...
        for n in range((len(raw_resp)-4)//4):
            first_points += raw_resp[4:][2*n*4:(2*n+1)*4]

        dt = np.dtype('>f')
        trace1 = np.fromstring(first_points, dtype=dt)

        return trace1


class HP8753D(VisaInstrument):
    """
    This is the QCoDeS driver for the Hewlett Packard 8753D Network Analyzer
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter(
            'start_freq',
            label='Sweep start frequency',
            unit='Hz',
            set_cmd=partial(self.invalidate_trace, 'STAR {} HZ'),
            get_cmd='STAR?',
            get_parser=float,
            vals=vals.Numbers(30000, 6000000000))

        self.add_parameter(
            'stop_freq',
            label='Sweep stop frequency',
            unit='Hz',
            set_cmd=partial(self.invalidate_trace, 'STOP {} HZ'),
            get_cmd='STOP?',
            get_parser=float,
            vals=vals.Numbers(30000, 6000000000))

        self.add_parameter(
            'averaging',
            label='Averaging state',
            set_cmd='AVERO{}',
            get_cmd='AVERO?',
            val_mapping={'ON': 1, 'OFF': 0})

        self.add_parameter(
            'number_of_averages',
            label='Number of averages',
            set_cmd='AVERFACT{}',
            get_cmd='AVERFACT?',
            get_parser=HPIntParser,
            vals=vals.Ints(0, 999))

        self.add_parameter(
            'trace_points',
            label='Number of points in trace',
            set_cmd=partial(self.invalidate_trace, 'POIN{}'),
            get_cmd='POIN?',
            get_parser=HPIntParser,
            vals=vals.Enum(3, 11, 26, 51, 101, 201, 401,
                           801, 1601))

        self.add_parameter(
            'sweep_time',
            label='Sweep time',
            set_cmd='SWET{}',
            get_cmd='SWET?',
            unit='s',
            get_parser=float,
            vals=vals.Numbers(0.01, 86400),
            )

        self.add_parameter(
            'output_power',
            label='Output power',
            unit='dBm',
            set_cmd='POWE{}',
            get_cmd='POWE?',
            get_parser=float,
            vals=vals.Numbers(-85, 20))

        self.add_parameter(
            's_parameter',
            label='S-parameter',
            set_cmd=self._s_parameter_setter,
            get_cmd=self._s_parameter_getter)

        # DISPLAY / Y SCALE PARAMETERS
        self.add_parameter(
            'display_format',
            label='Display format',
            set_cmd=self._display_format_setter,
            get_cmd=self._display_format_getter)

        # TODO: update this on startup and via display format
        self.add_parameter(
            'display_reference',
            label='Display reference level',
            unit=None,  # will be set by display_format
            get_cmd='REFV?',
            set_cmd='REFV{}',
            get_parser=float,
            vals=vals.Numbers(-500, 500))

        self.add_parameter(
            'display_scale',
            label='Display scale',
            unit=None,  # will be set by display_format
            get_cmd='SCAL?',
            set_cmd='SCAL{}',
            get_parser=float,
            vals=vals.Numbers(-500, 500))

        self.add_parameter(
            name='trace',
            parameter_class=HP8753DTrace)

        # Startup
        self.startup()
        self.connect_message()

    def reset(self) -> None:
        """
        Resets the instrument to factory default state
        """
        # use OPC to make sure we wait for operation to finish
        self.ask('OPC?;PRES')

    def run_continously(self) -> None:
        """
        Set the instrument in run continously mode
        """
        self.write('CONT')

    def run_N_times(self, N: int) -> None:
        """
        Run N sweeps and then hold. We wait for a response before returning
        """

        st = self.sweep_time.get_latest()

        if N not in range(1, 1000):
            raise ValueError(f'Can not run {N} times.' +
                             ' please select a number from 1-999.')

        # set a longer timeout, to not timeout during the sweep
        new_timeout = st*N + 2

        with self.timeout.set_to(new_timeout):
            log.debug(f'Making {N} blocking sweeps.' +
                      f' Setting VISA timeout to {new_timeout} s.')

            self.ask(f'OPC?;NUMG{N}')

    def invalidate_trace(self, cmd: str,
                         value: Union[float, int, str]) -> None:
        """
        Wrapper for set_cmds that make the trace not ready
        """
        self._traceready = False
        self.write(cmd.format(value))

    def startup(self) -> None:
        self._traceready = False
        self.display_format(self.display_format())

    def _s_parameter_setter(self, param: str) -> None:
        """
        set_cmd for the s_parameter parameter
        """
        if param not in ['S11', 'S12', 'S21', 'S22']:
            raise ValueError('Cannot set s-parameter to {}')

        # the trace labels changes
        self._traceready = False

        self.write(param)

    def _s_parameter_getter(self) -> str:
        """
        get_cmd for the s_parameter parameter
        """
        for cmd in ['S11?', 'S12?', 'S21?', 'S22?']:
            resp = self.ask(cmd)
            if resp in ['1', '1\n']:
                break

        return cmd.replace('?', '')

    def _display_format_setter(self, fmt: str) -> None:
        """
        set_cmd for the display_format parameter
        """
        val_mapping = {'Log mag': 'LOGM',
                       'Phase': 'PHAS',
                       'Delay': 'DELA',
                       'Smith chart': 'SMIC',
                       'Polar': 'POLA',
                       'Lin mag': 'LINM',
                       'Real': 'REAL',
                       'Imaginary': 'IMAG',
                       'SWR': 'SWR'}

        if fmt not in val_mapping.keys():
            raise ValueError(f'Cannot set display_format to {fmt}.')

        self._traceready = False
        self.display_reference.unit = _unit_map[fmt]
        self.display_scale.unit = _unit_map[fmt]

        self.write(val_mapping[fmt])

    def _display_format_getter(self) -> str:
        """
        get_cmd for the display_format parameter
        """
        val_mapping = {'LOGM': 'Log mag',
                       'PHAS': 'Phase',
                       'DELA': 'Delay',
                       'SMIC': 'Smith chart',
                       'POLA': 'Polar',
                       'LINM': 'Lin mag',
                       'REAL': 'Real',
                       'IMAG': 'Imaginary',
                       'SWR': 'SWR'}

        # keep asking until we find the currently used format
        for cmd in val_mapping.keys():
            resp = self.ask(f'{cmd}?')
            if resp in ['1', '1\n']:
                break

        return val_mapping[cmd]
