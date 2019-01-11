import time
from functools import partial
from typing import Dict, Union, Optional, Callable, List, cast
import logging
from distutils.version import LooseVersion

import numpy as np

from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.visa import VisaInstrument
from qcodes.math.field_vector import FieldVector

log = logging.getLogger(__name__)
visalog = logging.getLogger('qcodes.instrument.visa')


def _response_preparser(bare_resp: str) -> str:
    """
    Pre-parse response from the instrument
    """
    return bare_resp.replace(':', '')


def _signal_parser(our_scaling: float, response: str) -> float:
    """
    Parse a response string into a correct SI value.

    Args:
        our_scaling: Whatever scale we might need to apply to get from
            e.g. A/min to A/s.
        response: What comes back from instrument.ask
    """

    # there might be a scale before the unit. We only want to deal in SI
    # units, so we translate the scale
    scale_to_factor = {'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
                       'k': 1e3, 'M': 1e6}

    numchars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-']

    response = _response_preparser(response)
    digits = ''.join([d for d in response if d in numchars])
    scale_and_unit = response[len(digits):]
    if scale_and_unit == '':
        their_scaling: float = 1
    elif scale_and_unit[0] in scale_to_factor.keys():
        their_scaling = scale_to_factor[scale_and_unit[0]]
    else:
        their_scaling = 1

    return float(digits)*their_scaling*our_scaling


class MercurySlavePS(InstrumentChannel):
    """
    Class to hold a slave power supply for the MercuryiPS
    """

    def __init__(self, parent: VisaInstrument, name: str, UID: str) -> None:
        """
        Args:
            parent: The Instrument instance of the MercuryiPS
            name: The 'colloquial' name of the PS
            UID: The UID as used internally by the MercuryiPS, e.g.
                'GRPX'
        """
        if ':' in UID:
            raise ValueError('Invalid UID. Must be axis group name or device '
                             'name, e.g. "GRPX" or "PSU.M1"')

        super().__init__(parent, name)
        self.uid = UID

        # The firmware update from 2.5 -> 2.6 changed the command
        # syntax slightly
        if LooseVersion(self.root_instrument.firmware) >= LooseVersion('2.6'):
            self.psu_string = "SPSU"
        else:
            self.psu_string = "PSU"

        self.add_parameter('voltage',
                           label='Output voltage',
                           get_cmd=partial(self._param_getter, 'SIG:VOLT'),
                           unit='V',
                           get_parser=partial(_signal_parser, 1))

        self.add_parameter('current',
                           label='Output current',
                           get_cmd=partial(self._param_getter, 'SIG:CURR'),
                           unit='A',
                           get_parser=partial(_signal_parser, 1))

        self.add_parameter('current_persistent',
                           label='Output persistent current',
                           get_cmd=partial(self._param_getter, 'SIG:PCUR'),
                           unit='A',
                           get_parser=partial(_signal_parser, 1))

        self.add_parameter('current_target',
                           label='Target current',
                           get_cmd=partial(self._param_getter, 'SIG:CSET'),
                           unit='A',
                           get_parser=partial(_signal_parser, 1))

        self.add_parameter('field_target',
                           label='Target field',
                           get_cmd=partial(self._param_getter, 'SIG:FSET'),
                           set_cmd=partial(self._param_setter, 'SIG:FSET'),
                           unit='T',
                           get_parser=partial(_signal_parser, 1))

        # NB: The current ramp rate slavishly follows the field ramp rate
        # (converted via the ATOB param)
        self.add_parameter('current_ramp_rate',
                           label='Ramp rate (current)',
                           unit='A/s',
                           get_cmd=partial(self._param_getter, 'SIG:RCST'),
                           get_parser=partial(_signal_parser, 1/60))

        self.add_parameter('field_ramp_rate',
                           label='Ramp rate (field)',
                           unit='T/s',
                           set_cmd=partial(self._param_setter, 'SIG:RFST'),
                           get_cmd=partial(self._param_getter, 'SIG:RFST'),
                           get_parser=partial(_signal_parser, 1/60),
                           set_parser=lambda x: x*60)

        self.add_parameter('field',
                           label='Field strength',
                           unit='T',
                           get_cmd=partial(self._param_getter, 'SIG:FLD'),
                           get_parser=partial(_signal_parser, 1))

        self.add_parameter('field_persistent',
                           label='Persistent field strength',
                           unit='T',
                           get_cmd=partial(self._param_getter, 'SIG:PFLD'),
                           get_parser=partial(_signal_parser, 1))

        self.add_parameter('ATOB',
                           label='Current to field ratio',
                           unit='A/T',
                           get_cmd=partial(self._param_getter, 'ATOB'),
                           get_parser=partial(_signal_parser, 1),
                           set_cmd=partial(self._param_setter, 'ATOB'))

        self.add_parameter('ramp_status',
                           label='Ramp status',
                           get_cmd=partial(self._param_getter, 'ACTN'),
                           set_cmd=self._ramp_status_setter,
                           get_parser=_response_preparser,
                           val_mapping={'HOLD': 'HOLD',
                                        'TO SET': 'RTOS',
                                        'CLAMP': 'CLMP',
                                        'TO ZERO': 'RTOZ'})

    def ramp_to_target(self) -> None:
        """
        Unconditionally ramp this PS to its target
        """
        status = self.ramp_status()
        if status == 'CLAMP':
            self.ramp_status('HOLD')
        self.ramp_status('TO SET')

    def _ramp_status_setter(self, cmd: str) -> None:
        status_now = self.ramp_status()
        if status_now == 'CLAMP' and cmd == 'RTOS':
            raise ValueError(f'Error in ramping unit {self.uid}: '
                             'Can not ramp to target value; power supply is '
                             'clamped. Unclamp first by setting ramp status '
                             'to HOLD.')
        else:
            partial(self._param_setter, 'ACTN')(cmd)

    def _param_getter(self, get_cmd: str) -> str:
        """
        General getter function for parameters

        Args:
            get_cmd: raw string for the command, e.g. 'SIG:VOLT'

        Returns:
            The response. Cf. MercuryiPS.ask for how much is returned
        """
        dressed_cmd = f"READ:DEV:{self.uid}:{self.psu_string}:{get_cmd}"

        resp = self._parent.ask(dressed_cmd)

        return resp

    def _param_setter(self, set_cmd: str, value: Union[float, str]) -> None:
        """
        General setter function for parameters

        Args:
            set_cmd: raw string for the command, e.g. 'SIG:FSET'
        """
        dressed_cmd = f"SET:DEV:{self.uid}:{self.psu_string}:{set_cmd}:{value}"
        # the instrument always very verbosely responds
        # the return value of `ask`
        # holds the value reported back by the instrument
        self._parent.ask(dressed_cmd)

        # TODO: we could use the opportunity to check that we did set/achieve
        #  the intended value


class MercuryiPS(VisaInstrument):
    """
    Driver class for the QCoDeS Oxford Instruments MercuryiPS magnet power
    supply
    """

    def __init__(self, name: str, address: str, visalib=None,
                 field_limits: Optional[Callable[[float,
                                                  float,
                                                  float], bool]]=None,
                 **kwargs) -> None:
        """
        Args:
            name: The name to give this instrument internally in QCoDeS
            address: The VISA resource of the instrument. Note that a
                socket connection to port 7020 must be made
            visalib: The VISA library to use. Leave blank if not in simulation
                mode.
            field_limits: A function describing the allowed field
                range (T). The function shall take (x, y, z) as an input and
                return a boolean describing whether that field value is
                acceptable.
        """

        if field_limits is not None and not(callable(field_limits)):
            raise ValueError('Got wrong type of field_limits. Must be a '
                             'function from (x, y, z) -> Bool. Received '
                             f'{type(field_limits)} instead.')

        if visalib:
            visabackend = visalib.split('@')[1]
        else:
            visabackend = 'NI'

        # ensure that a socket is used unless we are in simulation mode
        if not address.endswith('SOCKET') and visabackend != 'sim':
            raise ValueError('Incorrect VISA resource name. Must be of type '
                             'TCPIP0::XXX.XXX.XXX.XXX::7020::SOCKET.')

        super().__init__(name, address, terminator='\n', visalib=visalib,
                         **kwargs)

        # to ensure a correct snapshot, we must wrap the get function
        self.IDN.get = self.IDN._wrap_get(self._idn_getter)

        self.firmware = self.IDN()['firmware']

        # TODO: Query instrument to ensure which PSUs are actually present
        for grp in ['GRPX', 'GRPY', 'GRPZ']:
            psu_name = grp
            psu = MercurySlavePS(self, psu_name, grp)
            self.add_submodule(psu_name, psu)

        self._field_limits = (field_limits if field_limits else
                              lambda x, y, z: True)

        self._target_vector = FieldVector(x=self.GRPX.field(),
                                          y=self.GRPY.field(),
                                          z=self.GRPZ.field())

        for coord, unit in zip(
                ['x', 'y', 'z', 'r', 'theta',   'phi',     'rho'],
                ['T', 'T', 'T', 'T', 'degrees', 'degrees', 'T']):
            self.add_parameter(name=f'{coord}_target',
                               label=f'{coord.upper()} target field',
                               unit=unit,
                               get_cmd=partial(self._get_component, coord),
                               set_cmd=partial(self._set_target, coord))

            self.add_parameter(name=f'{coord}_measured',
                               label=f'{coord.upper()} measured field',
                               unit='T',
                               get_cmd=partial(self._get_measured, [coord]))

        # FieldVector-valued parameters #

        self.add_parameter(name="field_target",
                           label="target field",
                           unit="T",
                           get_cmd=self._get_target_field,
                           set_cmd=self._set_target_field)

        self.add_parameter(name="field_measured",
                           label="measured field",
                           unit="T",
                           get_cmd=self._get_field)

        self.add_parameter(name="field_ramp_rate",
                           label="ramp rate",
                           unit="T/s",
                           get_cmd=self._get_ramp_rate,
                           set_cmd=self._set_ramp_rate)

        self.connect_message()

    def _get_component(self, coordinate: str) -> float:
        return self._target_vector.get_components(coordinate)[0]

    def _get_target_field(self) -> FieldVector:
        return FieldVector(
            **{
                coord: self._get_component(coord)
                for coord in 'xyz'
            }
        )

    def _get_ramp_rate(self) -> FieldVector:
        return FieldVector(
            x=self.GRPX.field_ramp_rate(),
            y=self.GRPY.field_ramp_rate(),
            z=self.GRPZ.field_ramp_rate(),
        )

    def _set_ramp_rate(self, rate: FieldVector) -> None:
        self.GRPX.field_ramp_rate(rate.x)
        self.GRPY.field_ramp_rate(rate.y)
        self.GRPZ.field_ramp_rate(rate.z)

    def _get_measured(self, coordinates: List[str]) -> Union[float,
                                                             List[float]]:
        """
        Get the measured value of a coordinate. Measures all three fields
        and computes whatever coordinate we asked for.
        """
        meas_field = FieldVector(x=self.GRPX.field(),
                                 y=self.GRPY.field(),
                                 z=self.GRPZ.field())

        if len(coordinates) == 1:
            return meas_field.get_components(*coordinates)[0]
        else:
            return meas_field.get_components(*coordinates)

    def _get_field(self) -> FieldVector:
        return FieldVector(
            x=self.x_measured(),
            y=self.y_measured(),
            z=self.z_measured()
        )

    def _set_target(self, coordinate: str, target: float) -> None:
        """
        The function to set a target value for a coordinate, i.e. the set_cmd
        for the XXX_target parameters
        """
        # first validate the new target
        valid_vec = FieldVector()
        valid_vec.copy(self._target_vector)
        valid_vec.set_component(**{coordinate: target})
        components = valid_vec.get_components('x', 'y', 'z')
        if not self._field_limits(*components):
            raise ValueError(f'Cannot set {coordinate} target to {target}, '
                             'that would violate the field_limits. ')

        # update our internal target cache
        self._target_vector.set_component(**{coordinate: target})

        # actually assign the target on the slaves
        cartesian_targ = self._target_vector.get_components('x', 'y', 'z')
        for targ, slave in zip(cartesian_targ, self.submodules.values()):
            slave.field_target(targ)

    def _set_target_field(self, field: FieldVector) -> None:
        for coord in 'xyz':
            self._set_target(coord, field[coord])

    def _idn_getter(self) -> Dict[str, str]:
        """
        Parse the raw non-SCPI compliant IDN string into an IDN dict

        Returns:
            The normal IDN dict
        """
        raw_idn_string = self.ask('*IDN?')
        resps = raw_idn_string.split(':')

        idn_dict = {'model': resps[2], 'vendor': resps[1],
                    'serial': resps[3], 'firmware': resps[4]}

        return idn_dict

    def _ramp_simultaneously(self) -> None:
        """
        Ramp all three fields to their target simultaneously at their given
        ramp rates. NOTE: there is NO guarantee that this does not take you
        out of your safe region. Use with care.
        """
        for slave in self.submodules.values():
            slave.ramp_to_target()

    def _ramp_safely(self) -> None:
        """
        Ramp all three fields to their target using the 'first-down-then-up'
        sequential ramping procedure. This function is BLOCKING.
        """
        meas_vals = self._get_measured(['x', 'y', 'z'])
        targ_vals = self._target_vector.get_components('x', 'y', 'z')
        order = np.argsort(np.abs(np.array(targ_vals) - np.array(meas_vals)))

        for slave in np.array(list(self.submodules.values()))[order]:
            slave.ramp_to_target()
            # now just wait for the ramp to finish
            # (unless we are testing)
            if self.visabackend == 'sim':
                pass
            else:
                while slave.ramp_status() == 'TO SET':
                    time.sleep(0.1)

    def is_ramping(self) -> bool:
        """
        Returns True if any axis has a ramp status that is either 'TO SET' or
        'TO ZERO'
        """
        ramping_statuus = ['TO SET', 'TO ZERO']
        is_x_ramping = self.GRPX.ramp_status() in ramping_statuus
        is_y_ramping = self.GRPY.ramp_status() in ramping_statuus
        is_z_ramping = self.GRPZ.ramp_status() in ramping_statuus

        return is_x_ramping or is_y_ramping or is_z_ramping

    def set_new_field_limits(self, limit_func: Callable) -> None:
        """
        Assign a new field limit function to the driver

        Args:
            limit_func: must be a function mapping (Bx, By, Bz) -> True/False
              where True means that the field is INSIDE the allowed region
        """

        # first check that the current target is allowed
        if not limit_func(*self._target_vector.get_components('x', 'y', 'z')):
            raise ValueError('Can not assign new limit function; present '
                             'target is illegal. Please change the target '
                             'and try again.')

        self._field_limits = limit_func

    def ramp(self, mode: str="safe") -> None:
        """
        Ramp the fields to their present target value

        Args:
            mode: how to ramp, either 'simul' or 'safe'. In 'simul' mode,
              the fields are ramping simultaneously in a non-blocking mode.
              There is no safety check that the safe zone is not exceeded. In
              'safe' mode, the fields are ramped one-by-one in a blocking way
              that ensures that the total field stays within the safe region
              (provided that this region is convex).
        """
        if mode not in ['simul', 'safe']:
            raise ValueError('Invalid ramp mode. Please provide either "simul"'
                             ' or "safe".')

        meas_vals = self._get_measured(['x', 'y', 'z'])
        # we asked for three coordinates, so we know that we got a list
        meas_vals = cast(List[float], meas_vals)

        for cur, slave in zip(meas_vals, self.submodules.values()):
            if slave.field_target() != cur:
                if slave.field_ramp_rate() == 0:
                    raise ValueError(f'Can not ramp {slave}; ramp rate set to'
                                     ' zero!')

        # then the actual ramp
        {'simul': self._ramp_simultaneously,
         'safe': self._ramp_safely}[mode]()

    def ask(self, cmd: str) -> str:
        """
        Since Oxford Instruments implement their own version of a SCPI-like
        language, we implement our own reader. Note that this command is used
        for getting and setting (asking and writing) alike.

        Args:
            cmd: the command to send to the instrument
        """

        visalog.debug(f"Writing to instrument {self.name}: {cmd}")
        resp = self.visa_handle.query(cmd)
        visalog.debug(f"Got instrument response: {resp}")

        if 'INVALID' in resp:
            log.error('Invalid command. Got response: {}'.format(resp))
            base_resp = resp
        # if the command was not invalid, it can either be a SET or a READ
        # SET:
        elif resp.endswith('VALID'):
            base_resp = resp.split(':')[-2]
        # READ:
        else:
            # For "normal" commands only (e.g. '*IDN?' is excepted):
            # the response of a valid command echoes back said command,
            # thus we remove that part
            base_cmd = cmd.replace('READ:', '')
            base_resp = resp.replace('STAT:{}'.format(base_cmd), '')

        return base_resp
