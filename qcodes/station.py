"""
Station objects - collect all the equipment you use to do an experiment.
"""


from contextlib import suppress
from typing import (
    Dict, List, Optional, Sequence, Any, cast, AnyStr, IO, Iterator, Tuple)
from types import ModuleType
from functools import partial
import importlib
import logging
import os
import itertools
import json
import pkgutil
import inspect
from copy import deepcopy, copy
from collections import UserDict
from typing import Union
import jsonschema
import warnings

import qcodes
from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import (
    DelegateAttributes, YAML, checked_getattr, get_qcodes_path,
    get_qcodes_user_path)
from qcodes.utils.deprecate import issue_deprecation_warning

from qcodes.instrument.base import Instrument, InstrumentBase
from qcodes.instrument.channel import ChannelList
from qcodes.instrument.parameter import (
    Parameter, ManualParameter,
    DelegateParameter, _BaseParameter)
import qcodes.utils.validators as validators
from qcodes.monitor.monitor import Monitor

from qcodes.actions import _actions_snapshot


log = logging.getLogger(__name__)

PARAMETER_ATTRIBUTES = ['label', 'unit', 'scale', 'inter_delay', 'post_delay',
                        'step', 'offset']

SCHEMA_TEMPLATE_PATH = os.path.join(
    get_qcodes_path('dist', 'schemas'),
    'station-template.schema.json')
SCHEMA_PATH = get_qcodes_user_path('schemas', 'station.schema.json')
STATION_YAML_EXT = '*.station.yaml'


def get_config_enable_forced_reconnect() -> bool:
    return qcodes.config["station"]["enable_forced_reconnect"]


def get_config_default_folder() -> Optional[str]:
    return qcodes.config["station"]["default_folder"]


def get_config_default_file() -> Optional[str]:
    return qcodes.config["station"]["default_file"]


def get_config_use_monitor() -> Optional[str]:
    return qcodes.config["station"]["use_monitor"]


ChannelOrInstrumentBase = Union[InstrumentBase, ChannelList]


class ValidationWarning(Warning):
    """Replacement for jsonschema.error.ValidationError as warning."""

    pass



class Station(Metadatable, DelegateAttributes):

    """
    A representation of the entire physical setup.

    Lists all the connected Components and the current default
    measurement (a list of actions).

    Args:
        *components (list[Any]): components to add immediately to the
            Station. Can be added later via ``self.add_component``.
        monitor (None): Not implemented, the object that monitors the system
            continuously.
        default (bool): Is this station the default?
        update_snapshot (bool): Immediately update the snapshot of each
            component as it is added to the Station.

    Attributes:
        default (Station): Class attribute to store the default station.
        delegate_attr_dicts (list): A list of names (strings) of dictionaries
            which are (or will be) attributes of ``self``,
            whose keys should be treated as attributes of ``self``.
    """

    default: Optional['Station'] = None

    def __init__(self, *components: Metadatable,
                 config_file: Optional[str] = None,
                 use_monitor: Optional[bool] = None, default: bool = True,
                 update_snapshot: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)

        # when a new station is defined, store it in a class variable
        # so it becomes the globally accessible default station.
        # You can still have multiple stations defined, but to use
        # other than the default one you must specify it explicitly.
        # If for some reason you want this new Station NOT to be the
        # default, just specify default=False
        if default:
            Station.default = self

        self.components: Dict[str, Metadatable] = {}
        for item in components:
            self.add_component(item, update_snapshot=update_snapshot)

        self.use_monitor = use_monitor
        self.config_file = config_file

        self.default_measurement: List[Any] = []
        self._added_methods: List[str] = []
        self._monitor_parameters: List[_BaseParameter] = []

        self.load_config_file(self.config_file)

    def snapshot_base(self, update: bool = True,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict:
        """
        State of the station as a JSON-compatible dictionary (everything that
        the custom JSON encoder class :class:`qcodes.utils.helpers.NumpyJSONEncoder`
        supports).

        Note: If the station contains an instrument that has already been
        closed, not only will it not be snapshotted, it will also be removed
        from the station during the execution of this function.

        Args:
            update: If ``True``, update the state by querying the
                all the children: f.ex. instruments, parameters,
                components, etc. If ``False``, just use the latest
                values in memory.
            params_to_skip_update: Not used.

        Returns:
            dict: Base snapshot.
        """
        snap = {
            'instruments': {},
            'parameters': {},
            'components': {},
            'default_measurement': _actions_snapshot(
                self.default_measurement, update)
        }

        components_to_remove = []

        for name, itm in self.components.items():
            if isinstance(itm, Instrument):
                # instruments can be closed during the lifetime of the
                # station object, hence this 'if' allows to avoid
                # snapshotting instruments that are already closed
                if Instrument.is_valid(itm):
                    snap['instruments'][name] = itm.snapshot(update=update)
                else:
                    components_to_remove.append(name)
            elif isinstance(itm, (Parameter,
                                  ManualParameter
                                  )):
                snap['parameters'][name] = itm.snapshot(update=update)
            else:
                snap['components'][name] = itm.snapshot(update=update)

        for c in components_to_remove:
            self.remove_component(c)

        return snap

    def add_component(self, component: Metadatable, name: str = None,
                      update_snapshot: bool = True) -> str:
        """
        Record one component as part of this Station.

        Args:
            component: Components to add to the Station.
            name: Name of the component.
            update_snapshot: Immediately update the snapshot
                of each component as it is added to the Station.

        Returns:
            str: The name assigned this component, which may have been changed
                to make it unique among previously added components.
        """
        try:
            component.snapshot(update=update_snapshot)
        except:
            pass
        if name is None:
            name = getattr(component, 'name',
                           'component{}'.format(len(self.components)))
        namestr = str(name)
        if namestr in self.components.keys():
            raise RuntimeError(
                f'Cannot add component "{namestr}", because a '
                'component of that name is already registered to the station')
        self.components[namestr] = component
        return namestr

    def remove_component(self, name: str) -> Optional[Metadatable]:
        """
        Remove a component with a given name from this Station.

        Args:
            name: Name of the component.

        Returns:
            The component that has been removed (this behavior is the same as
            for Python dictionaries).

        Raises:
            KeyError: If a component with the given name is not part of this
                station.
        """
        try:
            return self.components.pop(name)
        except KeyError as e:
            if name in str(e):
                raise KeyError(f'Component {name} is not part of the station')
            else:
                raise e

    def set_measurement(self, *actions):
        """
        Save a set ``*actions``` as the default measurement for this Station.

        These actions will be executed by default by a Loop if this is the
        default Station, and any measurements among them can be done once
        by ``.measure``.

        Args:
            *actions: parameters to set as default  measurement
        """
        # Validate now so the user gets an error message ASAP
        # and so we don't accept `Loop` as an action here, where
        # it would cause infinite recursion.
        # We need to import Loop inside here to avoid circular import
        from .loops import Loop
        Loop.validate_actions(*actions)

        self.default_measurement = actions

    def measure(self, *actions):
        """
        Measure the default measurement, or parameters in actions.

        Args:
            *actions: Parameters to mesure.
        """
        if not actions:
            actions = self.default_measurement

        out = []

        # this is a stripped down, uncompiled version of how
        # ActiveLoop handles a set of actions
        # callables (including Wait) return nothing, but can
        # change system state.
        for action in actions:
            if hasattr(action, 'get'):
                out.append(action.get())
            elif callable(action):
                action()

        return out

    # station['someitem'] and station.someitem are both
    # shortcuts to station.components['someitem']
    # (assuming 'someitem' doesn't have another meaning in Station)
    def __getitem__(self, key):
        """Shortcut to components dictionary."""
        return self.components[key]

    delegate_attr_dicts = ['components']

    def close_all_registered_instruments(self) -> None:
        """
        Closes all instruments that are registered to this `Station`
        object by calling the :meth:`.base.Instrument.close`-method on
        each one.
        The instruments will be removed from the station and from the
        QCoDeS monitor.
        """
        for c in tuple(self.components.values()):
            if isinstance(c, Instrument):
                self.close_and_remove_instrument(c)

    def load_config_file(self, filename: Optional[str] = None):
        """
        Loads a configuration from a YAML file. If `filename` is not specified
        the default file name from the qcodes configuration will be used.

        Loading of a configuration will update the snapshot of the station and
        make the instruments described in the config file available for
        instantiation with the :meth:`load_instrument` method.

        Additionally the shortcut methods ``load_<instrument_name>`` will be
        updated.
        """
        def get_config_file_path(
                filename: Optional[str] = None) -> Optional[str]:
            filename = filename or get_config_default_file()
            if filename is None:
                return None
            search_list = [filename]
            if (not os.path.isabs(filename) and
                    get_config_default_folder() is not None):
                config_folder = cast(str, get_config_default_folder())
                search_list += [os.path.join(config_folder, filename)]
            for p in search_list:
                if os.path.isfile(p):
                    return p
            return None


        path = get_config_file_path(filename)

        if path is None:
            if filename is not None:
                raise FileNotFoundError(path)
            else:
                if get_config_default_file() is not None:
                    log.warning(
                        'Could not load default config for Station: \n'
                        f'File {get_config_default_file()} not found. \n'
                        'You can change the default config file in '
                        '`qcodesrc.json`.')
                return

        with open(path, 'r') as f:
            self.load_config(f)

    def load_config(self, config: Union[str, IO[AnyStr]]) -> None:
        """
        Loads a configuration from a supplied string or file/stream handle.

        Loading of a configuration will update the snapshot of the station and
        make the instruments described in the config file available for
        instantiation with the :meth:`load_instrument` method.

        Additionally the shortcut methods ``load_<instrument_name>`` will be
        updated.
        """

        def update_station_configuration_snapshot():
            class StationConfig(UserDict):
                def snapshot(self, update=True):
                    return self

            self.components['config'] = StationConfig(self._config)

        def update_load_instrument_methods():
            #  create shortcut methods to instantiate instruments via
            # `load_<instrument_name>()` so that autocompletion can be used
            # first remove methods that have been added by a previous
            # :meth:`load_config_file` call
            while len(self._added_methods):
                delattr(self, self._added_methods.pop())

            # add shortcut methods
            for instrument_name in self._instrument_config.keys():
                method_name = f'load_{instrument_name}'
                if method_name.isidentifier():
                    setattr(self, method_name, partial(
                        self.load_instrument, identifier=instrument_name))
                    self._added_methods.append(method_name)
                else:
                    log.warning(f'Invalid identifier: '
                                f'for the instrument {instrument_name} no '
                                f'lazy loading method {method_name} could '
                                'be created in the Station.')

        # Load template schema, and thereby don't fail on instruments that are
        # not included in the user schema.
        yaml = YAML().load(config)
        with open(SCHEMA_TEMPLATE_PATH) as f:
            schema = json.load(f)
        try:
            jsonschema.validate(yaml, schema)
        except jsonschema.exceptions.ValidationError as e:
            message = e.message + '\n config:\n'
            if isinstance(config, str):
                message += config
            else:
                config.seek(0)
                message += config.read()
            warnings.warn(message, ValidationWarning)

        self._config = yaml

        self._instrument_config = self._config['instruments']
        update_station_configuration_snapshot()
        update_load_instrument_methods()

    def close_and_remove_instrument(self,
                                    instrument: Union[Instrument, str]
                                    ) -> None:
        """
        Safely close instrument and remove from station and monitor list.
        """
        # remove parameters related to this instrument from the
        # monitor list
        if isinstance(instrument, str):
            instrument = Instrument.find_instrument(instrument)

        self._monitor_parameters = [v for v in self._monitor_parameters
                                    if v.root_instrument is not instrument]
        # remove instrument from station snapshot
        self.remove_component(instrument.name)
        # del will remove weakref and close the instrument
        instrument.close()
        del instrument

    def load_instrument(self, identifier: str,
                        revive_instance: bool = False,
                        **kwargs) -> Instrument:
        """
        Creates an :class:`~.Instrument` instance as described by the
        loaded configuration file.

        Args:
            identifier: The identfying string that is looked up in the yaml
                configuration file, which identifies the instrument to be added.
            revive_instance: If ``True``, try to return an instrument with the
                specified name instead of closing it and creating a new one.
            **kwargs: Additional keyword arguments that get passed on to the
                ``__init__``-method of the instrument to be added.
        """
        # try to revive the instrument
        if revive_instance and Instrument.exist(identifier):
            return Instrument.find_instrument(identifier)

        # load file
        # try to reload file on every call. This makes script execution a
        # little slower but makes the overall workflow more convenient.
        self.load_config_file(self.config_file)

        # load from config
        if identifier not in self._instrument_config.keys():
            raise RuntimeError(f'Instrument {identifier} not found in '
                               'instrument config file')
        instr_cfg = self._instrument_config[identifier]

        # TODO: add validation of config for better verbose errors:

        # check if instrument is already defined and close connection
        if instr_cfg.get('enable_forced_reconnect',
                         get_config_enable_forced_reconnect()):
            with suppress(KeyError):
                self.close_and_remove_instrument(identifier)

        # instantiate instrument
        init_kwargs = instr_cfg.get('init', {})
        # somebody might have a empty init section in the config
        init_kwargs = {} if init_kwargs is None else init_kwargs
        if 'address' in instr_cfg:
            init_kwargs['address'] = instr_cfg['address']
        if 'port' in instr_cfg:
            init_kwargs['port'] = instr_cfg['port']
        # make explicitly passed arguments overide the ones from the config
        # file.
        # We are mutating the dict below
        # so make a copy to ensure that any changes
        # does not leek into the station config object
        # specifically we may be passing non pickleable
        # instrument instances via kwargs
        instr_kwargs = deepcopy(init_kwargs)
        instr_kwargs.update(kwargs)
        name = instr_kwargs.pop('name', identifier)

        if 'driver' in instr_cfg:
            issue_deprecation_warning(
                'use of the "driver"-keyword in the station '
                'configuration file',
                alternative='the "type"-keyword instead, prepending the '
                'driver value'
                ' to it')
            module_name = instr_cfg['driver']
            instr_class_name = instr_cfg['type']
        else:
            module_name = '.'.join(instr_cfg['type'].split('.')[:-1])
            instr_class_name = instr_cfg['type'].split('.')[-1]
        module = importlib.import_module(module_name)
        instr_class = getattr(module, instr_class_name)
        instr = instr_class(name, **instr_kwargs)

        def resolve_instrument_identifier(
            instrument: ChannelOrInstrumentBase,
            identifier: str
        ) -> ChannelOrInstrumentBase:
            """
            Get the instrument, channel or channel_list described by a nested
            string.

            E.g: 'dac.ch1' will return the instance of ch1.
            """
            try:
                for level in identifier.split('.'):
                    instrument = checked_getattr(
                        instrument, level,
                        (InstrumentBase, ChannelList))
            except TypeError:
                raise RuntimeError(
                    f'Cannot resolve `{level}` in {identifier} to an '
                    f'instrument/channel for base instrument '
                    f'{instrument!r}.')
            return instrument

        def resolve_parameter_identifier(
            instrument: ChannelOrInstrumentBase,
            identifier: str
        ) -> _BaseParameter:
            parts = identifier.split('.')
            if len(parts) > 1:
                instrument = resolve_instrument_identifier(
                    instrument,
                    '.'.join(parts[:-1]))
            try:
                return checked_getattr(instrument, parts[-1], _BaseParameter)
            except TypeError:
                raise RuntimeError(
                    f'Cannot resolve parameter identifier `{identifier}` to '
                    f'a parameter on instrument {instrument!r}.')

        def setup_parameter_from_dict(
            parameter: _BaseParameter,
            options: Dict[str, Any]
        ) -> None:
            for attr, val in options.items():
                if attr in PARAMETER_ATTRIBUTES:
                    # set the attributes of the parameter, that map 1 to 1
                    setattr(parameter, attr, val)
                # extra attributes that need parsing
                elif attr == 'limits':
                    if isinstance(val, str):
                        issue_deprecation_warning(
                            ('use of a comma separated string for the limits '
                             'keyword'),
                            alternative='an array like "[lower_lim, upper_lim]"')
                        lower, upper = [float(x) for x in val.split(',')]
                    else:
                        lower, upper = val
                    parameter.vals = validators.Numbers(lower, upper)
                elif attr == 'monitor' and val is True:
                    self._monitor_parameters.append(parameter)
                elif attr == 'alias':
                    setattr(parameter.instrument, val, parameter)
                elif attr == 'initial_value':
                    # skip value attribute so that it gets set last
                    # when everything else has been set up
                    pass
                else:
                    log.warning(f'Attribute {attr} not recognized when '
                                f'instatiating parameter \"{parameter.name}\"')
            if 'initial_value' in options:
                parameter.set(options['initial_value'])

        def add_parameter_from_dict(
            instr: InstrumentBase,
            name: str,
            options: Dict[str, Any]
        ) -> None:
            # keep the original dictionray intact for snapshot
            options = copy(options)
            param_type: type = _BaseParameter
            kwargs = {}
            if 'source' in options:
                param_type = DelegateParameter
                kwargs['source'] = resolve_parameter_identifier(
                    instr.root_instrument,
                    options['source'])
                options.pop('source')
            instr.add_parameter(name, param_type, **kwargs)
            setup_parameter_from_dict(instr.parameters[name], options)

        def update_monitor():
            if ((self.use_monitor is None and get_config_use_monitor())
                    or self.use_monitor):
                # restart Monitor
                Monitor(*self._monitor_parameters)

        for name, options in instr_cfg.get('parameters', {}).items():
            parameter = resolve_parameter_identifier(instr, name)
            setup_parameter_from_dict(parameter, options)
        for name, options in instr_cfg.get('add_parameters', {}).items():
            parts = name.split('.')
            local_instr = (
                instr if len(parts) < 2 else
                resolve_instrument_identifier(instr, '.'.join(parts[:-1])))
            add_parameter_from_dict(local_instr, parts[-1], options)
        self.add_component(instr)
        update_monitor()
        return instr


def update_config_schema(
    additional_instrument_modules: Optional[List[ModuleType]] = None
) -> None:
    """Update the json schema file 'station.schema.json'.

    Args:
        additional_instrument_modules: python modules that contain
            :class:`qcodes.instrument.base.InstrumentBase` definitions
            (and subclasses thereof) to be included as
            values for instrument definition in th station definition
            yaml files.

    """

    def instrument_names_from_module(module: ModuleType) -> Tuple[str, ...]:
        submodules = list(pkgutil.walk_packages(
            module.__path__,  # type: ignore  # mypy issue #1422
            module.__name__ + '.'))
        res = set()
        for s in submodules:
            with suppress(Exception):
                ms = inspect.getmembers(
                    importlib.import_module(s.name),
                    inspect.isclass)
            new_members = [
                f"{instr[1].__module__}.{instr[1].__name__}"
                for instr in ms
                if (issubclass(instr[1], InstrumentBase) and
                    instr[1].__module__.startswith(module.__name__))
            ]
            res.update(new_members)
        return tuple(res)

    def update_schema_file(
        template_path: str,
        output_path: str,
        instrument_names: Tuple[str, ...]
    ) -> None:
        with open(template_path, 'r+') as f:
            data = json.load(f)
        data['definitions']['instruments']['enum'] = instrument_names
        if os.path.exists(output_path):
            os.remove(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

    additional_instrument_modules = additional_instrument_modules or []
    update_schema_file(
        template_path=SCHEMA_TEMPLATE_PATH,
        output_path=SCHEMA_PATH,
        instrument_names=tuple(itertools.chain.from_iterable(
            instrument_names_from_module(m)
            for m in set(
                [qcodes.instrument_drivers] + additional_instrument_modules)
        ))
    )
