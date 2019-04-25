"""Station objects - collect all the equipment you use to do an experiment."""
from contextlib import suppress
from typing import Dict, List, Optional, Sequence, Any
from functools import partial
import importlib
import logging
import warnings
import os
from copy import deepcopy
from ruamel.yaml import YAML

from qcodes.utils.metadata import Metadatable
from qcodes.utils.helpers import make_unique, DelegateAttributes

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.parameter import StandardParameter
import qcodes.utils.validators as validators
from qcodes.monitor.monitor import Monitor
from qcodes.config.config import Config

from qcodes.actions import _actions_snapshot


log = logging.getLogger(__name__)

# config from the qcodesrc.json file (working directory, home or qcodes dir)
cfg = Config()
ENABLE_FORCED_RECONNECT = cfg["station_configurator"]["enable_forced_reconnect"]
DEFAULT_CONFIG_FOLDER = cfg["station_configurator"]["default_folder"]
DEFAULT_CONFIG_FILE = cfg["station_configurator"]["default_file"]

PARAMETER_ATTRIBUTES = ['label', 'unit', 'scale', 'inter_delay', 'delay',
                        'step', 'offset']


class Station(Metadatable, DelegateAttributes):

    """
    A representation of the entire physical setup.

    Lists all the connected Components and the current default
    measurement (a list of actions). Contains a convenience method
    `.measure()` to measure these defaults right now, but this is separate
    from the code used by `Loop`.

    Args:
        *components (list[Any]): components to add immediately to the
             Station. Can be added later via self.add_component

        monitor (None): Not implemented, the object that monitors the system
            continuously

        default (bool): is this station the default, which gets
            used in Loops and elsewhere that a Station can be specified,
            default true

        update_snapshot (bool): immediately update the snapshot
            of each component as it is added to the Station, default true

    Attributes:
        default (Station): class attribute to store the default station
        delegate_attr_dicts (list): a list of names (strings) of dictionaries
            which are (or will be) attributes of self, whose keys should be
            treated as attributes of self
    """

    default = None # type: 'Station'

    def __init__(self, *components: Metadatable,
                 config_file: Optional[str]=None,
                 monitor: Monitor=None, default: bool=True,
                 update_snapshot: bool=True, **kwargs) -> None:
        super().__init__(**kwargs)

        # when a new station is defined, store it in a class variable
        # so it becomes the globally accessible default station.
        # You can still have multiple stations defined, but to use
        # other than the default one you must specify it explicitly.
        # If for some reason you want this new Station NOT to be the
        # default, just specify default=False
        if default:
            Station.default = self

        self.components = {} # type: Dict[str, Metadatable]
        for item in components:
            self.add_component(item, update_snapshot=update_snapshot)

        self.monitor = monitor
        self.config_file = config_file

        self.default_measurement: List[Any] = []
        self._added_methods: List[str] = []
        self._monitor_parameters = []

        self.load_config_file(self.config_file)


    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str]=None) -> Dict:
        """
        State of the station as a JSON-compatible dict.

        Note: in the station contains an instrument that has already been
        closed, not only will it not be snapshotted, it will also be removed
        from the station during the execution of this function.

        Args:
            update (bool): If True, update the state by querying the
             all the children: f.ex. instruments, parameters, components, etc.
             If False, just use the latest values in memory.

        Returns:
            dict: base snapshot
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
                                  ManualParameter,
                                  StandardParameter
                                  )):
                snap['parameters'][name] = itm.snapshot(update=update)
            else:
                snap['components'][name] = itm.snapshot(update=update)

        for c in components_to_remove:
            self.remove_component(c)

        return snap

    def add_component(self, component: Metadatable, name: str=None,
                      update_snapshot: bool=True) -> str:
        """
        Record one component as part of this Station.

        Args:
            component (Any): components to add to the Station.
            name (str): name of the component
            update_snapshot (bool): immediately update the snapshot
                of each component as it is added to the Station, default true

        Returns:
            str: The name assigned this component, which may have been changed to
            make it unique among previously added components.

        """
        try:
            component.snapshot(update=update_snapshot)
        except:
            pass
        if name is None:
            name = getattr(component, 'name',
                           'component{}'.format(len(self.components)))
        namestr = make_unique(str(name), self.components)
        self.components[namestr] = component
        return namestr

    def remove_component(self, name: str) -> Optional[Metadatable]:
        """
        Remove a component with a given name from this Station.

        Args:
            name: name of the component

        Returns:
            the component that has been removed (this behavior is the same as
            for python dictionaries)

        Raises:
            KeyError: if a component with the given name is not part of this
                station
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
        by .measure
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
            *actions: parameters to mesure
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
        """Shortcut to components dict."""
        return self.components[key]

    delegate_attr_dicts = ['components']

    def load_config_file(self, filename: Optional[str] = None):

        # 1. load config from file
        if filename is None:
            filename = DEFAULT_CONFIG_FILE
        try:
            with open(filename, 'r') as f:
                self._config = YAML().load(f)
        except FileNotFoundError as e:
            try:
                if not os.path.isabs(filename) and DEFAULT_CONFIG_FOLDER is not None:
                    with open(os.path.join(DEFAULT_CONFIG_FOLDER, filename),
                              'r') as f:
                        self._config = YAML().load(f)
                else:
                    raise e
            except FileNotFoundError:
                log.warning('Could not load default config for Station: ' +
                            e.msg)
                return

        self._instrument_config = self._config['instruments']

        # 2. add config to snapshot componenents
        class ConfigComponent:
            def __init__(self, data):
                self.data = data

            def snapshot(self, update=True):
                return self.data

        # this overwrites any previous configuration
        # but does invoke snapshoting
        self.components['StationConfigurator'] = ConfigComponent(self._config)


        # 3. create shortcut methods to instantiate instruments via
        # `load_<instrument_name>()` so that autocompletion can be used
        # first remove methods that have been added by a previous
        # `load_config_file` call
        while len(self._added_methods):
            delattr(self, self._added_methods.pop())

        # add shortcut methods
        for instrument_name in self._instrument_config.keys():
            # TODO: check if name is valid (does not start with digit, contain
            # dot, other signs etc.)
            method_name = f'load_{instrument_name}'
            if method_name.isidentifier():
                setattr(self, method_name, partial(
                    self.load_instrument, identifier=instrument_name))

                self._added_methods.append(method_name)
            else:
                log.warning(f'Invalid identifier: ' +
                            f'for the instrument {instrument_name} no ' +
                            f'lazy loading method {method_name} could be ' +
                            'created in the StationConfigurator')


    def load_instrument(self, identifier: str,
                        revive_instance: bool=False,
                        **kwargs) -> Instrument:
        """
        Creates an instrument driver as described by the loaded config file.

        Args:
            identifier: the identfying string that is looked up in the yaml
                configuration file, which identifies the instrument to be added
            revive_instance: If true, try to return an instrument with the
                specified name instead of closing it and creating a new one.
            **kwargs: additional keyword arguments that get passed on to the
                __init__-method of the instrument to be added.
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
            raise RuntimeError('Instrument {} not found in config.'
                               .format(identifier))
        instr_cfg = self._instrument_config[identifier]

        # config is not parsed for errors. On errors qcodes should be able to
        # to report them

        # check if instrument is already defined and close connection
        if instr_cfg.get('ENABLE_FORCED_RECONNECT',
                         ENABLE_FORCED_RECONNECT):
            # save close instrument and remove from monitor list
            with suppress(KeyError):
                instr = Instrument.find_instrument(identifier)
                # remove parameters related to this instrument from the
                # monitor list
                self._monitor_parameters = [v for v in self._monitor_parameters
                                           if v.root_instrument is not instr]
                # remove instrument from station snapshot
                self.components.pop(instr.name)
                # del will remove weakref and close the instrument
                instr.close()
                del instr

        # instantiate instrument
        module = importlib.import_module(instr_cfg['driver'])
        instr_class = getattr(module, instr_cfg['type'])

        init_kwargs = instr_cfg.get('init',{})
        # somebody might have a empty init section in the config
        init_kwargs = {} if init_kwargs is None else init_kwargs
        if 'address' in instr_cfg:
            init_kwargs['address'] = instr_cfg['address']
        if 'port' in instr_cfg:
            init_kwargs['port'] = instr_cfg['port']
        # make explicitly passed arguments overide the ones from the config file
        # the intuitive line:

        # We are mutating the dict below
        # so make a copy to ensure that any changes
        # does not leek into the station config object
        # specifically we may be passing non pickleable
        # instrument instances via kwargs
        instr_kwargs = deepcopy(init_kwargs)
        instr_kwargs.update(kwargs)

        instr = instr_class(name=identifier, **instr_kwargs)
        # setup

        # local function to refactor common code from defining new parameter
        # and setting existing one
        def setup_parameter_from_dict(parameter, options_dict):
            for attr, val in options_dict.items():
                if attr in PARAMETER_ATTRIBUTES:
                    # set the attributes of the parameter, that map 1 to 1
                    setattr(parameter, attr, val)
                    # extra attributes that need parsing
                elif attr == 'limits':
                    lower, upper = [float(x) for x in val.split(',')]
                    parameter.vals = validators.Numbers(lower, upper)
                elif attr == 'monitor' and val is True:
                    self._monitor_parameters.append(parameter)
                elif attr == 'alias':
                    setattr(instr, val, parameter)
                elif attr == 'initial_value':
                    # skip value attribute so that it gets set last
                    # when everything else has been set up
                    pass
                else:
                    log.warning(f'Attribute {attr} no recognized when'
                                f' instatiating parameter \"{parameter.name}\"')
            if 'initial_value' in options_dict:
                parameter.set(options_dict['initial_value'])

        # setup existing parameters
        for name, options in instr_cfg.get('parameters', {}).items():
            # get the parameter object from its name:
            p = instr
            for level in name.split('.'):
                p = getattr(p, level)
            setup_parameter_from_dict(p, options)

        # setup new parameters
        for name, options in instr_cfg.get('add_parameters', {}).items():
            # allow only top level paremeters for now
            # pop source only temporarily
            source = options.pop('source', False)
            if source:
                source_p = instr
                for level in source.split('.'):
                    source_p = getattr(source_p, level)
                instr.add_parameter(name,
                                    DelegateParameter,
                                    source=source_p)
            else:
                instr.add_parameter(name, Parameter)
            p = getattr(instr, name)
            setup_parameter_from_dict(p, options)
            # restore source
            options['source'] = source

        # add the instrument to the station
        self.add_component(instr)

        # restart Monitor
        Monitor(*self._monitor_parameters)

        return instr
