from contextlib import suppress
from typing import Optional
from functools import partial
import importlib
import logging
import warnings
import os
from copy import deepcopy
import qcodes
from qcodes.instrument.base import Instrument
from qcodes.station import Station
import qcodes.utils.validators as validators
from qcodes.instrument.parameter import Parameter
from qcodes.monitor.monitor import Monitor
from qcodes.instrument.parameter import DelegateParameter
use_pyyaml = False
try:
    from ruamel.yaml import YAML
except ImportError:
    use_pyyaml = True
    warnings.warn(
        "ruamel yaml not found station configurator is falling back to pyyaml. "
        "It's highly recommended to install ruamel.yaml. This fixes issues with "
        "scientific notation and duplicate instruments in the YAML file")

log = logging.getLogger(__name__)

# config from the qcodesrc.json file (working directory, home or qcodes dir)
enable_forced_reconnect = qcodes.config["station_configurator"]["enable_forced_reconnect"]
default_folder = qcodes.config["station_configurator"]["default_folder"]
default_file = qcodes.config["station_configurator"]["default_file"]


class StationConfigurator:
    """
    The StationConfig class enables the easy creation of intstruments from a
    yaml file.
    Example:
    >>> scfg = StationConfig('exampleConfig.yaml')
    >>> dmm1 = scfg.create_instrument('dmm1')
    """

    PARAMETER_ATTRIBUTES = ['label', 'unit', 'scale', 'inter_delay', 'delay',
                            'step', 'offset']

    def __init__(self, filename: Optional[str] = None,
                 station: Optional[Station] = None) -> None:
        self.monitor_parameters = {}

        if station is None:
            station = Station.default or Station()
        self.station = station
        self.filename = filename

        self.load_file(self.filename)
        for instrument_name in self._instrument_config.keys():
            # TODO: check if name is valid (does not start with digit, contain
            # dot, other signs etc.)
            method_name = f'load_{instrument_name}'
            if method_name.isidentifier():
                setattr(self, method_name,
                        partial(self.load_instrument,
                                identifier=instrument_name))
            else:
                log.warning(f'Invalid identifier: ' +
                            f'for the instrument {instrument_name} no ' +
                            f'lazy loading method {method_name} could be ' +
                            'created in the StationConfigurator')

    def load_file(self, filename: Optional[str] = None):
        if use_pyyaml:
            import yaml
        else:
            yaml = YAML()
        if filename is None:
            filename = default_file
        try:
            with open(filename, 'r') as f:
                self.config = yaml.load(f)
        except FileNotFoundError as e:
            if not os.path.isabs(filename) and default_folder is not None:
                try:
                    with open(os.path.join(default_folder, filename),
                              'r') as f:
                        self.config = yaml.load(f)
                except FileNotFoundError:
                    raise e
            else:
                raise e

        self._instrument_config = self.config['instruments']

        class ConfigComponent:
            def __init__(self, data):
                self.data = data

            def snapshot(self, update=True):
                return self.data

        # this overwrites any previous station
        # configurator but does not call the snapshot
        self.station.components['StationConfigurator'] = ConfigComponent(self.config)

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
        self.load_file(self.filename)

        # load from config
        if identifier not in self._instrument_config.keys():
            raise RuntimeError('Instrument {} not found in config.'
                               .format(identifier))
        instr_cfg = self._instrument_config[identifier]

        # config is not parsed for errors. On errors qcodes should be able to
        # to report them

        # check if instrument is already defined and close connection
        if instr_cfg.get('enable_forced_reconnect',
                         enable_forced_reconnect):
            # save close instrument and remove from monitor list
            with suppress(KeyError):
                instr = Instrument.find_instrument(identifier)
                # remove parameters related to this instrument from the
                # monitor list
                self.monitor_parameters = {
                    k:v for k,v in self.monitor_parameters.items()
                    if v.root_instrument is not instr}
                instr.close()
                # remove instrument from station snapshot
                self.station.components.pop(instr.name)

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
                if attr in self.PARAMETER_ATTRIBUTES:
                    # set the attributes of the parameter, that map 1 to 1
                    setattr(parameter, attr, val)
                    # extra attributes that need parsing
                elif attr == 'limits':
                    lower, upper = [float(x) for x in val.split(',')]
                    parameter.vals = validators.Numbers(lower, upper)
                elif attr == 'monitor' and val is True:
                    self.monitor_parameters[id(parameter)] = parameter
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
        self.station.add_component(instr)

        # restart Monitor
        Monitor(*self.monitor_parameters.values())

        return instr
