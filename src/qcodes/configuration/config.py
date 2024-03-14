from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Mapping
from importlib.resources import files
from os.path import expanduser
from pathlib import Path
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)

EMPTY_USER_SCHEMA = "User schema at {} not found. User settings won't be validated"
MISS_DESC = """ Passing a description without a type does not make sense.
Description is ignored """

BASE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "description": "schema for a user qcodes config file",
    "properties": {},
    "required": []
}

# https://github.com/python/mypy/issues/4182
_PARENT_MODULE = ".".join(__loader__.name.split(".")[:-1])  # type: ignore[name-defined]


class Config:
    """
    QCoDeS config system

    Start with sane defaults, which you can't change, and
    then customize your experience using files that update the configuration.

    """

    config_file_name = "qcodesrc.json"
    """Name of config file"""
    schema_file_name = "qcodesrc_schema.json"
    """Name of schema file"""
    # get abs path of packge config file
    default_file_name = str(files(_PARENT_MODULE) / config_file_name)
    """Filename of default config"""
    current_config_path = default_file_name
    """Path of the last loaded config file"""

    # get abs path of schema  file
    schema_default_file_name = str(files(_PARENT_MODULE) / schema_file_name)
    """Filename of default schema"""

    # home dir, os independent
    home_file_name = expanduser(os.path.join("~", config_file_name))
    """Filename of home config"""
    schema_home_file_name = home_file_name.replace(config_file_name,
                                                   schema_file_name)
    """Filename of home schema"""

    # this is for *nix people
    env_file_name = os.environ.get("QCODES_CONFIG", "")
    """Filename of env config"""
    schema_env_file_name = env_file_name.replace(config_file_name,
                                                 schema_file_name)
    """Filename of env schema"""
    # current working dir
    cwd_file_name = os.path.join(Path.cwd(), config_file_name)
    """Filename of cwd config"""
    schema_cwd_file_name = cwd_file_name.replace(config_file_name,
                                                 schema_file_name)
    """Filename of cwd schema"""

    current_schema: DotDict | None = None
    """Validators and descriptions of config values"""
    current_config: DotDict | None = None
    """Valid config values"""

    defaults: DotDict
    """The default configuration"""
    defaults_schema: DotDict
    """The default schema"""

    def __init__(self, path: str | None = None) -> None:
        """
        Args:
            path: Optional path to directory containing
                a `qcodesrc.json` config file
        """
        self._loaded_config_files = [self.default_file_name]
        self._diff_config: dict[str, Any] = {}
        self._diff_schema: dict[str, Any] = {}

        self.config_file_path = path
        self.defaults, self.defaults_schema = self.load_default()
        self.update_config()

    def load_default(self) -> tuple[DotDict, DotDict]:
        defaults = self.load_config(self.default_file_name)
        defaults_schema = self.load_config(self.schema_default_file_name)
        self.validate(defaults, defaults_schema)
        return defaults, defaults_schema

    def update_config(self, path: str | None = None) -> dict[str, Any]:
        """
        Load defaults updates with cwd, env, home and the path specified
        and validates.
        A configuration file must be called qcodesrc.json
        A schema file must be called qcodesrc_schema.json
        Configuration files (and their schema) are loaded and updated from the
        directories in the following order:

            - default json config file from the repository
            - user json config in user home directory
            - user json config in $QCODES_CONFIG
            - user json config in current working directory
            - user json file in the path specified

        If a key/value is not specified in the user configuration the default
        is used. Key/value pairs loaded later will take preference over those
        loaded earlier.
        Configs are validated after every update.
        Validation is also performed against a user provided schema if it's
        found in the directory.

        Args:
            path: Optional path to directory containing a `qcodesrc.json`
               config file
        """
        config = copy.deepcopy(self.defaults)
        self.current_schema = copy.deepcopy(self.defaults_schema)

        self._loaded_config_files = [self.default_file_name]

        self._update_config_from_file(self.home_file_name,
                                      self.schema_home_file_name,
                                      config)
        self._update_config_from_file(self.env_file_name,
                                      self.schema_env_file_name,
                                      config)
        self._update_config_from_file(self.cwd_file_name,
                                      self.schema_cwd_file_name,
                                      config)
        if path is not None:
            self.config_file_path = path
        if self.config_file_path is not None:
            config_file = os.path.join(self.config_file_path,
                                       self.config_file_name)
            schema_file = os.path.join(self.config_file_path,
                                       self.schema_file_name)
            self._update_config_from_file(config_file, schema_file, config)
        if config is None:
            raise RuntimeError("Could not load config from any of the "
                               "expected locations.")
        self.current_config = config
        self.current_config_path = self._loaded_config_files[-1]

        return config

    def _update_config_from_file(
        self, file_path: str, schema: str, config: dict[str, Any]
    ) -> None:
        """
        Updated ``config`` dictionary with config information from file in
        ``file_path`` that has schema specified in ``schema``

        Args:
            file_path: Path to `qcodesrc.json` config file
            schema: Path to `qcodesrc_schema.json` to be used
            config: Config dictionary to be updated.
        """
        if os.path.isfile(file_path):
            self._loaded_config_files.append(file_path)
            my_config = self.load_config(file_path)
            config = update(config, my_config)
            self.validate(config, self.current_schema, schema)

    def validate(
        self,
        json_config: Mapping[str, Any] | None = None,
        schema: Mapping[str, Any] | None = None,
        extra_schema_path: str | None = None,
    ) -> None:
        """
        Validate configuration; if no arguments are passed, the default
        config is validated against the default schema. If either
        ``json_config`` or ``schema`` is passed the corresponding
        default is not used.

        Args:
            json_config: json dictionary to validate
            schema: schema dictionary
            extra_schema_path: schema path that contains extra validators to be
                added to schema dictionary
        """
        if schema is None:
            if self.current_schema is None:
                raise RuntimeError("Cannot validate as current_schema is None")
            schema = self.current_schema

        if json_config is None:
            json_config = self.current_config

        if extra_schema_path is not None:
            # add custom validation
            if os.path.isfile(extra_schema_path):
                with open(extra_schema_path) as f:
                    # user schema has to be both valid in itself
                    # but then just update the user properties
                    # so that default types and values can NEVER
                    # be overwritten
                    new_user = json.load(f)["properties"]["user"]
                    user = schema["properties"]['user']
                    user["properties"].update(new_user["properties"])
            else:
                logger.warning(EMPTY_USER_SCHEMA.format(extra_schema_path))

        jsonschema.validate(json_config, schema)

    def add(
        self,
        key: str,
        value: Any,
        value_type: str | None = None,
        description: str | None = None,
        default: Any | None = None,
    ) -> None:
        """Add custom config value in place

        Adds ``key``, ``value`` with optional ``value_type`` to user config and
        schema. If ``value_type`` is specified then the new value is validated.

        Args:
            key: key to be added under user config
            value: value to add to config
            value_type: type of value, allowed are string, boolean, integer
            description: description of key to add to schema
            default: default value, stored only in the schema

        Examples:
            >>> defaults.add("trace_color", "blue", "string", "description")

        will update the config:

        ::

            ...
            "user": { "trace_color": "blue"}
            ...

        and the schema:

        ::

            ...
            "user":{
                "type" : "object",
                "description": "controls user settings of qcodes"
                "properties" : {
                            "trace_color": {
                            "description" : "description",
                            "type": "string"
                            }
                    }
            }
            ...

        Todo:
            - Add enum  support for value_type
            - finish _diffing
        """
        if self.current_config is None:
            raise RuntimeError("Cannot add value to empty config")
        self.current_config["user"].update({key: value})

        if self._diff_config.get("user", True):
            self._diff_config["user"] = {}
        self._diff_config["user"].update({key: value})

        if value_type is None:
            if description is not None:
                logger.warning(MISS_DESC)
        else:
            # update schema!
            schema_entry: dict[str, dict[str, str | Any]]
            schema_entry = {key: {"type": value_type}}
            if description is not None:
                schema_entry = {
                    key: {
                        "type": value_type,
                        "default": default,
                        "description": description
                    }
                }
            # the schema is nested we only update properties of the user object
            if self.current_schema is None:
                raise RuntimeError("Cannot add value as no current schema is "
                                   "set")
            user = self.current_schema['properties']["user"]
            user["properties"].update(schema_entry)
            self.validate(self.current_config, self.current_schema)

            # TODO(giulioungaretti) finish diffing
            # now we update the entire schema
            # and the entire configuration
            # if it's saved then it will always
            # take precedence even if the defaults
            # values are changed upstream, and the local
            # ones were actually left to their default
            # values
            if not self._diff_schema:
                self._diff_schema = BASE_SCHEMA

            props = self._diff_schema['properties']
            if props.get("user", True):
                props["user"] = {}
            props["user"].update(schema_entry)

    @staticmethod
    def load_config(path: str) -> DotDict:
        """Load a config JSON file

        Args:
            path: path to the config file
        Return:
            a dot accessible dictionary config object
        Raises:
            FileNotFoundError: if config is missing
        """
        with open(path) as fp:
            config = json.load(fp)

        logger.debug(f'Loading config from {path}')

        config_dot_dict = DotDict(config)
        return config_dot_dict

    def save_config(self, path: str) -> None:
        """
        Save current config to file at given path.

        Args:
            path: path of new file
        """
        with open(path, "w") as fp:
            json.dump(self.current_config, fp, indent=4)

    def save_schema(self, path: str) -> None:
        """
        Save current schema to file at given path.

        Args:
            path: path of new file
        """
        with open(path, "w") as fp:
            json.dump(self.current_schema, fp, indent=4)

    def save_to_home(self) -> None:
        """Save config and schema to files in home dir"""
        self.save_config(self.home_file_name)
        self.save_schema(self.schema_home_file_name)

    def save_to_env(self) -> None:
        """Save config and schema to files in path specified in env variable"""
        self.save_config(self.env_file_name)
        self.save_schema(self.schema_env_file_name)

    def save_to_cwd(self) -> None:
        """Save config and schema to files in current working dir"""
        self.save_config(self.cwd_file_name)
        self.save_schema(self.schema_cwd_file_name)

    def describe(self, name: str) -> str:
        """
        Describe a configuration entry

        Args:
            name: name of entry to describe in 'dotdict' notation,
                e.g. name="user.scriptfolder"
        """
        val = self.current_config
        if val is None:
            raise RuntimeError("Config is empty, cannot describe entry.")
        if self.current_schema is None:
            raise RuntimeError("No schema found, cannot describe entry.")
        sch = self.current_schema["properties"]
        for key in name.split('.'):
            if val is None:
                raise RuntimeError(f"Cannot describe {name} Some part of it "
                                   f"is null")
            val = val[key]
            if sch.get(key):
                sch = sch[key]
            else:
                sch = sch['properties'][key]
        description = sch.get("description", None) or "Generic value"
        _type = str(sch.get("type", None)) or "Not defined"
        default = sch.get("default", None) or "Not defined"

        # add cool description to docstring
        base_docstring = """{}.\nCurrent value: {}. Type: {}. Default: {}."""

        doc = base_docstring.format(description, val, _type, default)

        return doc

    def __getitem__(self, name: str) -> Any:
        val = self.current_config
        for key in name.split('.'):
            if val is None:
                raise KeyError(f"{name} not found in current config")
            val = val[key]
        return val

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current_config, name)

    def __repr__(self) -> str:
        old = super().__repr__()
        output = (f"Current values: \n {self.current_config} \n"
                  f"Current paths: \n {self._loaded_config_files} \n"
                  f"{old}")
        return output


class DotDict(dict[str, Any]):
    """
    Wrapper dict that allows to get dotted attributes

    Requires keys to be strings.
    """

    def __init__(self, value: Mapping[str, Any] | None = None):
        if value is None:
            pass
        else:
            for key in value:
                self.__setitem__(key, value[key])

    def __setitem__(self, key: str, value: Any) -> None:
        if '.' in key:
            myKey, restOfKey = key.split('.', 1)
            target = self.setdefault(myKey, DotDict())
            target[restOfKey] = value
        else:
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
            dict.__setitem__(self, key, value)

    def __getitem__(self, key: str) -> Any:
        if '.' not in key:
            return dict.__getitem__(self, key)
        myKey, restOfKey = key.split('.', 1)
        target = dict.__getitem__(self, myKey)
        return target[restOfKey]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if '.' not in key:
            return super().__contains__(key)
        myKey, restOfKey = key.split('.', 1)
        target = dict.__getitem__(self, myKey)
        return restOfKey in target

    def __deepcopy__(self, memo: dict[Any, Any] | None) -> DotDict:
        return DotDict(copy.deepcopy(dict(self)))

    def __getattr__(self, name: str) -> Any:
        """
        Overwrite ``__getattr__`` to provide dot access
        """
        return self.__getitem__(name)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Overwrite ``__setattr__`` to provide dot access
        """
        self.__setitem__(key, value)


def update(d: dict[Any, Any], u: Mapping[Any, Any]) -> dict[Any, Any]:
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d
