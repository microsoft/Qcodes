import collections
import copy
import json
import logging
import os
import pkg_resources as pkgr

from os.path import expanduser
from pathlib import Path

import jsonschema
from typing import Dict

logger = logging.getLogger(__name__)

EMPTY_USER_SCHEMA = "User schema at {} not found." + \
                    "User settings won't be validated"
MISS_DESC = """ Passing a description without a type does not make sense.
Description is ignored """

BASE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "description": "schema for a user qcodes config file",
    "properties": {},
    "required": []
}


class Config():
    """
    QCoDeS config system

    Start with sane defaults, which you can't change, and
    then customize your experience using files that update the configuration.


    Attributes:
        config_file_name(str): Name of config file
        schema_file_name(str): Name of schema file

        default_file_name(str):Filene name of default config
        schema_default_file_name(str):Filene name of default schema

        home_file_name(str):Filene name of home config
        schema_home_file_name(str):Filene name of home schema

        env_file_name(str):Filene name of env config
        schema_env_file_name(str):Filene name of env schema

        cwd_file_name(str):Filene name of cwd config
        schema_cwd_file_name(str):Filene name of cwd schema

        current_config(dict): Vaild config values
        current_schema(dict): Validators and desciptions of config values
        current_config_path(path): Path of the currently loaded config

    """

    config_file_name = "qcodesrc.json"
    schema_file_name = "qcodesrc_schema.json"

    # get abs path of packge config file
    default_file_name = pkgr.resource_filename(__name__, config_file_name)
    current_config_path = default_file_name

    # get abs path of schema  file
    schema_default_file_name = pkgr.resource_filename(__name__,
                                                      schema_file_name)

    # home dir, os independent
    home_file_name = expanduser("~/{}".format(config_file_name))
    schema_home_file_name = home_file_name.replace(config_file_name,
                                                   schema_file_name)

    # this is for *nix people
    env_file_name = os.environ.get("QCODES_CONFIG", "")
    schema_env_file_name = env_file_name.replace(config_file_name,
                                                 schema_file_name)
    # current working dir
    cwd_file_name = "{}/{}".format(Path.cwd(), config_file_name)
    schema_cwd_file_name = cwd_file_name.replace(config_file_name,
                                                 schema_file_name)

    current_schema = None
    current_config = None

    defaults = None
    defaults_schema = None

    _diff_config: Dict[str, dict] = {}
    _diff_schema: Dict[str, dict] = {}

    def __init__(self):
        self.defaults, self.defaults_schema = self.load_default()
        self.current_config = self.update_config()

    def load_default(self):
        defaults = self.load_config(self.default_file_name)
        defaults_schema = self.load_config(self.schema_default_file_name)
        self.validate(defaults, defaults_schema)
        return defaults, defaults_schema

    def update_config(self):
        """
        Load defaults and validates.
        A  configuration file must be called qcodesrc.json
        A schema file must be called schema.json
        Configuration files (and their schema) are loaded and updated from the
        default directories in the following order:

            - default json config file from the repository
            - user json config in user home directory
            - user json config in $QCODES_CONFIG
            - user json config in current working directory

        If a key/value is not specified in the user configuration the default
        is used.  Configs are validated after every update.
        Validation is also performed against a user provied schema if it's
        found in the directory.
        """
        config = copy.deepcopy(self.defaults)
        self.current_schema = copy.deepcopy(self.defaults_schema)

        if os.path.isfile(self.home_file_name):
            home_config = self.load_config(self.home_file_name)
            config = update(config, home_config)
            self.validate(config, self.current_schema,
                          self.schema_home_file_name)

        if os.path.isfile(self.env_file_name):
            env_config = self.load_config(self.env_file_name)
            config = update(config, env_config)
            self.validate(config, self.current_schema,
                          self.schema_env_file_name)

        if os.path.isfile(self.cwd_file_name):
            cwd_config = self.load_config(self.cwd_file_name)
            config = update(config, cwd_config)
            self.validate(config, self.current_schema,
                          self.schema_cwd_file_name)

        return config

    def validate(self, json_config=None, schema=None, extra_schema_path=None):
        """
        Validate configuration, if no arguments are passed, the default
        validators are used.

        Args:
            json_config (Optiona[string]) : json file to validate
            schema (Optiona[dict]): schema dictionary
            extra_schema_path (Optiona[string]): schema path that contains
                    extra validators to be added to schema dictionary
        """
        if extra_schema_path is not None:
            # add custom validation
            if os.path.isfile(extra_schema_path):
                with open(extra_schema_path) as f:
                    # user schema has to be both vaild in itself
                    # but then just update the user properties
                    # so that default types and values can NEVER
                    # be overwritten
                    new_user = json.load(f)["properties"]["user"]
                    user = schema["properties"]['user']
                    user["properties"].update(new_user["properties"])
                jsonschema.validate(json_config, schema)
            else:
                logger.warning(EMPTY_USER_SCHEMA.format(extra_schema_path))
        else:
            if json_config is None and schema is None:
                jsonschema.validate(self.current_config, self.current_schema)
            else:
                jsonschema.validate(json_config, schema)

    def add(self, key, value, value_type=None, description=None, default=None):
        """ Add custom config value in place.
        Add  key, value with optional value_type to user cofnig and schema.
        If value_type is specified then the new value is validated.

        Args:
            key(str): key to be added under user config
            value (any): value to add to config
            value_type(Optional(string)): type of value
                allowed are string, boolean, integer
            default (str): default value, stored only in the schema
            description (str): description of key to add to schema

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
        self.current_config["user"].update({key: value})

        if self._diff_config.get("user", True):
            self._diff_config["user"] = {}
        self._diff_config.get("user").update({key: value})

        if value_type is None:
            if description is not None:
                logger.warning(MISS_DESC)
        else:
            # update schema!
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
            props.get("user").update(schema_entry)

    def load_config(self, path):
        """ Load a config JSON file
        As a side effect it records which file is loaded

        Args:
            path(str): path to the config file
        Raises:
            FileNotFoundError: if config is missing
        Return:
            Union[DotDict, None]: a dot accessible config object
        """
        with open(path, "r") as fp:
            config = json.load(fp)

        logger.debug(f'Loading config from {path}')

        config = DotDict(config)
        self.current_config_path = path
        return config

    def save_config(self, path):
        """ Save to file(s)
        Saves current config to path.

        Args:
            path (string): path of new file(s)
        """
        with open(path, "w") as fp:
            json.dump(self.current_config, fp, indent=4)

    def save_schema(self, path):
        """ Save to file(s)
        Saves current schema to path.

        Args:
            path (string): path of new file(s)
        """
        with open(path, "w") as fp:
            json.dump(self.current_schema, fp, indent=4)

    def save_to_home(self):
        """ Save  files to home dir
        """
        self.save_config(self.home_file_name)
        self.save_schema(self.schema_home_file_name)

    def save_to_env(self):
        """ Save  files to env path
        """
        self.save_config(self.env_file_name)
        self.save_schema(self.schema_env_file_name)

    def save_to_cwd(self):
        """ Save files to current working dir
        """
        self.save_config(self.cwd_file_name)
        self.save_schema(self.schema_cwd_file_name)

    def describe(self, name):
        """
        Describe a configuration entry

        Args:
            name (str): name of entry to describe
        """
        val = self.current_config
        sch = self.current_schema["properties"]
        for key in name.split('.'):
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

    def __getitem__(self, name):
        val = self.current_config
        for key in name.split('.'):
            val = val[key]
        return val

    def __getattr__(self, name):
        return getattr(self.current_config, name)

    def __repr__(self):
        old = super().__repr__()
        base = """Current values: \n {} \n Current path: \n {} \n {}"""
        return base.format(self.current_config, self.current_config_path, old)


class DotDict(dict):
    """
    Wrapper dict that allows to get dotted attributes
    """

    def __init__(self, value=None):
        if value is None:
            pass
        else:
            for key in value:
                self.__setitem__(key, value[key])

    def __setitem__(self, key, value):
        if '.' in key:
            myKey, restOfKey = key.split('.', 1)
            target = self.setdefault(myKey, DotDict())
            target[restOfKey] = value
        else:
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
            dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        if '.' not in key:
            return dict.__getitem__(self, key)
        myKey, restOfKey = key.split('.', 1)
        target = dict.__getitem__(self, myKey)
        return target[restOfKey]

    def __contains__(self, key):
        if '.' not in key:
            return dict.__contains__(self, key)
        myKey, restOfKey = key.split('.', 1)
        target = dict.__getitem__(self, myKey)
        return restOfKey in target

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self)))

    # dot acces baby
    __setattr__ = __setitem__
    __getattr__ = __getitem__


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d
