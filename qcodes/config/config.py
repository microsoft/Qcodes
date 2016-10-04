import json
import jsonschema
import logging
import os
import pkg_resources as pkgr

from os.path import expanduser
from pathlib import Path

logger = logging.getLogger(__name__)

EMPTY_USER_SCHEMA = "User schema at {} not found." + \
                    "User settings won't be validated"
MISS_DESC = """ Passing a description without a type does not make sense.
Description is ignored """


class Config():
    """ Qcodes config system

    This allows you to start with sane defaults, which you cant' change, and
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

    config_file_name = "config.json"
    schema_file_name = "schema.json"

    # get abs path of packge config file
    default_file_name = pkgr.resource_filename(__name__, config_file_name)
    current_config_path = default_file_name

    # get abs path of schema  file
    schema_default_file_name = pkgr.resource_filename(__name__,
                                                      schema_file_name)

    with open(schema_default_file_name, "r") as fp:
        current_schema = json.load(fp)

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

    current_config = None

    def __init__(self):
        self.current_config = self.load_default()

    def load_default(self):
        """
        Load defaults and validates.
        A  configuration file must be called config.json
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
        config = self.load_config(self.default_file_name)

        if os.path.isfile(self.home_file_name):
            home_config = self.load_config(self.home_file_name)
            config.update(home_config)
            self.validate(config, self.current_schema,
                          self.schema_home_file_name)

        if os.path.isfile(self.env_file_name):
            env_config = self.load_config(self.env_file_name)
            config.update(env_config)
            self.validate(config, self.current_schema,
                          self.schema_env_file_name)

        if os.path.isfile(self.cwd_file_name):
            cwd_config = self.load_config(self.cwd_file_name)
            config.update(cwd_config)
            self.validate(config, self.current_schema,
                          self.schema_cwd_file_name)

        return config

    def validate(self, json_config, schema, extra_schema_path=None):
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
                    user["properties"].update(
                            new_user["properties"]
                            )
                jsonschema.validate(json_config, schema)
            else:
                logger.warning(EMPTY_USER_SCHEMA.format(extra_schema_path))
        else:
            jsonschema.validate(json_config, schema)

    def add(self, key, value, value_type=None, description=None):
        """ Add custom config value in place.
        Add  key, value with optional value_type to user cofnig and schema.
        If value_type is specified then the new value is validated.

        Args:
            key(str): key to be added under user config
            value (any): value to add to config
            value_type(Optional(string)): type of value
                allowed are string, boolean, integer
            description (str): description of key to add to schema

        Examples:
            >>> defaults.add("trace_color", "blue", "string", "description")
        will update the config:
            `...
            "user": { "trace_color": "blue"}
            ...`
        and the schema:
            `...
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
            ...`

        Todo:
            - Add enum  support for value_type
        """
        self.current_config["user"].update({key: value})

        if value_type is None:
            if description is not None:
                logger.warning(MISS_DESC)
        else:
            # update schema!
            schema_entry = {key: {"type": value_type}}
            if description is not None:
                schema_entry = {key: {
                                    "type": value_type,
                                    "description": description}
                                }
            # the schema is nested we only update properties of the user object
            user = self.current_schema['properties']["user"]
            user["properties"].update(schema_entry)
            self.validate(self.current_config, self.current_schema)

# -------------------------------------------------------------------- I/O

    def load_config(self, path):
        """ Load a config JSON file
        As a side effect it records which file is loaded

        Args:
            path(str): path to the config file
        Raises:
            FileNotFoundError: if config is missing
        Return:
            Union[dict, None]: a vaild config or None
        """
        with open(path, "r") as fp:
            config = json.load(fp)

        self.current_config_path = path
        return config

    def save_config(self, path):
        """ Save to file(s)
        Saves current config to path.

        Args:
            path (string): path of new file(s)
        """
        with open(path, "w") as fp:
            json.dump(self.current_config, fp)

    def save_schema(self, path):
        """ Save to file(s)
        Saves current schema to path.

        Args:
            path (string): path of new file(s)
        """
        with open(path, "w") as fp:
            json.dump(self.current_schema, fp)

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

# -------------------------------------------------------------------- magic

    def __getitem__(self, name):
        val = self.current_config
        for key in name.split('.'):
            val = val[key]
        return val

    def describe(self, name):
        val = self.current_config
        sch = self.current_schema["properties"]
        for key in name.split('.'):
            val = val[key]
            if sch.get(key):
                sch = sch[key]
                if sch.get("properties"):
                    sch = sch["properties"]

        description = sch.get("description", None) or "Generic value"
        _type = str(sch.get("type", None)) or "Not defined"
        default = sch.get("default", None) or "Not defined"

        # add cool description to docstring
        base_docstring = """{}.\n Current value: {}. Type: {}. Default: {}."""

        doc = base_docstring.format(
                description,
                val,
                _type,
                default
                )
        return doc

    def __getattr__(self, name):
        if name in dir(self.current_config):
            return getattr(self.current_config, name)

    def __repr__(self):
        old = super().__repr__()
        return "\n".join([str(self.current_config),
                         self.current_config_path,
                         old])
