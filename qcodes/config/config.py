import json
import jsonschema
import logging
import os
import pkg_resources as pkgr

from os.path import expanduser
from pathlib import Path


class Config():
    config_file_name = "config.json"
    schema_file_name = "schema.json"

    # get abs path of packge config file
    default_file_name = pkgr.resource_filename(__name__,
                                               config_file_name)
    # get abs path of schema  file
    schema_default_file_name = pkgr.resource_filename(__name__,
                                                      schema_file_name)

    with open(schema_default_file_name, "r") as fp:
        schema = json.load(fp)

    # home dir, os independent
    home_file_name = expanduser(config_file_name)
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

        Configuration values are loaded and updated in the following order:
            - default json config file from the repository
            - user json config in user home directory
            - user json config in $QCODES_CONFIG
            - user json config in current working directory

        If a key/value is not specified in the user configuration the default
        is used.  Configs are validated after every update.
        Validation is also performed against a custom user provied schema
        if it's provided.
        """
        config = self.load_config(self.default_file_name)
        if os.path.isfile(self.home_file_name):
            home_config = self.load_config(self.home_file_name)
            config.update(home_config)
            self.validate(config, self.schema, self.schema_home_file_name)
        if os.path.isfile(self.env_file_name):
            env_config = self.load_config(self.env_file_name)
            config.update(env_config)
            self.validate(config, self.schema, self.schema_env_file_name)
        if os.path.isfile(self.cwd_file_name):
            cwd_config = self.load_config(self.cwd_file_name)
            config.update(cwd_config)
            self.validate(config, self.schema, self.schema_cwd_file_name)
        return config

    def validate(self, json_config, schema, extra_schema_path=None):
        if extra_schema_path is not None:
            # add custom validation
            if os.path.isfile(extra_schema_path):
                with open(extra_schema_path) as f:
                    # user schema has to be both vaild in itself
                    # but then just update the properties
                    schema["properties"].update(json.load(f)["properties"])

                jsonschema.validate(json_config, schema)
            else:
                logger.warning("User schema is empty.\
                               Custom settings won't be validated")
        else:
            jsonschema.validate(json_config, schema)

    def load_config(self, path):
        """Load a config JSON file

        Args:
            path(str): path to the config file
        Raises:
            FileNotFoundError: if config is missing
        Return:
            Union[dict, None]: a vaild config or None
        """
        with open(path, "r") as fp:
            config = json.load(fp)
        return config

    def __getitem__(self, name):
        val = self.current_config
        for key in name.split('.'):
            val = val[key]
        return val
