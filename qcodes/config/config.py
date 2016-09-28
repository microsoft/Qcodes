import json
import os
import pkg_resources as pkgr
import os
from os.path import expanduser
from pathlib import Path

from jsonschema import validate


class Config():
    config_file_name = "config.json"

    # get abs path of packge config file
    default_file_name = pkgr.resource_filename(__name__,
                                             config_file_name)
    # get abs path of schema  file
    schema_file_name = pkgr.resource_filename(__name__,
                                             "schema.json")

    with open(schema_file_name, "r") as fp:
        schema = json.load(fp)

    # home dir, os independent
    home_file_name = expanduser("~/{}".format(config_file_name))

    # this is for *nix people
    env_file_name = os.environ.get("QCODES_CONFIG", False)

    # current working dir
    cwd_file_name = "{}/{}".format(Path.cwd(), config_file_name)
    current_config = None

    def __init__(self):
        self.current_config = self.load_default()

    def load_default(self):
        """
        Load defaults.

        Configuration values are loaded and updated in the following order:
            - default json config file from the repository
            - user json config in user home directory
            - user json config in $QCODES_CONFIG
            - user json config in current working directory

        If a key/value is not specified in the user configuration the default is used.
        Configs are validated after every update.
        """
        config = self.load_config(self.default_file_name)
        if os.path.isfile(self.home_file_name):
            home_config = self.load_config(self.home_file_name)
            config.update(home_config)
            validate(config, self.schema )
        if os.path.isfile(self.env_file_name):
            env_config = self.load_config(self.env_file_name)
            config.update(env_config)
            validate(config, self.schema )
        if os.path.isfile(self.cwd_file_name):
            cwd_config = self.load_config(self.cwd_file_name)
            config.update(cwd_config)
            validate(config, self.schema )
        return config


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
        for key in name.split( '.' ):
            val = val[key]
        return val
    
