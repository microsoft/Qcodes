import copy
import jsonschema
import os
import json
import unittest
from pathlib import Path

from functools import partial
from contextlib import contextmanager
from unittest.mock import mock_open, patch, PropertyMock
from unittest import TestCase
from typing import Optional
import pytest
import tempfile
import qcodes.config

from qcodes.config import Config

VALID_JSON = "{}"
ENV_KEY = "/dev/random"
BAD_JSON = "{}"


# expected config after successful loading of config files
CONFIG = {"a": 1, "b": 2, "h": 2,
          "user": {"foo":  "1"},
          "c": 3, "bar": True, "z": 4}

# expected config after update by user
UPDATED_CONFIG = {"a": 1, "b": 2, "h": 2,
                  "user": {"foo":  "bar"},
                  "c": 3, "bar": True, "z": 4}

# the schema does not cover extra fields, so users can pass
# whatever they want
SCHEMA = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "a": {
                "type": "integer"
                },
            "b": {
                "type": "integer"
                },
            "z": {
                "type": "integer"
                },
            "c": {
                "type": "integer"
                },
            "bar": {
                "type": "boolean"
                },
            "user": {
                "type": "object",
                "properties": {}
                }
            },
        "required": [
            "z"
            ]
        }

# schema updated by adding custom fields by the
UPDATED_SCHEMA = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "a": {
                "type": "integer"
                },
            "b": {
                "type": "integer"
                },
            "z": {
                "type": "integer"
                },
            "c": {
                "type": "integer"
                },
            "bar": {
                "type": "boolean"
                },
            "user": {
                "type": "object",
                "properties": {
                           "foo":
                           {
                               "type": "string",
                               "default": "bar",
                               "description": "foo"
                               }
                           }
                   }
            },
        "required": [
            "z"
            ]
        }

USER_SCHEMA = """ {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                           "foo":
                           {
                               "type": "string",
                               "description": "foo"
                               }
                           }
                   }
               }
        } """

# example configs
GOOD_CONFIG_MAP = {Config.default_file_name: {"z": 1, "a": 1, "b": 0},
                   ENV_KEY: {"z": 3, "h": 2, "user": {"foo":  "1"}},
                   Config.home_file_name: {"z": 3, "b": 2},
                   Config.cwd_file_name: {"z": 4, "c": 3, "bar": True},
                   Config.schema_cwd_file_name: SCHEMA,
                   Config.schema_home_file_name: SCHEMA,
                   Config.schema_env_file_name: SCHEMA,
                   Config.schema_default_file_name: SCHEMA,
                   }

# in this case the home config is messing up a type
BAD_CONFIG_MAP = {Config.default_file_name: {"z": 1, "a": 1, "b": 0},
                  ENV_KEY: {"z": 3, "h": 2, "user": {"foo":  1}},
                  Config.home_file_name: {"z": 3, "b": "2", "user": "foo"},
                  Config.cwd_file_name: {"z": 4, "c": 3, "bar": True},
                  Config.schema_cwd_file_name: SCHEMA,
                  Config.schema_home_file_name: SCHEMA,
                  Config.schema_env_file_name: SCHEMA,
                  Config.schema_default_file_name: SCHEMA,
                  }


@contextmanager
def default_config(user_config: Optional[str] = None):
    """
    Context manager to temporarily establish default config settings.
    This is achieved by overwriting the config paths of the user-,
    environment-, and current directory-config files with the path of the
    config file in the qcodes repository.
    Additionally the current config object `qcodes.config` gets copied and
    reestablished.

    Args:
        user_config: represents the user config file content.
    """
    home_file_name = qcodes.Config.home_file_name
    schema_home_file_name = qcodes.Config.schema_home_file_name
    env_file_name = qcodes.Config.env_file_name
    schema_env_file_name = qcodes.Config.schema_env_file_name
    cwd_file_name = qcodes.Config.cwd_file_name
    schema_cwd_file_name = qcodes.Config.schema_cwd_file_name

    qcodes.Config.home_file_name = ''
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = os.path.join(tmpdirname, 'user_config.json')
        if user_config is not None:
            with open(file_name, 'w') as f:
                f.write(user_config)

        qcodes.Config.home_file_name = file_name
        qcodes.Config.schema_home_file_name = ''
        qcodes.Config.env_file_name = ''
        qcodes.Config.schema_env_file_name = ''
        qcodes.Config.cwd_file_name = ''
        qcodes.Config.schema_cwd_file_name = ''

        default_config_obj = copy.deepcopy(qcodes.config)
        qcodes.config = qcodes.Config()

        try:
            yield
        finally:
            qcodes.Config.home_file_name = home_file_name
            qcodes.Config.schema_home_file_name = schema_home_file_name
            qcodes.Config.env_file_name = env_file_name
            qcodes.Config.schema_env_file_name = schema_env_file_name
            qcodes.Config.cwd_file_name = cwd_file_name
            qcodes.Config.schema_cwd_file_name = schema_cwd_file_name

            qcodes.config = default_config_obj


def side_effect(map, name):
    return map[name]


@pytest.fixture(scope="function")
def path_to_config_file_on_disk():

    contents = {
        "core": {
            "loglevel": "WARNING",
            "file_loglevel": "INFO",
            "default_fmt": "data/{date}/#{counter}_{name}_{time}",
            "register_magic": True,
            "db_location": "~/experiments.db",
            "db_debug": True  # Different than default
        },  # we omit a required section (gui)
        "user": {
            "scriptfolder": ".",
            "mainfolder": "."
        }  # we omit a non-required section (stationconfigurator)
    }

    with tempfile.TemporaryDirectory() as tmpdirname:
        with open(os.path.join(tmpdirname, 'qcodesrc.json'), 'w') as f:
            f.write(json.dumps(contents))
        with open(os.path.join(tmpdirname, 'qcodesrc_schema.json'), 'w') as f:
            f.write(json.dumps(SCHEMA))

        yield tmpdirname


class TestConfig(TestCase):
    def setUp(self):
        self.conf = Config()

    def test_missing_config_file(self):
        with self.assertRaises(FileNotFoundError):
            self.conf.load_config("./missing.json")

    @patch.object(Config, 'current_schema', new_callable=PropertyMock)
    @patch.object(Config, 'env_file_name', new_callable=PropertyMock)
    @patch.object(Config, 'load_config')
    @patch('os.path.isfile')
    @unittest.skipIf(Path.cwd() == Path.home(),
                     'This test requires that working dir is different from'
                     'homedir.')
    def test_default_config_files(self, isfile, load_config, env, schema):
        # don't try to load custom schemas
        self.conf.schema_cwd_file_name = None
        self.conf.schema_home_file_name = None
        self.conf.schema_env_file_name = None
        schema.return_value = SCHEMA
        env.return_value = ENV_KEY
        isfile.return_value = True
        load_config.side_effect = partial(side_effect, GOOD_CONFIG_MAP)
        self.conf.defaults, self.defaults_schema = self.conf.load_default()
        config = self.conf.update_config()
        self.assertEqual(config, CONFIG)

    @patch.object(Config, 'current_schema', new_callable=PropertyMock)
    @patch.object(Config, 'env_file_name', new_callable=PropertyMock)
    @patch.object(Config, 'load_config')
    @patch('os.path.isfile')
    @unittest.skipIf(Path.cwd() == Path.home(),
                     'This test requires that working dir is different from'
                     'homedir.')
    def test_bad_config_files(self, isfile, load_config, env, schema):
        # don't try to load custom schemas
        self.conf.schema_cwd_file_name = None
        self.conf.schema_home_file_name = None
        self.conf.schema_env_file_name = None
        schema.return_value = SCHEMA
        env.return_value = ENV_KEY
        isfile.return_value = True
        load_config.side_effect = partial(side_effect, BAD_CONFIG_MAP)
        with self.assertRaises(jsonschema.exceptions.ValidationError):
                self.conf.defaults, self.defaults_schema = self.conf.load_default()
                self.conf.update_config()

    @patch.object(Config, 'current_schema', new_callable=PropertyMock)
    @patch.object(Config, 'env_file_name', new_callable=PropertyMock)
    @patch.object(Config, 'load_config')
    @patch('os.path.isfile')
    @patch("builtins.open", mock_open(read_data=USER_SCHEMA))
    @unittest.skipIf(Path.cwd() == Path.home(),
                     'This test requires that working dir is different from'
                     'homedir.')
    def test_user_schema(self, isfile, load_config, env, schema):
        schema.return_value = copy.deepcopy(SCHEMA)
        env.return_value = ENV_KEY
        isfile.return_value = True
        load_config.side_effect = partial(side_effect, GOOD_CONFIG_MAP)
        self.conf.defaults, self.defaults_schema = self.conf.load_default()
        config = self.conf.update_config()
        self.assertEqual(config, CONFIG)

    @patch.object(Config, 'current_schema', new_callable=PropertyMock)
    @patch.object(Config, 'env_file_name', new_callable=PropertyMock)
    @patch.object(Config, 'load_config')
    @patch('os.path.isfile')
    @patch("builtins.open", mock_open(read_data=USER_SCHEMA))
    def test_bad_user_schema(self, isfile, load_config, env, schema):
        schema.return_value = copy.deepcopy(SCHEMA)
        env.return_value = ENV_KEY
        isfile.return_value = True
        load_config.side_effect = partial(side_effect, BAD_CONFIG_MAP)
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            self.conf.defaults, self.defaults_schema = self.conf.load_default()
            self.conf.update_config()

    @patch.object(Config, "current_config", new_callable=PropertyMock)
    def test_update_user_config(self, config):
        # deep copy because we mutate state
        config.return_value = copy.deepcopy(CONFIG)
        self.conf.add("foo", "bar")
        self.assertEqual(self.conf.current_config, UPDATED_CONFIG)

    @patch.object(Config, 'current_schema', new_callable=PropertyMock)
    @patch.object(Config, "current_config", new_callable=PropertyMock)
    def test_update_and_validate_user_config(self, config, schema):
        self.maxDiff = None
        schema.return_value = copy.deepcopy(SCHEMA)
        # deep copy because we mutate state
        config.return_value = copy.deepcopy(CONFIG)
        self.conf.add("foo", "bar", "string", "foo", "bar")
        self.assertEqual(self.conf.current_config, UPDATED_CONFIG)
        self.assertEqual(self.conf.current_schema, UPDATED_SCHEMA)


def test_update_from_path(path_to_config_file_on_disk):
    with default_config():
        cfg = Config()

        # check that the default is still the default
        assert cfg["core"]["db_debug"] is False

        cfg.update_config(path=path_to_config_file_on_disk)
        assert cfg['core']['db_debug'] is True

        # check that the settings NOT specified in our config file on path
        # are still saved as configurations
        assert cfg['gui']['notebook'] is True
        assert cfg['station_configurator']['default_folder'] == '.'

        expected_path = os.path.join(path_to_config_file_on_disk,
                                     'qcodesrc.json')
        assert cfg.current_config_path == expected_path


def test_repr():
    cfg = Config()
    rep = cfg.__repr__()

    expected_rep = (f"Current values: \n {cfg.current_config} \n"
                    f"Current paths: \n {cfg._loaded_config_files} \n"
                    f"{super(Config, cfg).__repr__()}")

    assert rep == expected_rep


def test_add_and_describe():
    """
    Test that a key an be added and described
    """
    with default_config():

        key = 'newkey'
        value ='testvalue'
        value_type ='string'
        description ='A test'
        default = 'testdefault'

        cfg = Config()
        cfg.add(key=key, value=value, value_type=value_type,
                description=description, default=default)


        desc = cfg.describe(f'user.{key}')
        expected_desc = (f"{description}.\nCurrent value: {value}. "
                         f"Type: {value_type}. Default: {default}.")

        assert desc == expected_desc
