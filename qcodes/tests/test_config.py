import copy
import json
import os
from functools import partial
from pathlib import Path
from unittest.mock import PropertyMock, mock_open

import jsonschema
import pytest
import qcodes
from qcodes.configuration import Config
from qcodes.tests.common import default_config

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


def side_effect(map, name):
    return map[name]


@pytest.fixture(scope="function")
def path_to_config_file_on_disk(tmp_path):

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

    with open(str(tmp_path / 'qcodesrc.json'), 'w') as f:
        f.write(json.dumps(contents))
    with open(str(tmp_path / 'qcodesrc_schema.json'), 'w') as f:
        f.write(json.dumps(SCHEMA))

    yield str(tmp_path)


@pytest.fixture(name='config')
def _make_config():
    conf = Config()
    yield conf


@pytest.fixture(name='load_config')
def _make_mock_config(mocker):
    schema = mocker.patch.object(
        Config,
        'current_schema',
        new_callable=PropertyMock
    )
    env = mocker.patch.object(
        Config,
        'env_file_name',
        new_callable=PropertyMock
    )
    load_config = mocker.patch.object(Config, 'load_config')
    isfile = mocker.patch('os.path.isfile')
    schema.return_value = copy.deepcopy(SCHEMA)
    env.return_value = ENV_KEY
    isfile.return_value = True
    load_config.side_effect = partial(side_effect, GOOD_CONFIG_MAP)
    yield load_config


def test_missing_config_file(config):
    with pytest.raises(FileNotFoundError):
        config.load_config("./missing.json")


@pytest.mark.skipif(Path.cwd() == Path.home(),
                    reason="This test requires that "
                           "working dir is different from homedir.")
def test_default_config_files(
        config,
        load_config
):
    load_config.side_effect = partial(side_effect, GOOD_CONFIG_MAP)
    # don't try to load custom schemas
    config.schema_cwd_file_name = None
    config.schema_home_file_name = None
    config.schema_env_file_name = None
    config.defaults, _ = config.load_default()
    config = config.update_config()
    assert config == CONFIG


@pytest.mark.skipif(Path.cwd() == Path.home(),
                    reason="This test requires that "
                           "working dir is different from homedir.")
def test_bad_config_files(config, load_config):

    load_config.side_effect = partial(side_effect, BAD_CONFIG_MAP)
    # don't try to load custom schemas
    config.schema_cwd_file_name = None
    config.schema_home_file_name = None
    config.schema_env_file_name = None
    with pytest.raises(jsonschema.exceptions.ValidationError):
        config.defaults, _ = config.load_default()
        config.update_config()


@pytest.mark.skipif(Path.cwd() == Path.home(),
                    reason="This test requires that "
                           "working dir is different from homedir.")
def test_user_schema(config, load_config, mocker):
    mocker.patch("builtins.open", mock_open(read_data=USER_SCHEMA))
    load_config.side_effect = partial(side_effect, GOOD_CONFIG_MAP)
    config.defaults, _ = config.load_default()
    config = config.update_config()
    assert config == CONFIG


def test_bad_user_schema(config, load_config, mocker):
    mocker.patch("builtins.open", mock_open(read_data=USER_SCHEMA))
    load_config.side_effect = partial(side_effect, BAD_CONFIG_MAP)
    with pytest.raises(jsonschema.exceptions.ValidationError):
        config.defaults, _ = config.load_default()
        config.update_config()


def test_update_user_config(config, mocker):

    myconfig = mocker.patch.object(
        Config,
        "current_config",
        new_callable=PropertyMock
    )
    # deep copy because we mutate state
    myconfig.return_value = copy.deepcopy(CONFIG)

    config.add("foo", "bar")
    assert config.current_config == UPDATED_CONFIG


def test_update_and_validate_user_config(config, mocker):

    myconfig = mocker.patch.object(
        Config,
        "current_config",
        new_callable=PropertyMock
    )
    schema = mocker.patch.object(
        Config,
        'current_schema',
        new_callable=PropertyMock
    )
    schema.return_value = copy.deepcopy(SCHEMA)
    # deep copy because we mutate state
    myconfig.return_value = copy.deepcopy(CONFIG)
    config.add("foo", "bar", "string", "foo", "bar")
    assert config.current_config == UPDATED_CONFIG
    assert config.current_schema == UPDATED_SCHEMA


def test_update_from_path(path_to_config_file_on_disk):
    with default_config():
        cfg = qcodes.config

        # check that the default is still the default
        assert cfg["core"]["db_debug"] is False

        cfg.update_config(path=path_to_config_file_on_disk)
        assert cfg['core']['db_debug'] is True

        # check that the settings NOT specified in our config file on path
        # are still saved as configurations
        assert cfg['gui']['notebook'] is True
        assert cfg['station']['default_folder'] == '.'

        expected_path = os.path.join(path_to_config_file_on_disk,
                                     'qcodesrc.json')
        assert cfg.current_config_path == expected_path


def test_repr():
    cfg = qcodes.config
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
        value = 'testvalue'
        value_type = 'string'
        description = 'A test'
        default = 'testdefault'

        cfg = qcodes.config
        cfg.add(key=key, value=value, value_type=value_type,
                description=description, default=default)

        desc = cfg.describe(f'user.{key}')
        expected_desc = (f"{description}.\nCurrent value: {value}. "
                         f"Type: {value_type}. Default: {default}.")

        assert desc == expected_desc
