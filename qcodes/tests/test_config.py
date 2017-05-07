import copy
import jsonschema

from functools import partial
from unittest.mock import mock_open, patch, PropertyMock
from unittest import TestCase
from qcodes.config import Config

VALID_JSON = "{}"
ENV_KEY = "/dev/random"
BAD_JSON = "{}"


# expected config after successful loading of config files
CONFIG = {"a": 1, "b": 2, "h": 2,
          "user": {"foo":  "1"},
          "c": 3, "bar": True, "z": 4}

# expected config after updade by user
UPDATED_CONFIG = {"a": 1, "b": 2, "h": 2,
                  "user": {"foo":  "bar"},
                  "c": 3, "bar": True, "z": 4}

# the schema does not cover extra fields, so users can pass
# wathever they want
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

# schema updaed by adding custom fileds by the
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

# in this case the home config is messging up a type
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
