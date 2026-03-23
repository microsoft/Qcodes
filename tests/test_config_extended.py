"""Extended tests for the QCoDeS configuration module.

Targets uncovered lines in qcodes/configuration/config.py including:
DotDict operations, Config.describe, Config.__getitem__, Config.add,
Config.save_config/save_schema, Config.__repr__, Config.__getattr__,
and the module-level update() function.
"""

from __future__ import annotations

import copy
import json
import logging

import pytest

import qcodes
from qcodes.configuration import Config
from qcodes.configuration.config import MISS_DESC, DotDict, update

# ---------------------------------------------------------------------------
# DotDict tests
# ---------------------------------------------------------------------------


class TestDotDictInit:
    def test_init_none(self) -> None:
        d = DotDict(None)
        assert len(d) == 0

    def test_init_flat(self) -> None:
        d = DotDict({"a": 1, "b": 2})
        assert d["a"] == 1
        assert d["b"] == 2

    def test_init_nested_dict_becomes_dotdict(self) -> None:
        d = DotDict({"outer": {"inner": 42}})
        assert isinstance(d["outer"], DotDict)
        assert d["outer"]["inner"] == 42


class TestDotDictGetItem:
    def test_simple_key(self) -> None:
        d = DotDict({"x": 10})
        assert d["x"] == 10

    def test_dotted_key(self) -> None:
        d = DotDict({"a": {"b": {"c": 99}}})
        assert d["a.b.c"] == 99

    def test_missing_key_raises(self) -> None:
        d = DotDict({"a": 1})
        with pytest.raises(KeyError):
            d["nonexistent"]

    def test_missing_nested_key_raises(self) -> None:
        d = DotDict({"a": {"b": 1}})
        with pytest.raises(KeyError):
            d["a.z"]


class TestDotDictSetItem:
    def test_simple_set(self) -> None:
        d = DotDict()
        d["key"] = "value"
        assert d["key"] == "value"

    def test_dotted_set_creates_intermediates(self) -> None:
        d = DotDict()
        d["a.b.c"] = 42
        assert d["a"]["b"]["c"] == 42
        assert isinstance(d["a"], DotDict)
        assert isinstance(d["a"]["b"], DotDict)

    def test_set_plain_dict_converts_to_dotdict(self) -> None:
        d = DotDict()
        d["x"] = {"nested": 1}
        assert isinstance(d["x"], DotDict)
        assert d["x"]["nested"] == 1

    def test_overwrite_value(self) -> None:
        d = DotDict({"a": {"b": 1}})
        d["a.b"] = 2
        assert d["a.b"] == 2


class TestDotDictContains:
    def test_simple_contains(self) -> None:
        d = DotDict({"a": 1})
        assert "a" in d
        assert "missing" not in d

    def test_dotted_contains(self) -> None:
        d = DotDict({"a": {"b": {"c": 1}}})
        assert "a.b.c" in d
        assert "a.b" in d
        assert "a.b.z" not in d

    def test_non_string_key_returns_false(self) -> None:
        d = DotDict({"a": 1})
        assert (123 in d) is False


class TestDotDictDeepCopy:
    def test_deepcopy_returns_dotdict(self) -> None:
        d = DotDict({"a": {"b": [1, 2, 3]}})
        d2 = copy.deepcopy(d)
        assert isinstance(d2, DotDict)
        assert d2["a"]["b"] == [1, 2, 3]
        # mutating copy does not affect original
        d2["a"]["b"].append(4)
        assert d["a"]["b"] == [1, 2, 3]


class TestDotDictAttrAccess:
    def test_getattr(self) -> None:
        d = DotDict({"hello": "world"})
        assert d.hello == "world"

    def test_setattr(self) -> None:
        d = DotDict()
        d.foo = "bar"
        assert d["foo"] == "bar"


# ---------------------------------------------------------------------------
# update() function tests
# ---------------------------------------------------------------------------


class TestUpdateFunction:
    def test_simple_update(self) -> None:
        d: dict = {"a": 1, "b": 2}
        u = {"b": 3, "c": 4}
        result = update(d, u)
        assert result == {"a": 1, "b": 3, "c": 4}
        assert result is d  # in-place

    def test_nested_recursive_merge(self) -> None:
        d: dict = {"x": {"y": 1, "z": 2}}
        u = {"x": {"z": 99, "w": 100}}
        result = update(d, u)
        assert result["x"]["y"] == 1
        assert result["x"]["z"] == 99
        assert result["x"]["w"] == 100

    def test_non_mapping_replaces(self) -> None:
        d: dict = {"a": {"nested": True}}
        u = {"a": "flat_now"}
        result = update(d, u)
        assert result["a"] == "flat_now"

    def test_new_nested_key(self) -> None:
        d: dict = {}
        u = {"a": {"b": 1}}
        result = update(d, u)
        assert result["a"]["b"] == 1


# ---------------------------------------------------------------------------
# Config class tests
# ---------------------------------------------------------------------------


class TestConfigGetItem:
    def test_access_top_level_section(self) -> None:
        cfg = qcodes.config
        core = cfg["core"]
        assert isinstance(core, DotDict)

    def test_access_nested_key(self) -> None:
        cfg = qcodes.config
        val = cfg["core.db_debug"]
        assert isinstance(val, bool)

    def test_missing_key_raises(self) -> None:
        cfg = qcodes.config
        with pytest.raises(KeyError):
            cfg["nonexistent_section_xyz"]


class TestConfigDescribe:
    def test_describe_known_key(self) -> None:
        cfg = qcodes.config
        desc = cfg.describe("core.db_debug")
        assert isinstance(desc, str)
        assert "Current value:" in desc
        assert "Type:" in desc
        assert "Default:" in desc

    def test_describe_user_section(self) -> None:
        cfg = qcodes.config
        cfg.add("testdesc", "val", "string", "My description", "default_val")
        desc = cfg.describe("user.testdesc")
        assert "My description" in desc
        assert "val" in desc
        assert "string" in desc


class TestConfigAdd:
    def test_add_without_type(self) -> None:
        cfg = qcodes.config
        cfg.add("simple_key", "simple_val")
        assert cfg.current_config is not None
        assert cfg.current_config["user"]["simple_key"] == "simple_val"

    def test_add_with_type_only(self) -> None:
        cfg = qcodes.config
        cfg.add("typed_key", 42, "integer")
        assert cfg.current_config is not None
        assert cfg.current_config["user"]["typed_key"] == 42

    def test_add_with_type_and_description(self) -> None:
        cfg = qcodes.config
        cfg.add("full_key", "hello", "string", "A full key", "default_hello")
        assert cfg.current_config is not None
        assert cfg.current_config["user"]["full_key"] == "hello"
        # Verify schema was updated
        assert cfg.current_schema is not None
        user_props = cfg.current_schema["properties"]["user"]["properties"]
        assert "full_key" in user_props
        assert user_props["full_key"]["type"] == "string"
        assert user_props["full_key"]["description"] == "A full key"
        assert user_props["full_key"]["default"] == "default_hello"

    def test_add_description_without_type_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = qcodes.config
        with caplog.at_level(logging.WARNING, logger="qcodes.configuration.config"):
            cfg.add("warn_key", "val", description="ignored desc")
        assert MISS_DESC.strip() in caplog.text.strip()


class TestConfigRepr:
    def test_repr_contains_current_info(self) -> None:
        cfg = qcodes.config
        r = repr(cfg)
        assert "Current values:" in r
        assert "Current paths:" in r


class TestConfigGetattr:
    def test_getattr_delegates(self) -> None:
        cfg = qcodes.config
        # Accessing an attribute that exists in current_config
        user = cfg.user
        assert isinstance(user, DotDict)


class TestConfigSave:
    def test_save_config(self, tmp_path) -> None:
        cfg = Config()
        path = str(tmp_path / "saved_config.json")
        cfg.save_config(path)
        with open(path) as f:
            data = json.load(f)
        assert "core" in data

    def test_save_schema(self, tmp_path) -> None:
        cfg = Config()
        path = str(tmp_path / "saved_schema.json")
        cfg.save_schema(path)
        with open(path) as f:
            data = json.load(f)
        assert "properties" in data

    def test_save_and_reload(self, tmp_path) -> None:
        cfg = Config()
        config_path = str(tmp_path / "roundtrip_config.json")
        cfg.save_config(config_path)
        loaded = Config.load_config(config_path)
        assert isinstance(loaded, DotDict)
        assert loaded["core"]["db_debug"] == cfg.current_config["core"]["db_debug"]


class TestConfigLoadConfig:
    def test_load_config_returns_dotdict(self) -> None:
        loaded = Config.load_config(Config.default_file_name)
        assert isinstance(loaded, DotDict)

    def test_load_config_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            Config.load_config("no_such_file_anywhere.json")


class TestConfigValidate:
    def test_validate_uses_current_if_no_args(self) -> None:
        cfg = Config()
        # Should not raise - validates current_config against current_schema
        cfg.validate()

    def test_validate_no_schema_raises(self) -> None:
        cfg = Config()
        cfg.current_schema = None
        with pytest.raises(RuntimeError, match="Cannot validate"):
            cfg.validate()

    def test_validate_no_config_uses_current(self) -> None:
        cfg = Config()
        # Passing schema but no json_config should use current_config
        assert cfg.current_schema is not None
        cfg.validate(schema=cfg.current_schema)
