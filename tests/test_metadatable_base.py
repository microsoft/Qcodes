"""
Tests for qcodes.metadatable.metadatable_base covering branches
not exercised by test_metadata.py.
"""

from __future__ import annotations

from typing import Any

from qcodes.metadatable import Metadatable
from qcodes.metadatable.metadatable_base import MetadatableWithName, Snapshot

# --------------- helpers ---------------


class ConcreteMetadatableWithName(MetadatableWithName):
    """Minimal concrete implementation for testing."""

    def __init__(
        self,
        name: str,
        full: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._short_name = name
        self._full_name = full or name
        super().__init__(metadata=metadata)

    @property
    def short_name(self) -> str:
        return self._short_name

    @property
    def full_name(self) -> str:
        return self._full_name


# --------------- Snapshot type alias ---------------


def test_snapshot_type_alias_is_dict_str_any() -> None:
    assert Snapshot == dict[str, Any]


# --------------- Metadatable.__init__ ---------------


def test_init_with_none_metadata() -> None:
    m = Metadatable(metadata=None)
    assert m.metadata == {}


def test_init_with_no_arguments() -> None:
    m = Metadatable()
    assert m.metadata == {}


def test_init_with_metadata() -> None:
    m = Metadatable(metadata={"key": "value"})
    assert m.metadata == {"key": "value"}


# --------------- snapshot without / with metadata ---------------


def test_snapshot_without_metadata_returns_base_only() -> None:
    m = Metadatable()
    snap = m.snapshot()
    assert snap == {}
    assert "metadata" not in snap


def test_snapshot_with_metadata_includes_metadata_key() -> None:
    m = Metadatable(metadata={"x": 1})
    snap = m.snapshot()
    assert "metadata" in snap
    assert snap["metadata"] == {"x": 1}


def test_snapshot_metadata_removed_after_clear() -> None:
    m = Metadatable(metadata={"a": 1})
    assert "metadata" in m.snapshot()
    m.metadata.clear()
    assert "metadata" not in m.snapshot()


# --------------- snapshot_base default ---------------


def test_snapshot_base_default_returns_empty_dict() -> None:
    m = Metadatable()
    assert m.snapshot_base() == {}
    assert m.snapshot_base(update=True) == {}
    assert m.snapshot_base(params_to_skip_update=["p1"]) == {}


# --------------- load_metadata deep_update ---------------


def test_load_metadata_deep_updates_nested_dicts() -> None:
    m = Metadatable(metadata={"outer": {"a": 1, "b": 2}})
    m.load_metadata({"outer": {"b": 99, "c": 3}})
    assert m.metadata == {"outer": {"a": 1, "b": 99, "c": 3}}


def test_load_metadata_adds_new_top_level_keys() -> None:
    m = Metadatable(metadata={"first": 1})
    m.load_metadata({"second": 2})
    assert m.metadata == {"first": 1, "second": 2}


# --------------- MetadatableWithName ---------------


def test_metadatable_with_name_has_abstract_methods() -> None:
    # MetadatableWithName uses @abstractmethod for static analysis;
    # verify the property descriptors are marked abstract.
    for attr_name in ("short_name", "full_name"):
        descriptor = getattr(MetadatableWithName, attr_name)
        assert isinstance(descriptor, property)
        assert getattr(descriptor.fget, "__isabstractmethod__", False)


def test_concrete_metadatable_with_name_short_name() -> None:
    obj = ConcreteMetadatableWithName("sensor")
    assert obj.short_name == "sensor"


def test_concrete_metadatable_with_name_full_name() -> None:
    obj = ConcreteMetadatableWithName("sensor", full="instrument_sensor")
    assert obj.full_name == "instrument_sensor"


def test_concrete_metadatable_with_name_inherits_metadata() -> None:
    obj = ConcreteMetadatableWithName("s", metadata={"cal": True})
    assert obj.metadata == {"cal": True}
    snap = obj.snapshot()
    assert snap["metadata"] == {"cal": True}


def test_concrete_metadatable_with_name_snapshot_no_metadata() -> None:
    obj = ConcreteMetadatableWithName("s")
    snap = obj.snapshot()
    assert snap == {}
    assert "metadata" not in snap
