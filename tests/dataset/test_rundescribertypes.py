"""
Tests for qcodes.dataset.descriptions.versioning.rundescribertypes.

Verifies the TypedDict classes, inheritance relationships, type aliases,
and the RunDescriberDicts union.
"""

from __future__ import annotations

import typing

from typing_extensions import get_annotations, get_original_bases

from qcodes.dataset.descriptions.versioning.rundescribertypes import (
    InterDependencies_Dict,
    InterDependenciesDict,
    RunDescriberDicts,
    RunDescriberV0Dict,
    RunDescriberV1Dict,
    RunDescriberV2Dict,
    RunDescriberV3Dict,
    Shapes,
)

# --------------- Shapes type alias ---------------


def test_shapes_type_alias() -> None:
    sample: Shapes = {"param": (1, 2, 3)}
    assert sample["param"] == (1, 2, 3)


# --------------- InterDependenciesDict ---------------


def test_interdependencies_dict_instantiation() -> None:
    d: InterDependenciesDict = {"paramspecs": ()}
    assert d["paramspecs"] == ()


# --------------- InterDependencies_Dict ---------------


def test_interdependencies_underscore_dict_instantiation() -> None:
    d: InterDependencies_Dict = {
        "parameters": {},
        "dependencies": {},
        "inferences": {},
        "standalones": [],
    }
    assert d["parameters"] == {}
    assert d["standalones"] == []


# --------------- RunDescriberV0Dict ---------------


def test_v0_dict_instantiation() -> None:
    d: RunDescriberV0Dict = {
        "version": 0,
        "interdependencies": {"paramspecs": ()},
    }
    assert d["version"] == 0


# --------------- RunDescriberV1Dict ---------------


def test_v1_dict_instantiation() -> None:
    d: RunDescriberV1Dict = {
        "version": 1,
        "interdependencies": {
            "parameters": {},
            "dependencies": {},
            "inferences": {},
            "standalones": [],
        },
    }
    assert d["version"] == 1


# --------------- RunDescriberV2Dict inherits from V0 ---------------


def test_v2_dict_inherits_from_v0() -> None:
    # typing_extensions TypedDict flattens __bases__ to (dict,) at runtime;
    # verify structural inheritance via __orig_bases__ and annotations.
    assert RunDescriberV0Dict in get_original_bases(RunDescriberV2Dict)
    # V2 should contain all V0 keys plus its own
    v0_keys = set(get_annotations(RunDescriberV0Dict))
    v2_keys = set(get_annotations(RunDescriberV2Dict))
    assert v0_keys.issubset(v2_keys)


def test_v2_dict_instantiation() -> None:
    d: RunDescriberV2Dict = {
        "version": 2,
        "interdependencies": {"paramspecs": ()},
        "interdependencies_": {
            "parameters": {},
            "dependencies": {},
            "inferences": {},
            "standalones": [],
        },
    }
    assert d["version"] == 2
    assert "interdependencies_" in d


# --------------- RunDescriberV3Dict inherits from V2 ---------------


def test_v3_dict_inherits_from_v2() -> None:
    assert RunDescriberV2Dict in get_original_bases(RunDescriberV3Dict)
    v2_keys = set(get_annotations(RunDescriberV2Dict))
    v3_keys = set(get_annotations(RunDescriberV3Dict))
    assert v2_keys.issubset(v3_keys)


def test_v3_dict_inherits_from_v0_transitively() -> None:
    # V3 inherits from V2 which inherits from V0 — all V0 keys present
    v0_keys = set(get_annotations(RunDescriberV0Dict))
    v3_keys = set(get_annotations(RunDescriberV3Dict))
    assert v0_keys.issubset(v3_keys)


def test_v3_dict_instantiation() -> None:
    d: RunDescriberV3Dict = {
        "version": 3,
        "interdependencies": {"paramspecs": ()},
        "interdependencies_": {
            "parameters": {},
            "dependencies": {},
            "inferences": {},
            "standalones": [],
        },
        "shapes": {"x": (10,)},
    }
    assert d["version"] == 3
    assert d["shapes"] == {"x": (10,)}


def test_v3_dict_shapes_none() -> None:
    d: RunDescriberV3Dict = {
        "version": 3,
        "interdependencies": {"paramspecs": ()},
        "interdependencies_": {
            "parameters": {},
            "dependencies": {},
            "inferences": {},
            "standalones": [],
        },
        "shapes": None,
    }
    assert d["shapes"] is None


# --------------- RunDescriberDicts union ---------------


def test_rundescriber_dicts_includes_all_versions() -> None:
    args = typing.get_args(RunDescriberDicts)
    expected = {
        RunDescriberV0Dict,
        RunDescriberV1Dict,
        RunDescriberV2Dict,
        RunDescriberV3Dict,
    }
    assert set(args) == expected
