import json
import re
from copy import deepcopy

import pytest
from networkx import NetworkXError

from qcodes.dataset.descriptions.dependencies import (
    FrozenInterDependencies_,
    IncompleteSubsetError,
    InterDependencies_,
)
from qcodes.dataset.descriptions.param_spec import ParamSpec
from qcodes.dataset.descriptions.versioning.converters import new_to_old, old_to_new
from qcodes.dataset.descriptions.versioning.v0 import InterDependencies
from qcodes.parameters import ParamSpecBase
from tests.common import error_caused_by


def test_wrong_input_raises() -> None:
    for pspecs in (
        ["p1", "p2", "p3"],
        [ParamSpec("p1", paramtype="numeric"), "p2"],
        ["p1", ParamSpec("p2", paramtype="text")],
    ):
        with pytest.raises(ValueError):
            InterDependencies(pspecs)  # type: ignore[arg-type]


def test_init(some_paramspecbases) -> None:
    """
    Assert that the init functions correctly sets up the object.
    Assert via the public-facing methods.
    """

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps1 = InterDependencies_(dependencies={ps1: (ps2,)})
    idps2 = InterDependencies_(dependencies={ps1: (ps2, ps2, ps2)})

    assert idps1 == idps2
    assert idps1.what_depends_on(ps2) == (ps1,)
    assert idps1.what_is_inferred_from(ps2) == ()
    assert idps1.top_level_parameters == (ps1,)

    idps1 = InterDependencies_(dependencies={ps1: (ps2, ps3)})
    idps2 = InterDependencies_(dependencies={ps1: (ps3, ps2)})

    assert idps1.what_depends_on(ps2) == (ps1,)
    assert idps1.what_depends_on(ps3) == (ps1,)
    assert idps1.top_level_parameters == (ps1,)
    assert idps2.top_level_parameters == (ps1,)

    idps = InterDependencies_(dependencies={ps1: (ps3, ps2), ps4: (ps3,)})
    assert set(idps.what_depends_on(ps3)) == {ps1, ps4}
    assert idps.top_level_parameters == (ps1, ps4)


def test_init_validation_raises(some_paramspecbases) -> None:
    (ps1, ps2, ps3, ps4) = some_paramspecbases

    # First test validation of trees invalid in their own right

    invalid_trees = (
        [ps1, ps2],
        {"ps1": "ps2"},
        {ps1: "ps2"},
        {ps1: ("ps2",)},
        {ps1: (ps2,), ps2: (ps1,)},
    )
    causes = (
        "ParamSpecTree must be a dict",
        "ParamSpecTree must have ParamSpecs as keys",
        "ParamSpecTree must have tuple values",
        "ParamSpecTree can only have tuples of ParamSpecs as values",
        "ParamSpecTree can not have cycles",
    )

    for tree, cause in zip(invalid_trees, causes):
        with pytest.raises(ValueError, match="Invalid dependencies") as ei:
            InterDependencies_(dependencies=tree, inferences={})  # type: ignore[arg-type]

        assert error_caused_by(ei, cause=cause)

    for tree, cause in zip(invalid_trees, causes):
        with pytest.raises(ValueError, match="Invalid inferences") as ei:
            InterDependencies_(dependencies={}, inferences=tree)  # type: ignore[arg-type]

        assert error_caused_by(ei, cause=cause)

    with pytest.raises(ValueError, match="Invalid standalones") as ei:
        InterDependencies_(standalones=("ps1", "ps2"))  # type: ignore[arg-type]

    assert error_caused_by(ei, cause="Standalones must be a sequence of ParamSpecs")

    # Now test trees that are invalid together

    invalid_trees_2 = [
        {"deps": {ps1: (ps2, ps3)}, "inffs": {ps2: (ps4, ps1)}},
    ]
    for inv in invalid_trees_2:
        with pytest.raises(ValueError, match="already exists"):
            InterDependencies_(
                dependencies=inv["deps"],  # type: ignore[arg-type]
                inferences=inv["inffs"],  # type: ignore[arg-type]
            )


def test_to_dict(some_paramspecbases) -> None:
    def tester(idps) -> None:
        ser = idps._to_dict()
        json.dumps(ser)
        idps_deser = InterDependencies_._from_dict(ser)
        assert idps == idps_deser

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps = InterDependencies_(standalones=(ps1, ps2), dependencies={ps3: (ps4,)})
    tester(idps)

    idps = InterDependencies_(standalones=(ps1, ps2, ps3, ps4))
    tester(idps)

    idps = InterDependencies_(
        dependencies={ps1: (ps2, ps3)}, inferences={ps2: (ps4,), ps3: (ps4,)}
    )
    tester(idps)


def test_old_to_new_and_back(some_paramspecs) -> None:
    idps_old = InterDependencies(*some_paramspecs[1].values())
    idps_new = old_to_new(idps_old)

    assert new_to_old(idps_new) == idps_old


def test_old_to_new(some_paramspecs) -> None:
    ps1 = some_paramspecs[1]["ps1"]
    ps2 = some_paramspecs[1]["ps2"]
    ps3 = some_paramspecs[1]["ps3"]
    ps4 = some_paramspecs[1]["ps4"]
    ps5 = some_paramspecs[1]["ps5"]
    ps6 = some_paramspecs[1]["ps6"]

    idps_old = InterDependencies(ps1, ps2, ps3)

    idps_new = old_to_new(idps_old)

    ps1_base = ps1.base_version()
    ps2_base = ps2.base_version()
    ps3_base = ps3.base_version()
    ps4_base = ps4.base_version()
    ps5_base = ps5.base_version()
    ps6_base = ps6.base_version()

    assert idps_new.dependencies == {}
    assert idps_new.inferences == {ps3_base: (ps1_base,)}
    assert idps_new.standalones == {ps2_base}
    paramspecs = (ps1_base, ps2_base, ps3_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs}

    idps_old = InterDependencies(ps2, ps4, ps1, ps2, ps3, ps5, ps6)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == {
        ps5_base: (ps3_base, ps4_base),
        ps6_base: (ps3_base, ps4_base),
    }
    assert idps_new.inferences == {ps3_base: (ps1_base,), ps4_base: (ps2_base,)}
    assert idps_new.standalones == set()
    paramspecs2 = (ps1_base, ps2_base, ps3_base, ps4_base, ps5_base, ps6_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs2}

    idps_old = InterDependencies(ps1, ps2)

    idps_new = old_to_new(idps_old)

    assert idps_new.dependencies == {}
    assert idps_new.inferences == {}
    assert idps_new.standalones == {ps1_base, ps2_base}
    paramspecs3 = (ps1_base, ps2_base)
    assert idps_new._id_to_paramspec == {ps.name: ps for ps in paramspecs3}


def test_new_to_old(some_paramspecbases) -> None:
    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps_new = InterDependencies_(dependencies={ps1: (ps2, ps3)}, standalones=(ps4,))

    paramspec1 = ParamSpec(
        name=ps1.name,
        paramtype=ps1.type,
        label=ps1.label,
        unit=ps1.unit,
        depends_on=[ps2.name, ps3.name],
    )
    paramspec2 = ParamSpec(
        name=ps2.name, paramtype=ps2.type, label=ps2.label, unit=ps2.unit
    )
    paramspec3 = ParamSpec(
        name=ps3.name, paramtype=ps3.type, label=ps3.label, unit=ps3.unit
    )
    paramspec4 = ParamSpec(
        name=ps4.name, paramtype=ps4.type, label=ps4.label, unit=ps4.unit
    )
    idps_old_expected = InterDependencies(
        paramspec2, paramspec3, paramspec1, paramspec4
    )

    assert new_to_old(idps_new) == idps_old_expected

    idps_new = InterDependencies_(inferences={ps1: (ps2, ps3)}, standalones=(ps4,))

    paramspec1 = ParamSpec(
        name=ps1.name,
        paramtype=ps1.type,
        label=ps1.label,
        unit=ps1.unit,
        inferred_from=[ps2.name, ps3.name],
    )
    paramspec2 = ParamSpec(
        name=ps2.name, paramtype=ps2.type, label=ps2.label, unit=ps2.unit
    )
    paramspec3 = ParamSpec(
        name=ps3.name, paramtype=ps3.type, label=ps3.label, unit=ps3.unit
    )
    paramspec4 = ParamSpec(
        name=ps4.name, paramtype=ps4.type, label=ps4.label, unit=ps4.unit
    )
    idps_old_expected = InterDependencies(
        paramspec2, paramspec3, paramspec1, paramspec4
    )

    assert new_to_old(idps_new) == idps_old_expected


def test_validate_subset(some_paramspecbases) -> None:
    ps1, ps2, ps3, ps4 = some_paramspecbases

    idps = InterDependencies_(
        dependencies={ps1: (ps2, ps3)}, inferences={ps2: (ps4,), ps3: (ps4,)}
    )

    idps.validate_subset((ps4,))
    idps.validate_subset((ps2, ps4))
    idps.validate_subset((ps2, ps3, ps4))
    idps.validate_subset(())
    idps.validate_subset([])

    with pytest.raises(IncompleteSubsetError) as exc_info1:
        idps.validate_subset((ps1,))
    assert exc_info1.value._subset_params == {"psb1"}
    assert exc_info1.value._missing_params == {"psb2", "psb3", "psb4"}

    with pytest.raises(IncompleteSubsetError) as exc_info2:
        idps.validate_subset((ps1, ps2, ps4))
    assert exc_info2.value._subset_params == {"psb1", "psb2", "psb4"}
    assert exc_info2.value._missing_params == {"psb3"}

    with pytest.raises(IncompleteSubsetError) as exc_info3:
        idps.validate_subset((ps3,))
    assert exc_info3.value._subset_params == {"psb3"}
    assert exc_info3.value._missing_params == {"psb4"}

    with pytest.raises(IncompleteSubsetError) as exc_info4:
        idps2 = InterDependencies_(
            dependencies={ps1: (ps2, ps3)}, inferences={ps3: (ps4,)}
        )
        idps2.validate_subset((ps1, ps2, ps3))
    assert exc_info4.value._subset_params == {"psb1", "psb2", "psb3"}
    assert exc_info4.value._missing_params == {"psb4"}

    with pytest.raises(NetworkXError, match="ps42"):
        ps42 = ParamSpecBase("ps42", paramtype="text", label="", unit="it")
        idps.validate_subset((ps2, ps42, ps4))


def test_extend(some_paramspecbases) -> None:
    ps1, ps2, ps3, _ = some_paramspecbases

    idps = InterDependencies_(standalones=(ps1, ps2))

    idps_ext = idps.extend(dependencies={ps1: (ps3,)})
    idps_expected = InterDependencies_(standalones=(ps2,), dependencies={ps1: (ps3,)})
    assert idps_ext == idps_expected

    assert idps_ext is not idps
    assert idps_ext.graph is not idps.graph

    idps = InterDependencies_(standalones=(ps2,))
    idps_ext = idps.extend(dependencies={ps1: (ps2,)})
    idps_expected = InterDependencies_(dependencies={ps1: (ps2,)})
    assert idps_ext == idps_expected

    idps = InterDependencies_(dependencies={ps1: (ps2,)})
    idps_ext = idps.extend(dependencies={ps1: (ps2, ps3)})
    idps_expected = InterDependencies_(dependencies={ps1: (ps2, ps3)})
    assert idps_ext == idps_expected

    idps = InterDependencies_()
    idps_ext = idps.extend(standalones=(ps1, ps2))
    idps_expected = InterDependencies_(standalones=(ps2, ps1))
    assert idps_ext == idps_expected

    ps_nu = deepcopy(ps1)
    ps_nu.unit += "/s"
    idps = InterDependencies_(standalones=(ps1,))
    with pytest.raises(ValueError, match="already exists"):
        idps_ext = idps.extend(standalones=(ps_nu,))
    ps_nu.name = "psbnu"
    idps_ext = idps.extend(standalones=(ps_nu,))

    idps_expected = InterDependencies_(standalones=(ps_nu, ps1))
    assert idps_ext == idps_expected

    idps = InterDependencies_(dependencies={ps1: (ps2,)})
    with pytest.raises(ValueError, match="already exists"):
        idps_ext = idps.extend(inferences={ps2: (ps1,)})


def test_remove(some_paramspecbases) -> None:
    ps1, ps2, ps3, ps4 = some_paramspecbases

    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)}, inferences={ps2: (ps4,)})
    idps_rem = idps.remove(ps1)
    idps_expected = InterDependencies_(inferences={ps2: (ps4,)}, standalones=(ps3,))
    assert idps_rem == idps_expected

    for p in [ps4, ps2, ps3]:
        match = re.escape(f"Cannot remove {p.name}, other parameters")
        with pytest.raises(ValueError, match=match):
            idps_rem = idps.remove(p)

    idps = InterDependencies_(dependencies={ps1: (ps3,)}, inferences={ps2: (ps4,)})
    idps_rem = idps.remove(ps2)
    idps_expected = InterDependencies_(dependencies={ps1: (ps3,)}, standalones=(ps4,))

    assert idps_rem == idps_expected

    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)}, standalones=(ps4,))
    idps_rem = idps.remove(ps4)
    idps_expected = InterDependencies_(dependencies={ps1: (ps2, ps3)})
    assert idps_rem == idps_expected

    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)}, standalones=(ps4,))
    idps_rem = idps.remove(ps1)
    idps_expected = InterDependencies_(standalones=(ps2, ps3, ps4))
    assert idps_rem == idps_expected


def test_equality_old(some_paramspecs) -> None:
    # TODO: make this more fancy with itertools

    ps1 = some_paramspecs[1]["ps1"]
    ps2 = some_paramspecs[1]["ps2"]
    ps3 = some_paramspecs[1]["ps3"]
    ps4 = some_paramspecs[1]["ps4"]
    ps5 = some_paramspecs[1]["ps5"]
    ps6 = some_paramspecs[1]["ps6"]

    assert InterDependencies(ps1, ps2, ps3) == InterDependencies(ps3, ps2, ps1)
    assert InterDependencies(ps1, ps6, ps3) == InterDependencies(ps3, ps6, ps1)
    assert InterDependencies(ps4, ps5, ps3) == InterDependencies(ps3, ps4, ps5)


def test_non_dependents() -> None:
    ps1 = ParamSpecBase("ps1", paramtype="numeric", label="Raw Data 1", unit="V")
    ps2 = ParamSpecBase("ps2", paramtype="array", label="Raw Data 2", unit="V")
    ps3 = ParamSpecBase("ps3", paramtype="text", label="Axis 1", unit="")
    ps4 = ParamSpecBase("ps4", paramtype="numeric", label="Axis 2", unit="V")
    ps5 = ParamSpecBase("ps5", paramtype="numeric", label="Signal", unit="Conductance")
    ps6 = ParamSpecBase("ps6", paramtype="text", label="Goodness", unit="")

    idps1 = InterDependencies_(
        dependencies={ps5: (ps3, ps4), ps6: (ps3, ps4)},
        inferences={ps4: (ps2,), ps3: (ps1,)},
    )

    assert idps1.top_level_parameters == (ps5, ps6)

    idps2 = InterDependencies_(dependencies={ps2: (ps1,)})

    assert idps2.top_level_parameters == (ps2,)

    idps3 = InterDependencies_(dependencies={ps6: (ps1,)}, standalones=(ps2,))

    assert idps3.top_level_parameters == (ps2, ps6)


def test_collect_related(
    some_paramspecbases: tuple[
        ParamSpecBase, ParamSpecBase, ParamSpecBase, ParamSpecBase
    ],
) -> None:
    """
    Test that find_all_parameters_in_tree collects all parameters
    """

    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps1 = InterDependencies_(dependencies={ps1: (ps2,)})

    collected_params = idps1.find_all_parameters_in_tree(ps1)

    assert collected_params == {ps1, ps2}

    idps2 = InterDependencies_(dependencies={ps1: (ps2,)}, inferences={ps2: (ps3,)})
    collected_params = idps2.find_all_parameters_in_tree(ps1)
    assert collected_params == {ps1, ps2, ps3}

    idps3 = InterDependencies_(
        dependencies={ps1: (ps2,)}, inferences={ps2: (ps3,), ps4: (ps1,)}
    )
    collected_params = idps3.find_all_parameters_in_tree(ps1)
    assert collected_params == {ps1, ps2, ps4, ps3}

    idps4 = InterDependencies_(dependencies={ps1: (ps2,)}, standalones=(ps3, ps4))
    assert idps4.find_all_parameters_in_tree(ps1) == {ps1, ps2}
    assert idps4.find_all_parameters_in_tree(ps3) == {ps3}
    assert idps4.find_all_parameters_in_tree(ps4) == {ps4}
    assert idps4.top_level_parameters == (ps1, ps3, ps4)

    idps5 = InterDependencies_(dependencies={ps1: (ps2,)}, inferences={ps3: (ps4,)})
    assert idps5.find_all_parameters_in_tree(ps1) == {ps1, ps2}
    assert idps5.find_all_parameters_in_tree(ps3) == {ps3, ps4}
    assert idps5.find_all_parameters_in_tree(ps4) == {ps3, ps4}
    assert idps5.top_level_parameters == (ps1, ps3)


def test_collect_related__complex(
    some_paramspecbases: tuple[
        ParamSpecBase, ParamSpecBase, ParamSpecBase, ParamSpecBase
    ],
) -> None:
    (ps1, ps2, ps3, ps4) = some_paramspecbases
    idps1 = InterDependencies_(dependencies={ps1: (ps2,)}, inferences={ps2: (ps3, ps4)})
    assert idps1.top_level_parameters == (ps1,)
    assert idps1.find_all_parameters_in_tree(ps1) == {ps1, ps2, ps3, ps4}

    idps2 = InterDependencies_(dependencies={ps1: (ps2,)}, inferences={ps3: (ps2,)})
    assert idps2.top_level_parameters == (ps1,)
    assert idps2.find_all_parameters_in_tree(ps1) == {ps1, ps2, ps3}


def test_all_parameters_in_tree_by_group_raises_on_non_top_level(some_paramspecbases):
    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps = InterDependencies_(dependencies={ps1: (ps2,)}, inferences={ps2: (ps3,)})

    with pytest.raises(
        ValueError, match=f"Parameter '{ps4.name}' is not part of the graph."
    ):
        idps.all_parameters_in_tree_by_group(ps4)

    with pytest.raises(
        ValueError, match=f"Parameter '{ps4.name}' is not part of the graph."
    ):
        idps.find_all_parameters_in_tree(ps4)


def test_dependency_on_middle_parameter(
    some_paramspecbases: tuple[
        ParamSpecBase, ParamSpecBase, ParamSpecBase, ParamSpecBase
    ],
) -> None:
    """
    Test that a dependency on a middle parameter is correctly handled.
    """
    (ps1, ps2, ps3, ps4) = some_paramspecbases

    idps = InterDependencies_(dependencies={ps1: (ps2,)}, inferences={ps2: (ps3,)})

    idps.add_inferences({ps4: (ps2,)})

    # navively one might expect that ps4 is a top level parameter
    # since it has no in edges. However, since we include inferred parameters
    # in both directions, ps4 is actually a member of the tree for ps1
    assert idps.top_level_parameters == (ps1,)
    assert idps.find_all_parameters_in_tree(ps1) == {ps1, ps2, ps3, ps4}


def test_frozen_interdependencies(some_paramspecbases) -> None:
    ps1, ps2, ps3, ps4 = some_paramspecbases
    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)}, inferences={ps2: (ps4,)})

    frozen = FrozenInterDependencies_(idps)

    assert frozen.dependencies == idps.dependencies
    assert frozen.inferences == idps.inferences
    assert frozen.standalones == idps.standalones
    assert frozen.top_level_parameters == idps.top_level_parameters

    # Test immutability
    with pytest.raises(TypeError, match="FrozenInterDependencies_ is immutable"):
        frozen.add_dependencies({ps4: (ps1,)})

    with pytest.raises(TypeError, match="FrozenInterDependencies_ is immutable"):
        frozen.add_inferences({ps4: (ps1,)})

    with pytest.raises(TypeError, match="FrozenInterDependencies_ is immutable"):
        frozen.add_standalones((ps4,))

    with pytest.raises(TypeError, match="FrozenInterDependencies_ is immutable"):
        frozen.remove(ps1)

    with pytest.raises(TypeError, match="FrozenInterDependencies_ is immutable"):
        frozen.add_paramspecs((ps1,))

    # Test extend returns InterDependencies_ (mutable)
    ps5 = ParamSpecBase("psb5", "numeric", "number", "")
    extended = frozen.extend(standalones=(ps5,))
    assert isinstance(extended, InterDependencies_)
    assert not isinstance(extended, FrozenInterDependencies_)
    assert ps5 in extended.standalones

    # Test caching of properties
    # Access properties to trigger caching
    _ = frozen.dependencies
    _ = frozen.inferences
    _ = frozen.standalones
    _ = frozen.top_level_parameters

    assert frozen._dependencies_cache is not None
    assert frozen._inferences_cache is not None
    assert frozen._standalones_cache is not None
    assert frozen._top_level_parameters_cache is not None


def test_frozen_from_dict(some_paramspecbases) -> None:
    ps1, ps2, ps3, _ = some_paramspecbases
    idps = InterDependencies_(dependencies={ps1: (ps2, ps3)})
    ser = idps._to_dict()

    frozen = FrozenInterDependencies_._from_dict(ser)
    assert isinstance(frozen, FrozenInterDependencies_)
    assert frozen == FrozenInterDependencies_(idps)
