import random
from typing import TYPE_CHECKING

import pytest

from qcodes.dataset import Measurement
from qcodes.parameters import ManualParameter
from tests.dataset.conftest import SimpleMultiParam

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def control_parameters() -> (
    "Generator[tuple[ManualParameter, ManualParameter, ManualParameter], None, None]"
):
    comp1 = ManualParameter("comp1")
    comp2 = ManualParameter("comp2")
    control1 = ManualParameter("control1")

    comp1.is_controlled_by.add(control1)
    comp2.is_controlled_by.add(control1)
    control1.has_control_of.add(comp1)
    control1.has_control_of.add(comp2)
    yield control1, comp1, comp2


@pytest.fixture
def dependent_parameters() -> (
    "Generator[tuple[ManualParameter, ManualParameter, ManualParameter], None, None]"
):
    indep1 = ManualParameter("indep1")
    indep2 = ManualParameter("indep2")
    dep1 = ManualParameter("dep1")

    dep1.depends_on.add(indep1)
    dep1.depends_on.add(indep2)
    yield dep1, indep1, indep2


def test_registering_control_param_registers_components(control_parameters) -> None:
    control1, comp1, comp2 = control_parameters
    meas = Measurement()
    meas.register_parameter(control1)

    assert comp1 in meas._registered_parameters
    assert comp2 in meas._registered_parameters


def test_registering_component_param_registers_control(control_parameters) -> None:
    control1, comp1, comp2 = control_parameters
    meas = Measurement()
    meas.register_parameter(comp1)

    assert control1 in meas._registered_parameters
    assert comp2 in meas._registered_parameters


def test_registering_dependent_param_registers_indeps(dependent_parameters) -> None:
    dep1, indep1, indep2 = dependent_parameters
    meas = Measurement()
    meas.register_parameter(dep1)

    assert indep1 in meas._registered_parameters
    assert indep2 in meas._registered_parameters
    # Note, registering indep1 is not expected to also register dep1 automatically


def test_registering_chain_of_control_parameters(control_parameters) -> None:
    control1, comp1, comp2 = control_parameters
    comp11 = ManualParameter("comp11")
    comp12 = ManualParameter("comp12")
    comp21 = ManualParameter("comp21")
    comp22 = ManualParameter("comp22")

    new_controls = {comp1: [comp11, comp12], comp2: [comp21, comp22]}
    for comp_n, comp_mms in new_controls.items():
        for comp_mm in comp_mms:
            comp_n.has_control_of.add(comp_mm)
            comp_mm.is_controlled_by.add(comp_n)

    meas = Measurement()
    meas.register_parameter(control1)

    assert comp1 in meas._registered_parameters
    assert comp2 in meas._registered_parameters
    assert comp11 in meas._registered_parameters
    assert comp12 in meas._registered_parameters
    assert comp21 in meas._registered_parameters
    assert comp22 in meas._registered_parameters


def test_registering_dependent_param_with_setpoints(dependent_parameters) -> None:
    dep1, indep1, indep2 = dependent_parameters
    setpoints1 = ManualParameter("setpoints1")
    setpoints2 = ManualParameter("setpoints2")
    meas = Measurement()
    meas.register_parameter(dep1, setpoints=[setpoints1, setpoints2])

    assert indep1 in meas._registered_parameters
    assert indep2 in meas._registered_parameters
    assert setpoints1 in meas._registered_parameters
    assert setpoints2 in meas._registered_parameters

    dependency_tree = meas._interdeps.dependencies
    assert len(dependency_tree) == 1
    assert dep1.param_spec in dependency_tree.keys()

    # Ensure that order in the dependency spec tree is preserved
    # Explicit Setpoints first, then internal depends_on parameters
    # In the case where setpoints have equal dimension, this is the only
    # way to preserve the correct relationship of the multidimensional data
    assert dependency_tree[dep1.param_spec][0] == setpoints1.param_spec
    assert dependency_tree[dep1.param_spec][1] == setpoints2.param_spec
    assert dependency_tree[dep1.param_spec][2] == indep1.param_spec
    assert dependency_tree[dep1.param_spec][3] == indep2.param_spec


def test_registering_multi_param() -> None:
    multi = SimpleMultiParam("multi", npts=random.randint(1, 11))

    meas = Measurement()
    meas.register_parameter(multi)
    dependency_tree = meas._interdeps.dependencies

    assert len(meas._registered_parameters) == 4
    assert len(dependency_tree) == 2

    for param in multi.register_parameters:
        assert param in meas._registered_parameters
        for setpoints in param.setpoints:
            assert setpoints in meas._registered_parameters
            assert dependency_tree[param.param_spec][0] == setpoints.param_spec

    # no setpoints
    multi = SimpleMultiParam("multi", npts=0)

    meas = Measurement()
    meas.register_parameter(multi)
    dependency_tree = meas._interdeps.dependencies

    assert len(meas._registered_parameters) == 2
    assert len(dependency_tree) == 0

    for param in multi.register_parameters:
        assert param in meas._registered_parameters


def test_registering_controlled_multi_param(control_parameters) -> None:
    control1, comp1, comp2 = control_parameters
    multi = SimpleMultiParam("multi")
    multi.has_control_of.add(comp1)
    multi.has_control_of.add(comp2)
    control1.has_control_of.add(multi)

    meas = Measurement()
    meas.register_parameter(multi)
    inferences_tree = meas._interdeps.inferences

    assert len(meas._registered_parameters) == 7
    assert len(inferences_tree) == 4

    for param in multi.register_parameters:
        assert param in meas._registered_parameters
        for setpoints in param.setpoints:
            assert setpoints in meas._registered_parameters
            assert inferences_tree[param.param_spec][0] == control1.param_spec
            assert param.param_spec in inferences_tree[comp1.param_spec]
            assert param.param_spec in inferences_tree[comp2.param_spec]

    assert comp1 in meas._registered_parameters
    assert comp2 in meas._registered_parameters
    assert control1 in meas._registered_parameters

    assert control1.param_spec in inferences_tree[comp1.param_spec]
    assert control1.param_spec in inferences_tree[comp2.param_spec]


def test_registering_dependent_multi_param(dependent_parameters) -> None:
    _, indep1, indep2 = dependent_parameters
    multi = SimpleMultiParam("multi")
    multi.depends_on.add(indep1)
    multi.depends_on.add(indep2)

    meas = Measurement()
    meas.register_parameter(multi)
    dependency_tree = meas._interdeps.dependencies

    assert len(meas._registered_parameters) == 6
    assert len(dependency_tree) == 2

    for param in multi.register_parameters:
        assert param in meas._registered_parameters
        for setpoints in param.setpoints:
            assert setpoints in meas._registered_parameters
            assert dependency_tree[param.param_spec] == (
                indep1.param_spec,
                indep2.param_spec,
                setpoints.param_spec,
            )

    assert indep1 in meas._registered_parameters
    assert indep2 in meas._registered_parameters


def test_registering_param_dependent_on_multi_param(dependent_parameters) -> None:
    dep1, *_ = dependent_parameters
    multi = SimpleMultiParam("multi", npts=random.randint(1, 11))
    # Chained dependency
    dep1.depends_on.add(multi)

    meas = Measurement()
    with pytest.raises(
        ValueError,
        match=r"Paramspec a both depends on \['sp_a'\] and is depended upon by \['dep1'\]",
    ):
        meas.register_parameter(dep1)


def test_registering_param_dependent_on_multi_param_no_sp(dependent_parameters) -> None:
    dep1, indep1, indep2 = dependent_parameters
    multi = SimpleMultiParam("multi", npts=0)
    dep1.depends_on.add(multi)

    meas = Measurement()
    meas.register_parameter(dep1)
    dependency_tree = meas._interdeps.dependencies

    assert len(meas._registered_parameters) == 5
    assert len(dependency_tree) == 1

    for param in multi.register_parameters:
        assert param in meas._registered_parameters

    assert dep1 in meas._registered_parameters
    assert indep1 in meas._registered_parameters
    assert indep2 in meas._registered_parameters

    assert dependency_tree[dep1.param_spec] == (
        indep1.param_spec,
        indep2.param_spec,
        *(param.param_spec for param in multi.register_parameters),
    )
