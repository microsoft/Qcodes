import pytest

from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.dependencies import InterDependencies


@pytest.fixture
def some_paramspecs():
    """
    Some different paramspecs for testing. The idea is that we just add a
    new group of paramspecs as the need arises
    """

    groups = {}

    # A valid group. Corresponding to a heatmap with a text label at each point
    first = {}
    first['ps1'] = ParamSpec('ps1', paramtype='numeric', label='Raw Data 1',
                             unit='V')
    first['ps2'] = ParamSpec('ps2', paramtype='array', label='Raw Data 2',
                             unit='V')
    first['ps3'] = ParamSpec('ps3', paramtype='text', label='Axis 1',
                             unit='', inferred_from=[first['ps1']])
    first['ps4'] = ParamSpec('ps4', paramtype='numeric', label='Axis 2',
                             unit='V', inferred_from=[first['ps2']])
    first['ps5'] = ParamSpec('ps5', paramtype='numeric', label='Signal',
                             unit='Conductance',
                             depends_on=[first['ps3'], first['ps4']])
    first['ps6'] = ParamSpec('ps6', paramtype='text', label='Goodness',
                             unit='', depends_on=[first['ps3'], first['ps4']])

    groups[1] = first

    return groups


def test_serialization_and_back(some_paramspecs):

    idp = InterDependencies(*some_paramspecs[1].values())
    desc = RunDescriber(interdeps=idp)

    ser_desc = desc.serialize()

    new_desc = RunDescriber.deserialize(ser_desc)

    assert isinstance(new_desc, RunDescriber)


def test_yaml_creation(some_paramspecs):

    idp = InterDependencies(*some_paramspecs[1].values())
    desc = RunDescriber(interdeps=idp)

    yaml_str = desc.output_yaml()
    assert isinstance(yaml_str, str)
