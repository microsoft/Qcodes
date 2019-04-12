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

    # a small, valid group
    second = {}
    second['ps1'] = ParamSpec('ps1', paramtype='numeric',
                              label='setpoint', unit='Hz')
    second['ps2'] = ParamSpec('ps2', paramtype='numeric', label='signal',
                              unit='V', depends_on=[second['ps1']])
    groups[2] = second

    return groups


def test_wrong_input_type_raises():

    for interdeps in ['interdeps', ['p1', 'p2'], 0]:

        with pytest.raises(ValueError):
            RunDescriber(interdeps=interdeps)


def test_equality(some_paramspecs):
    idp1 = InterDependencies(*some_paramspecs[1].values())
    idp2 = InterDependencies(*some_paramspecs[2].values())
    idp3 = InterDependencies(*some_paramspecs[1].values())

    desc_1 = RunDescriber(interdeps=idp1)
    desc_2 = RunDescriber(interdeps=idp2)
    desc_3 = RunDescriber(interdeps=idp3)

    assert desc_1 == desc_3
    assert desc_1 != desc_2
    assert desc_3 != desc_2


def test_serialization_dict_keys(some_paramspecs):
    idp = InterDependencies(*some_paramspecs[1].values())
    desc = RunDescriber(interdeps=idp)

    ser_desc = desc.serialize()
    assert list(ser_desc.keys()) == ['interdependencies']


def test_serialization_and_back(some_paramspecs):

    idp = InterDependencies(*some_paramspecs[1].values())
    desc = RunDescriber(interdeps=idp)

    ser_desc = desc.serialize()

    new_desc = RunDescriber.deserialize(ser_desc)

    assert isinstance(new_desc, RunDescriber)
    assert desc == new_desc


def test_yaml_creation_and_loading(some_paramspecs):

    try:
        YAML = RunDescriber._ruamel_importer()
    except ImportError:
        pytest.skip('No ruamel module installed, skipping test')

    yaml = YAML()

    for group in some_paramspecs.values():
        paramspecs = group.values()
        idp = InterDependencies(*paramspecs)
        desc = RunDescriber(interdeps=idp)

        yaml_str = desc.to_yaml()
        assert isinstance(yaml_str, str)
        ydict = dict(yaml.load(yaml_str))
        assert list(ydict.keys()) == ['interdependencies']

        new_desc = RunDescriber.from_yaml(yaml_str)
        assert new_desc == desc
