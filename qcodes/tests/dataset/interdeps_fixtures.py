import pytest

from qcodes.dataset.descriptions.param_spec import (ParamSpec, ParamSpecBase)
from qcodes.dataset.descriptions.dependencies import InterDependencies_


@pytest.fixture
def some_paramspecbases():

    psb1 = ParamSpecBase('psb1', paramtype='text', label='blah', unit='')
    psb2 = ParamSpecBase('psb2', paramtype='array', label='', unit='V')
    psb3 = ParamSpecBase('psb3', paramtype='array', label='', unit='V')
    psb4 = ParamSpecBase('psb4', paramtype='numeric', label='number', unit='')

    return (psb1, psb2, psb3, psb4)


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


@pytest.fixture
def some_interdeps():
    """
    Some different InterDependencies_ objects for testing
    """
    idps_list = []
    ps1 = ParamSpecBase('ps1', paramtype='numeric', label='Raw Data 1',
                        unit='V')
    ps2 = ParamSpecBase('ps2', paramtype='array', label='Raw Data 2',
                        unit='V')
    ps3 = ParamSpecBase('ps3', paramtype='text', label='Axis 1',
                        unit='')
    ps4 = ParamSpecBase('ps4', paramtype='numeric', label='Axis 2',
                        unit='V')
    ps5 = ParamSpecBase('ps5', paramtype='numeric', label='Signal',
                        unit='Conductance')
    ps6 = ParamSpecBase('ps6', paramtype='text', label='Goodness',
                    unit='')

    idps = InterDependencies_(dependencies={ps5: (ps3, ps4), ps6: (ps3, ps4)},
                              inferences={ps4: (ps2,), ps3: (ps1,)})

    idps_list.append(idps)

    ps1 = ParamSpecBase('ps1', paramtype='numeric',
                        label='setpoint', unit='Hz')
    ps2 = ParamSpecBase('ps2', paramtype='numeric', label='signal',
                        unit='V')
    idps = InterDependencies_(dependencies={ps2: (ps1,)})

    idps_list.append(idps)

    return idps_list
