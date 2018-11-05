import pytest

from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.param_spec import ParamSpec


def test_wrong_input_raises():

    for pspecs in [['p1', 'p2', 'p3'],
                   [ParamSpec('p1', paramtype='numeric'), 'p2'],
                   ['p1', ParamSpec('p2', paramtype='text')]]:

        with pytest.raises(ValueError):
            InterDependencies(pspecs)
