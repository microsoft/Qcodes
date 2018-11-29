import pytest
import numpy as np
from qcodes.dataset.param_spec import ParamSpec

# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import dataset
# pylint: enable=unused-import


@pytest.fixture
def scalar_dataset(dataset):
    n_params = 3
    n_rows = 10**3
    params_indep = [ParamSpec(f'param_{i}',
                              'numeric',
                              label=f'param_{i}',
                              unit='V')
                    for i in range(n_params)]
    params = params_indep + [ParamSpec(f'param_{n_params}',
                                       'numeric',
                                       label=f'param_{n_params}',
                                       unit='Ohm',
                                       depends_on=params_indep)]
    for p in params:
        dataset.add_parameter(p)
    dataset.add_results([{p.name: np.random.rand(1)[0] for p in params}
                         for _ in range(n_rows)])
    dataset.mark_complete()
    yield dataset


