import pytest
import numpy as np

from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.measurements import Measurement
from qcodes.tests.instrument_mocks import ArraySetPointParam, Multi2DSetPointParam
from qcodes.instrument.parameter import Parameter

# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import dataset, experiment
# pylint: enable=unused-import


@pytest.fixture
def scalar_dataset(dataset):
    n_params = 3
    n_rows = 10**3
    params_indep = [ParamSpecBase(f'param_{i}',
                                  'numeric',
                                  label=f'param_{i}',
                                  unit='V')
                    for i in range(n_params)]
    param_dep = ParamSpecBase(f'param_{n_params}',
                              'numeric',
                              label=f'param_{n_params}',
                              unit='Ohm')

    all_params = params_indep + [param_dep]

    idps = InterDependencies_(dependencies={param_dep: tuple(params_indep)})

    dataset.set_interdependencies(idps)
    dataset.mark_started()
    dataset.add_results([{p.name: np.int(n_rows*10*pn+i)
                          for pn, p in enumerate(all_params)}
                         for i in range(n_rows)])
    dataset.mark_completed()
    yield dataset


@pytest.fixture
def scalar_dataset_with_nulls(dataset):
    """
    A very simple dataset. A scalar is varied, and two parameters are measured
    one by one
    """
    sp = ParamSpecBase('setpoint', 'numeric')
    val1 = ParamSpecBase('first_value', 'numeric')
    val2 = ParamSpecBase('second_value', 'numeric')

    idps = InterDependencies_(dependencies={val1: (sp,), val2: (sp,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()

    dataset.add_results([{sp.name: 0, val1.name: 1},
                         {sp.name: 0, val2.name: 2}])
    dataset.mark_completed()
    yield dataset


@pytest.fixture(scope="function",
                params=["array", "numeric"])
def array_dataset(experiment, request):
    meas = Measurement()
    param = ArraySetPointParam()
    meas.register_parameter(param, paramtype=request.param)

    with meas.run() as datasaver:
        datasaver.add_result((param, param.get(),))
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()

@pytest.fixture(scope="function",
                params=["array", "numeric"])
def array_dataset_with_nulls(experiment, request):
    """
    A dataset where two arrays are measured, one as a function
    of two other (setpoint) arrays, the other as a function of just one
    of them
    """
    meas = Measurement()
    meas.register_custom_parameter('sp1', paramtype=request.param)
    meas.register_custom_parameter('sp2', paramtype=request.param)
    meas.register_custom_parameter('val1', paramtype=request.param,
                                   setpoints=('sp1', 'sp2'))
    meas.register_custom_parameter('val2', paramtype=request.param,
                                   setpoints=('sp1',))

    with meas.run() as datasaver:
        sp1_vals = np.arange(0, 5)
        sp2_vals = np.arange(5, 10)
        val1_vals = np.ones(5)
        val2_vals = np.zeros(5)
        datasaver.add_result(('sp1', sp1_vals),
                             ('sp2', sp2_vals),
                             ('val1', val1_vals))
        datasaver.add_result(('sp1', sp1_vals),
                             ('val2', val2_vals))
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function",
                params=["array", "numeric"])
def multi_dataset(experiment, request):
    meas = Measurement()
    param = Multi2DSetPointParam()

    meas.register_parameter(param, paramtype=request.param)

    with meas.run() as datasaver:
        datasaver.add_result((param, param.get(),))
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function")
def array_in_scalar_dataset(experiment):
    meas = Measurement()
    scalar_param = Parameter('scalarparam', set_cmd=None)
    param = ArraySetPointParam()
    meas.register_parameter(scalar_param)
    meas.register_parameter(param, setpoints=(scalar_param,),
                            paramtype='array')

    with meas.run() as datasaver:
        for i in range(1, 10):
            scalar_param.set(i)
            datasaver.add_result((scalar_param, scalar_param.get()),
                                 (param, param.get()))
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function")
def varlen_array_in_scalar_dataset(experiment):
    meas = Measurement()
    scalar_param = Parameter('scalarparam', set_cmd=None)
    param = ArraySetPointParam()
    meas.register_parameter(scalar_param)
    meas.register_parameter(param, setpoints=(scalar_param,),
                            paramtype='array')
    np.random.seed(0)
    with meas.run() as datasaver:
        for i in range(1, 10):
            scalar_param.set(i)
            param.setpoints = (np.arange(i),)
            datasaver.add_result((scalar_param, scalar_param.get()),
                                 (param, np.random.rand(i)))
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function")
def array_in_scalar_dataset_unrolled(experiment):
    """
    This fixture yields a dataset where an array-valued parameter is registered
    as a 'numeric' type and has an additional single-valued setpoint. We
    expect data to be saved as individual scalars, with the scalar setpoint
    repeated.
    """
    meas = Measurement()
    scalar_param = Parameter('scalarparam', set_cmd=None)
    param = ArraySetPointParam()
    meas.register_parameter(scalar_param)
    meas.register_parameter(param, setpoints=(scalar_param,),
                            paramtype='numeric')

    with meas.run() as datasaver:
        for i in range(1, 10):
            scalar_param.set(i)
            datasaver.add_result((scalar_param, scalar_param.get()),
                                 (param, param.get()))
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function",
                params=["array", "numeric"])
def array_in_str_dataset(experiment, request):
    meas = Measurement()
    scalar_param = Parameter('textparam', set_cmd=None)
    param = ArraySetPointParam()
    meas.register_parameter(scalar_param, paramtype='text')
    meas.register_parameter(param, setpoints=(scalar_param,),
                            paramtype=request.param)

    with meas.run() as datasaver:
        for i in ['A', 'B', 'C']:
            scalar_param.set(i)
            datasaver.add_result((scalar_param, scalar_param.get()),
                                 (param, param.get()))
    try:
        yield datasaver.dataset
    finally:
        datasaver.dataset.conn.close()


@pytest.fixture
def standalone_parameters_dataset(dataset):
    n_params = 3
    n_rows = 10**3
    params_indep = [ParamSpecBase(f'param_{i}',
                                  'numeric',
                                  label=f'param_{i}',
                                  unit='V')
                    for i in range(n_params)]

    param_dep = ParamSpecBase(f'param_{n_params}',
                              'numeric',
                              label=f'param_{n_params}',
                              unit='Ohm')

    params_all = params_indep + [param_dep]

    idps = InterDependencies_(
        dependencies={param_dep: tuple(params_indep[0:1])},
        standalones=tuple(params_indep[1:]))

    dataset.set_interdependencies(idps)

    dataset.mark_started()
    dataset.add_results([{p.name: np.int(n_rows*10*pn+i)
                          for pn, p in enumerate(params_all)}
                         for i in range(n_rows)])
    dataset.mark_completed()
    yield dataset
