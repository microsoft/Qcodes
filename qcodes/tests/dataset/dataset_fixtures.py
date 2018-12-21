import pytest
import numpy as np
from qcodes.dataset.param_spec import ParamSpec

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
    dataset.add_results([{p.name: np.int(n_rows*10*pn+i)
                          for pn, p in enumerate(params)}
                         for i in range(n_rows)])
    dataset.mark_complete()
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
def array_in_scalar_dataset_unrolled(experiment):
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
    params_indep = [ParamSpec(f'param_{i}',
                              'numeric',
                              label=f'param_{i}',
                              unit='V')
                    for i in range(n_params)]

    params = params_indep + [ParamSpec(f'param_{n_params}',
                                       'numeric',
                                       label=f'param_{n_params}',
                                       unit='Ohm',
                                       depends_on=params_indep[0:1])]
    for p in params:
        dataset.add_parameter(p)
    dataset.add_results([{p.name: np.int(n_rows*10*pn+i)
                          for pn, p in enumerate(params)}
                         for i in range(n_rows)])
    dataset.mark_complete()
    yield dataset
