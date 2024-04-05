from __future__ import annotations

import gc
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pytest import FixtureRequest

import qcodes as qc
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec, ParamSpecBase
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.database import connect
from qcodes.instrument_drivers.mock_instruments import (
    ArraySetPointParam,
    DummyChannelInstrument,
    DummyInstrument,
    Multi2DSetPointParam,
    Multi2DSetPointParam2Sizes,
    setpoint_generator,
)
from qcodes.parameters import ArrayParameter, Parameter, ParameterWithSetpoints
from qcodes.validators import Arrays, ComplexNumbers, Numbers

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


@pytest.fixture(scope="function", name="non_created_db")
def _make_non_created_db(tmp_path) -> Generator[None, None, None]:
    # set db location to a non existing file
    try:
        qc.config["core"]["db_location"] = str(tmp_path / "temp.db")
        if os.environ.get("QCODES_SQL_DEBUG"):
            qc.config["core"]["db_debug"] = True
        else:
            qc.config["core"]["db_debug"] = False
        yield
    finally:
        # there is a very real chance that the tests will leave open
        # connections to the database. These will have gone out of scope at
        # this stage but a gc collection may not have run. The gc
        # collection ensures that all connections belonging to now out of
        # scope objects will be closed
        gc.collect()


@pytest.fixture(scope='function')
def empty_temp_db_connection(tmp_path):
    """
    Yield connection to an empty temporary DB file.
    """
    path = str(tmp_path / 'source.db')
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()
        # there is a very real chance that the tests will leave open
        # connections to the database. These will have gone out of scope at
        # this stage but a gc collection may not have run. The gc
        # collection ensures that all connections belonging to now out of
        # scope objects will be closed
        gc.collect()


@pytest.fixture(scope='function')
def two_empty_temp_db_connections(tmp_path):
    """
    Yield connections to two empty files. Meant for use with the
    test_database_extract_runs
    """

    source_path = str(tmp_path / 'source.db')
    target_path = str(tmp_path / 'target.db')
    source_conn = connect(source_path)
    target_conn = connect(target_path)
    try:
        yield (source_conn, target_conn)
    finally:
        source_conn.close()
        target_conn.close()
        # there is a very real chance that the tests will leave open
        # connections to the database. These will have gone out of scope at
        # this stage but a gc collection may not have run. The gc
        # collection ensures that all connections belonging to now out of
        # scope objects will be closed
        gc.collect()


@contextmanager
def temporarily_copied_DB(filepath: str, **kwargs):
    """
    Make a temporary copy of a db-file and delete it after use. Meant to be
    used together with the old version database fixtures, lest we change the
    fixtures on disk. Yields the connection object

    Args:
        filepath: path to the db-file

    Kwargs:
        kwargs to be passed to connect
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dbname_new = os.path.join(tmpdir, 'temp.db')
        shutil.copy2(filepath, dbname_new)

        conn = connect(dbname_new, **kwargs)

        try:
            yield conn

        finally:
            conn.close()


@pytest.fixture(name="scalar_dataset")
def _make_scalar_dataset(dataset):
    n_params = 3
    n_rows = 10**3
    params_indep = [
        ParamSpecBase(f"param_{i}", "numeric", label=f"param_{i}", unit="V")
        for i in range(n_params)
    ]
    param_dep = ParamSpecBase(
        f"param_{n_params}", "numeric", label=f"param_{n_params}", unit="Ohm"
    )

    all_params = params_indep + [param_dep]

    idps = InterDependencies_(dependencies={param_dep: tuple(params_indep)})

    dataset.set_interdependencies(idps)
    dataset.mark_started()
    dataset.add_results(
        [
            {p.name: int(n_rows * 10 * pn + i) for pn, p in enumerate(all_params)}
            for i in range(n_rows)
        ]
    )
    dataset.mark_completed()
    yield dataset


@pytest.fixture(
    name="scalar_datasets_parameterized", params=((3, 10**3), (5, 10**3), (10, 50))
)
def _make_scalar_datasets_parameterized(dataset, request: FixtureRequest):
    n_params = request.param[0]
    n_rows = request.param[1]
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
    dataset.add_results([{p.name: int(n_rows*10*pn+i)
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
def array_dataset(experiment, request: FixtureRequest):
    meas = Measurement()
    param = ArraySetPointParam()
    meas.register_parameter(param, paramtype=request.param)

    with meas.run() as datasaver:
        datasaver.add_result((param, param.get(),))
    try:
        yield datasaver.dataset
    finally:
        assert isinstance(datasaver.dataset, DataSet)
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function",
                params=["array", "numeric"])
def array_dataset_with_nulls(experiment, request: FixtureRequest):
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
        assert isinstance(datasaver.dataset, DataSet)
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function",
                params=["array", "numeric"])
def multi_dataset(experiment, request: FixtureRequest):
    meas = Measurement()
    param = Multi2DSetPointParam()

    meas.register_parameter(param, paramtype=request.param)

    with meas.run() as datasaver:
        datasaver.add_result((param, param.get(),))
    try:
        yield datasaver.dataset
    finally:
        assert isinstance(datasaver.dataset, DataSet)
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function",
                params=["array"])
def different_setpoint_dataset(experiment, request: FixtureRequest):
    meas = Measurement()
    param = Multi2DSetPointParam2Sizes()

    meas.register_parameter(param, paramtype=request.param)

    with meas.run() as datasaver:
        datasaver.add_result((param, param.get(),))
    try:
        yield datasaver.dataset
    finally:
        assert isinstance(datasaver.dataset, DataSet)
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
        assert isinstance(datasaver.dataset, DataSet)
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
        assert isinstance(datasaver.dataset, DataSet)
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
        assert isinstance(datasaver.dataset, DataSet)
        datasaver.dataset.conn.close()


@pytest.fixture(scope="function",
                params=["array", "numeric"])
def array_in_str_dataset(experiment, request: FixtureRequest):
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
        assert isinstance(datasaver.dataset, DataSet)
        datasaver.dataset.conn.close()


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


@pytest.fixture(name="DAC")  # scope is "function" per default
def _make_dac():
    dac = DummyInstrument('dummy_dac', gates=['ch1', 'ch2'])
    yield dac
    dac.close()


@pytest.fixture(name="DAC3D")  # scope is "function" per default
def _make_dac_3d():
    dac = DummyInstrument("dummy_dac", gates=["ch1", "ch2", "ch3"])
    yield dac
    dac.close()


@pytest.fixture(name="DAC_with_metadata")  # scope is "function" per default
def _make_dac_with_metadata():
    dac = DummyInstrument('dummy_dac', gates=['ch1', 'ch2'],
                          metadata={"dac": "metadata"})
    yield dac
    dac.close()


@pytest.fixture(name="DMM")
def _make_dmm():
    dmm = DummyInstrument('dummy_dmm', gates=['v1', 'v2'])
    yield dmm
    dmm.close()


@pytest.fixture
def channel_array_instrument():
    channelarrayinstrument = DummyChannelInstrument('dummy_channel_inst')
    yield channelarrayinstrument
    channelarrayinstrument.close()


@pytest.fixture
def complex_num_instrument():

    class MyParam(Parameter):

        def get_raw(self):
            assert self.instrument is not None
            return self.instrument.setpoint() + 1j*self.instrument.setpoint()

    class RealPartParam(Parameter):

        def get_raw(self):
            assert self.instrument is not None
            return self.instrument.complex_setpoint().real

    dummyinst = DummyInstrument('dummy_channel_inst', gates=())

    dummyinst.add_parameter('setpoint',
                            parameter_class=Parameter,
                            initial_value=0,
                            label='Some Setpoint',
                            unit="Some Unit",
                            vals=Numbers(),
                            get_cmd=None, set_cmd=None)

    dummyinst.add_parameter('complex_num',
                            parameter_class=MyParam,
                            initial_value=0+0j,
                            label='Complex Num',
                            unit="complex unit",
                            vals=ComplexNumbers(),
                            get_cmd=None, set_cmd=None)

    dummyinst.add_parameter('complex_setpoint',
                            initial_value=0+0j,
                            label='Complex Setpoint',
                            unit="complex unit",
                            vals=ComplexNumbers(),
                            get_cmd=None, set_cmd=None)

    dummyinst.add_parameter('real_part',
                            parameter_class=RealPartParam,
                            label='Real Part',
                            unit="real unit",
                            vals=Numbers(),
                            set_cmd=None)

    dummyinst.add_parameter('some_array_setpoints',
                            label='Some Array Setpoints',
                            unit='some other unit',
                            vals=Arrays(shape=(5,)),
                            set_cmd=False,
                            get_cmd=lambda: np.arange(5))

    dummyinst.add_parameter('some_array',
                            parameter_class=ParameterWithSetpoints,
                            setpoints=(dummyinst.some_array_setpoints,),
                            label='Some Array',
                            unit='some_array_unit',
                            vals=Arrays(shape=(5,)),
                            get_cmd=lambda: np.ones(5),
                            set_cmd=False)

    dummyinst.add_parameter('some_complex_array_setpoints',
                            label='Some complex array setpoints',
                            unit='some_array_unit',
                            get_cmd=lambda: np.arange(5),
                            set_cmd=False)

    dummyinst.add_parameter('some_complex_array',
                            label='Some Array',
                            unit='some_array_unit',
                            get_cmd=lambda: np.ones(5) + 1j*np.ones(5),
                            set_cmd=False)

    yield dummyinst
    dummyinst.close()


@pytest.fixture
def SpectrumAnalyzer():
    """
    Yields a DummyInstrument that holds ArrayParameters returning
    different types
    """

    class BaseSpectrum(ArrayParameter):

        def __init__(self, name, instrument, **kwargs):
            super().__init__(
                name=name,
                shape=(1,),  # this attribute should be removed
                label="Flower Power Spectrum",
                unit="V/sqrt(Hz)",
                setpoint_names=("Frequency",),
                setpoint_units=("Hz",),
                instrument=instrument,
                **kwargs,
            )

            self.npts = 100
            self.start = 0
            self.stop = 2e6

        def get_data(self):
            # This is how it should be: the setpoints are generated at the
            # time we get. But that will of course not work with the old Loop
            self.setpoints = (tuple(np.linspace(self.start, self.stop,
                                                self.npts)),)
            # not the best SA on the market; it just returns noise...
            return np.random.randn(self.npts)

    class Spectrum(BaseSpectrum):
        def get_raw(self):
            return super().get_data()

    class MultiDimSpectrum(ArrayParameter):

        def __init__(self, name, instrument, **kwargs):
            self.start = 0
            self.stop = 2e6
            self.npts = (100, 50, 20)
            sp1 = np.linspace(self.start, self.stop,
                              self.npts[0])
            sp2 = np.linspace(self.start, self.stop,
                              self.npts[1])
            sp3 = np.linspace(self.start, self.stop,
                              self.npts[2])
            setpoints = setpoint_generator(sp1, sp2, sp3)

            super().__init__(
                name=name,
                instrument=instrument,
                setpoints=setpoints,
                shape=(100, 50, 20),
                label="Flower Power Spectrum in 3D",
                unit="V/sqrt(Hz)",
                setpoint_names=("Frequency0", "Frequency1", "Frequency2"),
                setpoint_units=("Hz", "Other Hz", "Third Hz"),
                **kwargs,
            )

        def get_raw(self):
            return np.random.randn(*self.npts)

    class ListSpectrum(BaseSpectrum):

        def get_raw(self):
            output = super().get_data()
            return list(output)

    class TupleSpectrum(BaseSpectrum):

        def get_raw(self):
            output = super().get_data()
            return tuple(output)

    SA = DummyInstrument('dummy_SA')
    SA.add_parameter('spectrum', parameter_class=Spectrum)
    SA.add_parameter('listspectrum', parameter_class=ListSpectrum)
    SA.add_parameter('tuplespectrum', parameter_class=TupleSpectrum)
    SA.add_parameter('multidimspectrum', parameter_class=MultiDimSpectrum)
    yield SA

    SA.close()


@pytest.fixture
def meas_with_registered_param(experiment, DAC, DMM):
    meas = Measurement()
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DMM.v1, setpoints=[DAC.ch1])
    yield meas


@pytest.fixture
def meas_with_registered_param_2d(experiment, DAC, DMM):
    meas = Measurement()
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2)
    meas.register_parameter(DMM.v1, setpoints=[DAC.ch1, DAC.ch2])
    yield meas


@pytest.fixture
def meas_with_registered_param_3d(experiment, DAC3D, DMM):
    meas = Measurement()
    meas.register_parameter(DAC3D.ch1)
    meas.register_parameter(DAC3D.ch2)
    meas.register_parameter(DAC3D.ch3)
    meas.register_parameter(DMM.v1, setpoints=[DAC3D.ch1, DAC3D.ch2, DAC3D.ch3])
    yield meas


@pytest.fixture(name="meas_with_registered_param_complex")
def _make_meas_with_registered_param_complex(experiment, DAC, complex_num_instrument):
    meas = Measurement()
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(complex_num_instrument.complex_num, setpoints=[DAC.ch1])
    yield meas


@pytest.fixture(name="dummyinstrument")
def _make_dummy_instrument() -> Iterator[DummyChannelInstrument]:
    inst = DummyChannelInstrument('dummyinstrument')
    try:
        yield inst
    finally:
        inst.close()


class ArrayshapedParam(Parameter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        assert isinstance(self.vals, Arrays)
        shape = self.vals.shape

        return np.random.rand(*shape)
