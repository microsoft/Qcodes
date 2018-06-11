from hypothesis import given, settings
import hypothesis.strategies as hst
import numpy as np
import itertools

import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, experiments
from qcodes import load_by_id, load_by_counter
from qcodes.dataset.sqlite_base import connect, init_db, _unicode_categories
import qcodes.dataset.data_set
from qcodes.dataset.sqlite_base import get_user_version, set_user_version, atomic_transaction
from qcodes.dataset.data_set import CompletedError
from qcodes.dataset.database import initialise_database

import qcodes.dataset.experiment_container
import pytest
import tempfile
import os

n_experiments = 0


@pytest.fixture(scope="function")
def empty_temp_db():
    global n_experiments
    n_experiments = 0
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = True
        initialise_database()
        yield


@pytest.fixture(scope='function')
def experiment(empty_temp_db):
    e = new_experiment("test-experiment", sample_name="test-sample")
    yield e
    e.conn.close()


@pytest.fixture(scope='function')
def dataset(experiment):
    dataset = new_data_set("test-dataset")
    yield dataset
    dataset.conn.close()


def test_tabels_exists(empty_temp_db):
    print(qc.config["core"]["db_location"])
    conn = connect(qc.config["core"]["db_location"], qc.config["core"]["db_debug"])
    cursor = conn.execute("select sql from sqlite_master where type = 'table'")
    expected_tables = ['experiments', 'runs', 'layouts', 'dependencies']
    for row, expected_table in zip(cursor, expected_tables):
        assert expected_table in row['sql']
    conn.close()


@given(experiment_name=hst.text(min_size=1),
       sample_name=hst.text(min_size=1),
       dataset_name=hst.text(hst.characters(whitelist_categories=_unicode_categories),
                             min_size=1))
def test_add_experiments(empty_temp_db, experiment_name,
                         sample_name, dataset_name):
    global n_experiments
    n_experiments += 1

    _ = new_experiment(experiment_name, sample_name=sample_name)
    exps = experiments()
    assert len(exps) == n_experiments
    exp = exps[-1]
    assert exp.name == experiment_name
    assert exp.sample_name == sample_name
    assert exp.last_counter == 0

    dataset = new_data_set(dataset_name)
    dsid = dataset.run_id
    loaded_dataset = load_by_id(dsid)
    expected_ds_counter = 1
    assert loaded_dataset.name == dataset_name
    assert loaded_dataset.counter == expected_ds_counter
    assert loaded_dataset.table_name == "{}-{}-{}".format(dataset_name,
                                                          exp.exp_id,
                                                          loaded_dataset.counter)
    expected_ds_counter += 1
    dataset = new_data_set(dataset_name)
    dsid = dataset.run_id
    loaded_dataset = load_by_id(dsid)
    assert loaded_dataset.name == dataset_name
    assert loaded_dataset.counter == expected_ds_counter
    assert loaded_dataset.table_name == "{}-{}-{}".format(dataset_name,
                                                          exp.exp_id,
                                                          loaded_dataset.counter)


def test_add_paramspec(dataset):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 1

    parameter_a = ParamSpec("a_param", "NUMERIC")
    parameter_b = ParamSpec("b_param", "NUMERIC", key="value", number=1)
    parameter_c = ParamSpec("c_param", "array", inferred_from=[parameter_a,
                                                               parameter_b])
    dataset.add_parameters([parameter_a, parameter_b, parameter_c])

    # Now retrieve the paramspecs

    paramspecs = dataset.paramspecs
    expected_keys = ['a_param', 'b_param', 'c_param']
    keys = sorted(list(paramspecs.keys()))
    assert keys == expected_keys
    for expected_param_name in expected_keys:
        ps = paramspecs[expected_param_name]
        assert ps.name == expected_param_name

    assert paramspecs['c_param'].inferred_from == 'a_param, b_param'


def test_add_paramspec_one_by_one(dataset):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 1

    parameters = [ParamSpec("a", "NUMERIC"),
                  ParamSpec("b", "NUMERIC", key="value", number=1),
                  ParamSpec("c", "array")]
    for parameter in parameters:
        dataset.add_parameter(parameter)
    paramspecs = dataset.paramspecs
    expected_keys = ['a', 'b', 'c']
    keys = sorted(list(paramspecs.keys()))
    assert keys == expected_keys
    for expected_param_name in expected_keys:
        ps = paramspecs[expected_param_name]
        assert ps.name == expected_param_name

    with pytest.raises(ValueError):
        dataset.add_parameter(parameters[0])
    assert len(dataset.paramspecs.keys()) == 3


def test_add_data_1d(experiment):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 0

    psx = ParamSpec("x", "numeric")
    psy = ParamSpec("y", "numeric", depends_on=['x'])

    mydataset = new_data_set("test-dataset", specs=[psx, psy])

    expected_x = []
    expected_y = []
    for x in range(100):
        expected_x.append([x])
        y = 3 * x + 10
        expected_y.append([y])
        mydataset.add_result({"x": x, "y": y})
    assert mydataset.get_data('x') == expected_x
    assert mydataset.get_data('y') == expected_y

    with pytest.raises(ValueError):
        mydataset.add_result({'y': 500})

    assert mydataset.completed is False
    mydataset.mark_complete()
    assert mydataset.completed is True

    with pytest.raises(ValueError):
        mydataset.add_result({'y': 500})

    with pytest.raises(CompletedError):
        mydataset.add_result({'x': 5})


def test_add_data_array(experiment):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 0

    mydataset = new_data_set("test", specs=[ParamSpec("x", "numeric"),
                                            ParamSpec("y", "array")])

    expected_x = []
    expected_y = []
    for x in range(100):
        expected_x.append([x])
        y = np.random.random_sample(10)
        expected_y.append([y])
        mydataset.add_result({"x": x, "y": y})
    assert mydataset.get_data('x') == expected_x
    y_data = mydataset.get_data('y')
    np.testing.assert_allclose(y_data, expected_y)


def test_adding_too_many_results(experiment):
    """
    This test really tests the "chunking" functionality of the
    insert_many_values function of the sqlite_base module
    """
    dataset = new_data_set("test_adding_too_many_results")
    xparam = ParamSpec("x", "numeric", label="x parameter",
                       unit='V')
    yparam = ParamSpec("y", 'numeric', label='y parameter',
                       unit='Hz', depends_on=[xparam])
    dataset.add_parameter(xparam)
    dataset.add_parameter(yparam)
    n_max = qc.SQLiteSettings.limits['MAX_VARIABLE_NUMBER']

    vals = np.linspace(0, 1, int(n_max/2)+2)
    results = [{'x': val} for val in vals]
    dataset.add_results(results)

    vals = np.linspace(0, 1, int(n_max/2)+1)
    results = [{'x': val, 'y': val} for val in vals]
    dataset.add_results(results)

    vals = np.linspace(0, 1, n_max*3)
    results = [{'x': val} for val in vals]
    dataset.add_results(results)


def test_modify_result(experiment):
    dataset = new_data_set("test_modify_result")
    xparam = ParamSpec("x", "numeric", label="x parameter",
                       unit='V')
    yparam = ParamSpec("y", 'numeric', label='y parameter',
                       unit='Hz', depends_on=[xparam])
    zparam = ParamSpec("z", 'array', label='z parameter',
                       unit='sqrt(Hz)', depends_on=[xparam])
    dataset.add_parameter(xparam)
    dataset.add_parameter(yparam)
    dataset.add_parameter(zparam)

    xdata = 0
    ydata = 1
    zdata = np.linspace(0, 1, 100)

    dataset.add_result({'x': 0, 'y': 1, 'z': zdata})

    assert dataset.get_data('x')[0][0] == xdata
    assert dataset.get_data('y')[0][0] == ydata
    assert (dataset.get_data('z')[0][0] == zdata).all()

    with pytest.raises(ValueError):
        dataset.modify_result(0, {' x': 1})

    xdata = 1
    ydata = 12
    zdata = np.linspace(0, 1, 99)

    dataset.modify_result(0, {'x': xdata})
    assert dataset.get_data('x')[0][0] == xdata

    dataset.modify_result(0, {'y': ydata})
    assert dataset.get_data('y')[0][0] == ydata

    dataset.modify_result(0, {'z': zdata})
    assert (dataset.get_data('z')[0][0] == zdata).all()

    dataset.mark_complete()

    with pytest.raises(CompletedError):
        dataset.modify_result(0, {'x': 2})


@settings(max_examples=25)
@given(N=hst.integers(min_value=1, max_value=10000),
       M=hst.integers(min_value=1, max_value=10000))
def test_add_parameter_values(experiment, N, M):

    mydataset = new_data_set("test_add_parameter_values")
    xparam = ParamSpec('x', 'numeric')
    xparam.type = 'number'
    mydataset.add_parameter(xparam)

    x_results = [{'x': x} for x in range(N)]
    mydataset.add_results(x_results)

    if N != M:
        with pytest.raises(ValueError):
            mydataset.add_parameter_values(ParamSpec("y", "numeric"),
                                           [y for y in range(M)])

    mydataset.add_parameter_values(ParamSpec("y", "numeric"),
                                   [y for y in range(N)])

    mydataset.mark_complete()


def test_load_by_counter(dataset):
    dataset = load_by_counter(1, 1)
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert dataset.name == "test-dataset"
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 1


def test_dataset_with_no_experiment_raises(empty_temp_db):
    with pytest.raises(ValueError):
        new_data_set("test-dataset",
                     specs=[ParamSpec("x", "numeric"),
                            ParamSpec("y", "numeric")])


def test_database_upgrade(empty_temp_db):
    connection = connect(qc.config["core"]["db_location"],
                 qc.config["core"]["db_debug"])
    userversion = get_user_version(connection)
    if userversion != 0:
        raise RuntimeError("trying to upgrade from version 0"
                           " but your database is version"
                           " {}".format(userversion))
    sql = 'ALTER TABLE "runs" ADD COLUMN "quality"'

    atomic_transaction(connection, sql)
    set_user_version(connection, 1)


def test_numpy_ints(dataset):
    """
     Test that we can insert numpy integers in the data set
    """
    xparam = ParamSpec('x', 'numeric')
    dataset.add_parameters([xparam])

    numpy_ints = [
        np.int, np.int8, np.int16, np.int32, np.int64,
        np.uint, np.uint8, np.uint16, np.uint32, np.uint64
    ]

    results = [{"x": tp(1)} for tp in numpy_ints]
    dataset.add_results(results)
    expected_result = len(numpy_ints) * [[1]]
    assert dataset.get_data("x") == expected_result


def test_numpy_floats(dataset):
    """
    Test that we can insert numpy floats in the data set
    """
    float_param = ParamSpec('y', 'numeric')
    dataset.add_parameters([float_param])

    numpy_floats = [np.float, np.float16, np.float32, np.float64]
    results = [{"y": tp(1.2)} for tp in numpy_floats]
    dataset.add_results(results)
    expected_result = [[tp(1.2)] for tp in numpy_floats]
    assert np.allclose(dataset.get_data("y"), expected_result, atol=1E-8)


def test_numpy_nan(dataset):
    parameter_m = ParamSpec("m", "numeric")
    dataset.add_parameters([parameter_m])

    data_dict = [{"m": value} for value in [0.0, np.nan, 1.0]]
    dataset.add_results(data_dict)
    retrieved = dataset.get_data("m")
    assert np.isnan(retrieved[1])


def test_missing_keys(dataset):
    """
    Test that we can now have partial results with keys missing. This is for
    example handy when having an interleaved 1D and 2D sweep.
    """

    x = ParamSpec("x", paramtype='numeric')
    y = ParamSpec("y", paramtype='numeric')
    a = ParamSpec("a", paramtype='numeric', depends_on=[x])
    b = ParamSpec("b", paramtype='numeric', depends_on=[x, y])

    dataset.add_parameter(x)
    dataset.add_parameter(y)
    dataset.add_parameter(a)
    dataset.add_parameter(b)

    def fa(xv):
        return xv + 1

    def fb(xv, yv):
        return xv + 2 - yv * 3

    results = []
    xvals = [1, 2, 3]
    yvals = [2, 3, 4]

    for xv in xvals:
        results.append({"x": xv, "a": fa(xv)})
        for yv in yvals:
            results.append({"x": xv, "y": yv, "b": fb(xv, yv)})

    dataset.add_results(results)

    assert dataset.get_values("x") == [[r["x"]] for r in results]
    assert dataset.get_values("y") == [[r["y"]] for r in results if "y" in r]
    assert dataset.get_values("a") == [[r["a"]] for r in results if "a" in r]
    assert dataset.get_values("b") == [[r["b"]] for r in results if "b" in r]

    assert dataset.get_setpoints("a") == [[[xv] for xv in xvals]]

    tmp = [list(t) for t in zip(*(itertools.product(xvals, yvals)))]
    expected_setpoints = [[[v] for v in vals] for vals in tmp]

    assert dataset.get_setpoints("b") == expected_setpoints
