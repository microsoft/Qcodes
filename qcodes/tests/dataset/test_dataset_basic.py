import itertools
import tempfile
import os
from contextlib import contextmanager
from copy import deepcopy
import logging

import pytest
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as hst

import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, experiments
from qcodes import load_by_id, load_by_counter

import qcodes.dataset.data_set

from qcodes.dataset.sqlite_base import (connect, _unicode_categories,
                                        get_user_version,
                                        atomic_transaction,
                                        perform_db_upgrade_0_to_1,
                                        update_GUIDs)

from qcodes.dataset.data_set import CompletedError
from qcodes.dataset.database import (initialise_database,
                                     initialise_or_create_database_at)
from qcodes.dataset.guids import parse_guid


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


@contextmanager
def location_and_station_set_to(location: int, work_station: int):
    cfg = qc.Config()
    old_cfg = deepcopy(cfg.current_config)
    cfg['GUID_components']['location'] = location
    cfg['GUID_components']['work_station'] = work_station
    cfg.save_to_home()

    yield

    cfg.current_config = old_cfg
    cfg.save_to_home()


def test_tables_exist(empty_temp_db):
    for version in [-1, 0, 1]:
        conn = connect(qc.config["core"]["db_location"],
                       qc.config["core"]["db_debug"],
                       version=version)
        cursor = conn.execute("select sql from sqlite_master"
                              " where type = 'table'")
        expected_tables = ['experiments', 'runs', 'layouts', 'dependencies']
        rows = [row for row in cursor]
        assert len(rows) == len(expected_tables)
        for row, expected_table in zip(rows, expected_tables):
            assert expected_table in row['sql']
        conn.close()


def test_initialise_database_at_for_nonexisting_db():
    with tempfile.TemporaryDirectory() as tmpdirname:
        db_location = os.path.join(tmpdirname, 'temp.db')
        assert not os.path.exists(db_location)

        initialise_or_create_database_at(db_location)

        assert os.path.exists(db_location)
        assert qc.config["core"]["db_location"] == db_location

        test_tables_exist(None)


def test_initialise_database_at_for_existing_db():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define DB location
        db_location = os.path.join(tmpdirname, 'temp.db')
        assert not os.path.exists(db_location)

        # Create DB file
        qc.config["core"]["db_location"] = db_location
        initialise_database()

        # Check if it has been created correctly
        assert os.path.exists(db_location)
        assert qc.config["core"]["db_location"] == db_location
        test_tables_exist(None)

        # Call function under test
        initialise_or_create_database_at(db_location)

        # Check if the DB is still correct
        assert os.path.exists(db_location)
        assert qc.config["core"]["db_location"] == db_location
        test_tables_exist(None)


@settings(deadline=None)
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


@settings(max_examples=25, deadline=None)
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


def test_guid(dataset):
    guid = dataset.guid
    assert len(guid) == 36
    parse_guid(guid)


def test_update_existing_guids(empty_temp_db, caplog):

    old_loc = 101
    old_ws = 1200

    new_loc = 232
    new_ws = 52123

    # prepare five runs with different location and work station codes

    with location_and_station_set_to(0, 0):
        new_experiment('test', sample_name='test_sample')

        ds1 = new_data_set('ds_one')
        xparam = ParamSpec('x', 'numeric')
        ds1.add_parameter(xparam)
        ds1.add_result({'x': 1})

        ds2 = new_data_set('ds_two')
        ds2.add_parameter(xparam)
        ds2.add_result({'x': 2})

        guid_comps_1 = parse_guid(ds1.guid)
        assert guid_comps_1['location'] == 0
        assert guid_comps_1['work_station'] == 0

        guid_comps_2 = parse_guid(ds2.guid)
        assert guid_comps_2['location'] == 0
        assert guid_comps_2['work_station'] == 0

    with location_and_station_set_to(0, old_ws):
        ds3 = new_data_set('ds_three')
        xparam = ParamSpec('x', 'numeric')
        ds3.add_parameter(xparam)
        ds3.add_result({'x': 3})

    with location_and_station_set_to(old_loc, 0):
        ds4 = new_data_set('ds_four')
        xparam = ParamSpec('x', 'numeric')
        ds4.add_parameter(xparam)
        ds4.add_result({'x': 4})

    with location_and_station_set_to(old_loc, old_ws):
        ds5 = new_data_set('ds_five')
        xparam = ParamSpec('x', 'numeric')
        ds5.add_parameter(xparam)
        ds5.add_result({'x': 5})

    with location_and_station_set_to(new_loc, new_ws):

        caplog.clear()
        expected_levels = ['INFO',
                           'INFO', 'INFO',
                           'INFO', 'INFO',
                           'INFO', 'WARNING',
                           'INFO', 'WARNING',
                           'INFO', 'INFO']

        with caplog.at_level(logging.INFO):
            update_GUIDs(ds1.conn)

            for record, lvl in zip(caplog.records, expected_levels):
                assert record.levelname == lvl

        guid_comps_1 = parse_guid(ds1.guid)
        assert guid_comps_1['location'] == new_loc
        assert guid_comps_1['work_station'] == new_ws

        guid_comps_2 = parse_guid(ds2.guid)
        assert guid_comps_2['location'] == new_loc
        assert guid_comps_2['work_station'] == new_ws

        guid_comps_3 = parse_guid(ds3.guid)
        assert guid_comps_3['location'] == 0
        assert guid_comps_3['work_station'] == old_ws

        guid_comps_4 = parse_guid(ds4.guid)
        assert guid_comps_4['location'] == old_loc
        assert guid_comps_4['work_station'] == 0

        guid_comps_5 = parse_guid(ds5.guid)
        assert guid_comps_5['location'] == old_loc
        assert guid_comps_5['work_station'] == old_ws


def test_perform_actual_upgrade_0_to_1():
    # we cannot use the empty_temp_db, since that has already called connect
    # and is therefore latest version already
    connection = connect(':memory:', debug=False,
                         version=0)

    assert get_user_version(connection) == 0

    guid_table_query = "SELECT guid FROM runs"

    with pytest.raises(RuntimeError):
        atomic_transaction(connection, guid_table_query)

    perform_db_upgrade_0_to_1(connection)
    assert get_user_version(connection) == 1

    c = atomic_transaction(connection, guid_table_query)
    assert len(c.fetchall()) == 0


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

    assert dataset.get_setpoints("a")['x'] == [[xv] for xv in xvals]

    tmp = [list(t) for t in zip(*(itertools.product(xvals, yvals)))]
    expected_setpoints = [[[v] for v in vals] for vals in tmp]

    assert dataset.get_setpoints("b")['x'] == expected_setpoints[0]
    assert dataset.get_setpoints("b")['y'] == expected_setpoints[1]

