import itertools
import os
import random
import re
from copy import copy
from typing import Dict, List, Optional, Sequence, Tuple
from unittest.mock import patch

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given, settings

import qcodes as qc
from qcodes import (experiments, load_by_counter, load_by_id, new_data_set,
                    new_experiment)
from qcodes.dataset.data_set import CompletedError, DataSet
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.guids import parse_guid
from qcodes.dataset.sqlite.connection import path_to_dbfile
from qcodes.dataset.sqlite.database import get_DB_location
from qcodes.dataset.sqlite.queries import _unicode_categories
from qcodes.tests.common import error_caused_by
from qcodes.tests.dataset.test_links import generate_some_links
from qcodes.utils.deprecate import QCoDeSDeprecationWarning

pytest.register_assert_rewrite('qcodes.tests.dataset.helper_functions')
from qcodes.tests.dataset.helper_functions import verify_data_dict

n_experiments = 0


def make_shadow_dataset(dataset: DataSet):
    """
    Creates a new DataSet object that points to the same run_id in the same
    database file as the given dataset object. Note that for a pristine run,
    the shadow dataset may be out of sync with its input dataset.

    Note that in order to achieve it `path_to_db` because this will create a
    new sqlite3 connection object behind the scenes. This is very useful for
    situations where one needs to assert the underlying modifications to the
    database file.
    """

    return DataSet(path_to_db=dataset.path_to_db, run_id=dataset.run_id)


@pytest.mark.usefixtures("experiment")
def test_has_attributes_after_init():
    """
    Ensure that all attributes are populated after __init__ in BOTH cases
    (run_id is None / run_id is not None)
    """

    attrs = ['path_to_db', 'conn', '_run_id', 'run_id',
             '_debug', 'subscribers', '_completed', 'name', 'table_name',
             'guid', 'number_of_results', 'counter', 'parameters',
             'paramspecs', 'exp_id', 'exp_name', 'sample_name',
             'run_timestamp_raw', 'completed_timestamp_raw', 'completed',
             'snapshot', 'snapshot_raw']

    path_to_db = get_DB_location()
    ds = DataSet(path_to_db, run_id=None)

    for attr in attrs:
        assert hasattr(ds, attr)
        getattr(ds, attr)

    ds = DataSet(path_to_db, run_id=1)

    for attr in attrs:
        assert hasattr(ds, attr)
        getattr(ds, attr)


def test_dataset_location(empty_temp_db_connection):
    """
    Test that an dataset and experiment points to the correct db file when
    a connection is supplied.
    """
    exp = new_experiment("test", "test1", conn=empty_temp_db_connection)
    ds = DataSet(conn=empty_temp_db_connection)
    assert path_to_dbfile(empty_temp_db_connection) == \
           empty_temp_db_connection.path_to_dbfile
    assert exp.path_to_db == empty_temp_db_connection.path_to_dbfile
    assert ds.path_to_db == empty_temp_db_connection.path_to_dbfile


@pytest.mark.usefixtures("experiment")
def test_dataset_states():
    """
    Test the interplay between pristine, started, running, and completed
    """

    ds = DataSet()

    assert ds.pristine is True
    assert ds.running is False
    assert ds.started is False
    assert ds.completed is False

    with pytest.raises(RuntimeError, match='Can not mark DataSet as complete '
                                           'before it has '
                                           'been marked as started.'):
        ds.mark_completed()

    match = ('This DataSet has not been marked as started. '
             'Please mark the DataSet as started before '
             'adding results to it.')
    with pytest.raises(RuntimeError, match=match):
        ds.add_results([{'x': 1}])

    parameter = ParamSpecBase(name='single', paramtype='numeric',
                              label='', unit='N/A')
    idps = InterDependencies_(standalones=(parameter,))
    ds.set_interdependencies(idps)

    ds.mark_started()

    assert ds.pristine is False
    assert ds.running is True
    assert ds.started is True
    assert ds.completed is False

    match = ('Can not set interdependencies on a DataSet that has '
             'been started.')

    with pytest.raises(RuntimeError, match=match):
        ds.set_interdependencies(idps)

    ds.add_results([{parameter.name: 1}])

    ds.mark_completed()

    assert ds.pristine is False
    assert ds.running is False
    assert ds.started is True
    assert ds.completed is True

    match = ('Can not set interdependencies on a DataSet that has '
             'been started.')

    with pytest.raises(RuntimeError, match=match):
        ds.set_interdependencies(idps)

    match = ('This DataSet is complete, no further '
             'results can be added to it.')

    with pytest.raises(CompletedError, match=match):
        ds.add_results([{parameter.name: 1}])


@pytest.mark.parametrize("start_bg_writer", (True, False))
@pytest.mark.usefixtures('experiment')
def test_mark_completed_twice(start_bg_writer):
    """
    Ensure that its not an error to call mark_completed
    on an already completed dataset
    """
    ds = DataSet()
    ds.mark_started(start_bg_writer=start_bg_writer)
    ds.mark_completed()
    ds.mark_completed()


@pytest.mark.usefixtures('experiment')
def test_timestamps_are_none():
    ds = DataSet()

    assert ds.run_timestamp_raw is None
    assert ds.run_timestamp() is None

    ds.mark_started()

    assert isinstance(ds.run_timestamp_raw, float)
    assert isinstance(ds.run_timestamp(), str)


def test_dataset_read_only_properties(dataset):
    read_only_props = ['run_id', 'path_to_db', 'name', 'table_name', 'guid',
                       'number_of_results', 'counter', 'parameters',
                       'paramspecs', 'exp_id', 'exp_name', 'sample_name',
                       'run_timestamp_raw', 'completed_timestamp_raw',
                       'snapshot', 'snapshot_raw', 'dependent_parameters']

    # It is not expected to be possible to set readonly properties
    for prop in read_only_props:
        with pytest.raises(AttributeError, match="can't set attribute"):
            setattr(dataset, prop, True)


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("non_existing_run_id", (1, 0, -1, 'number#42'))
def test_create_dataset_from_non_existing_run_id(non_existing_run_id):
    with pytest.raises(ValueError, match=f"Run with run_id "
                                         f"{non_existing_run_id} does not "
                                         f"exist in the database"):
        _ = DataSet(run_id=non_existing_run_id)


def test_create_dataset_pass_both_connection_and_path_to_db(experiment):
    with pytest.raises(ValueError, match="Received BOTH conn and path_to_db. "
                                         "Please provide only one or "
                                         "the other."):
        some_valid_connection = experiment.conn
        _ = DataSet(path_to_db="some valid path", conn=some_valid_connection)


def test_load_by_id(dataset):
    ds = load_by_id(dataset.run_id)
    assert dataset.run_id == ds.run_id
    assert dataset.path_to_db == ds.path_to_db


@pytest.mark.usefixtures('experiment')
@pytest.mark.parametrize('non_existing_run_id', (1, 0, -1))
def test_load_by_id_for_nonexisting_run_id(non_existing_run_id):
    with pytest.raises(ValueError, match=f'Run with run_id '
                                         f'{non_existing_run_id} does not '
                                         f'exist in the database'):
        _ = load_by_id(non_existing_run_id)


@pytest.mark.usefixtures('experiment')
def test_load_by_id_for_none():
    with pytest.raises(ValueError, match='run_id has to be a positive integer, '
                                         'not None.'):
        _ = load_by_id(None)


@settings(deadline=None, max_examples=6)
@given(experiment_name=hst.text(min_size=1),
       sample_name=hst.text(min_size=1),
       dataset_name=hst.text(hst.characters(whitelist_categories=_unicode_categories),
                             min_size=1))
@pytest.mark.usefixtures("empty_temp_db")
def test_add_experiments(experiment_name,
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


@pytest.mark.usefixtures("experiment")
def test_dependent_parameters():

    pss: List[ParamSpecBase] = []

    for n in range(5):
        pss.append(ParamSpecBase(f'ps{n}', paramtype='numeric'))

    idps = InterDependencies_(dependencies={pss[0]: (pss[1], pss[2])})
    ds = DataSet(specs=idps)
    assert ds.dependent_parameters == (pss[0],)

    idps = InterDependencies_(dependencies={pss[0]: (pss[1], pss[2])},
                              standalones=(pss[3], pss[4]))
    ds = DataSet(specs=idps)
    assert ds.dependent_parameters == (pss[0],)

    idps = InterDependencies_(dependencies={pss[0]: (pss[1], pss[2]),
                                            pss[3]: (pss[4],)})

    ds = DataSet(specs=idps)
    assert ds.dependent_parameters == (pss[0], pss[3])

    idps = InterDependencies_(dependencies={pss[3]: (pss[1], pss[2]),
                                            pss[0]: (pss[4],)})

    ds = DataSet(specs=idps)
    assert ds.dependent_parameters == (pss[3], pss[0])


def test_set_interdependencies(dataset):
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 1

    parameter_a = ParamSpecBase("a_param", "NUMERIC")
    parameter_b = ParamSpecBase("b_param", "NUMERIC")
    parameter_c = ParamSpecBase("c_param", "array")

    idps = InterDependencies_(
        inferences={parameter_c: (parameter_a, parameter_b)})

    dataset.set_interdependencies(idps)

    # write the parameters to disk
    dataset.mark_started()

    # Now retrieve the paramspecs

    shadow_ds = make_shadow_dataset(dataset)

    paramspecs = shadow_ds.paramspecs

    expected_keys = ['a_param', 'b_param', 'c_param']
    keys = sorted(list(paramspecs.keys()))
    assert keys == expected_keys
    for expected_param_name in expected_keys:
        ps = paramspecs[expected_param_name]
        assert ps.name == expected_param_name

    assert paramspecs == dataset.paramspecs


@pytest.mark.usefixtures("experiment")
def test_add_data_1d():
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 0

    psx = ParamSpecBase("x", "numeric")
    psy = ParamSpecBase("y", "numeric")

    idps = InterDependencies_(dependencies={psy: (psx,)})

    mydataset = new_data_set("test-dataset")
    mydataset.set_interdependencies(idps)
    mydataset.mark_started()

    expected_x = []
    expected_y = []
    for x in range(100):
        expected_x.append(x)
        y = 3 * x + 10
        expected_y.append(y)
        mydataset.add_results([{"x": x, "y": y}])

    shadow_ds = make_shadow_dataset(mydataset)

    np.testing.assert_array_equal(mydataset.get_parameter_data()['y']['x'], expected_x)
    np.testing.assert_array_equal(mydataset.get_parameter_data()['y']['y'], expected_y)
    np.testing.assert_array_equal(shadow_ds.get_parameter_data()['y']['x'], expected_x)
    np.testing.assert_array_equal(shadow_ds.get_parameter_data()['y']['y'], expected_y)

    assert mydataset.completed is False
    mydataset.mark_completed()
    assert mydataset.completed is True

    with pytest.raises(CompletedError):
        mydataset.add_results([{'y': 500}])

    with pytest.raises(CompletedError):
        mydataset.add_results([{'x': 5}])


@pytest.mark.usefixtures("experiment")
def test_add_data_array():
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 0

    idps = InterDependencies_(
        standalones=(ParamSpecBase("x", "numeric"),
                     ParamSpecBase("y", "array")))
    mydataset = new_data_set("test")
    mydataset.set_interdependencies(idps)
    mydataset.mark_started()

    expected_x = np.arange(100)
    expected_y = []
    for x in range(100):
        y = np.random.random_sample(10)
        expected_y.append(y)
        mydataset.add_results([{"x": x, "y": y}])

    shadow_ds = make_shadow_dataset(mydataset)

    np.testing.assert_array_equal(mydataset.get_parameter_data()['x']['x'], np.array(expected_x))
    np.testing.assert_array_equal(shadow_ds.get_parameter_data()['x']['x'], np.array(expected_x))

    y_data = mydataset.get_parameter_data()['y']['y']
    np.testing.assert_allclose(y_data, expected_y)
    y_data = shadow_ds.get_parameter_data()['y']['y']
    np.testing.assert_allclose(y_data, expected_y)


@pytest.mark.usefixtures("experiment")
def test_adding_too_many_results():
    """
    This test really tests the "chunking" functionality of the
    insert_many_values function of the sqlite.query_helpers module
    """
    dataset = new_data_set("test_adding_too_many_results")
    xparam = ParamSpecBase("x", "numeric", label="x parameter",
                           unit='V')
    yparam = ParamSpecBase("y", 'numeric', label='y parameter',
                           unit='Hz')
    idps = InterDependencies_(dependencies={yparam: (xparam,)})
    dataset.set_interdependencies(idps)
    dataset.mark_started()
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


@pytest.mark.usefixtures("dataset")
def test_load_by_counter():
    exps = experiments()
    assert len(exps) == 1
    exp = exps[0]
    assert exp.name == "test-experiment"
    assert exp.sample_name == "test-sample"
    assert exp.last_counter == 1

    dataset = load_by_counter(1, 1)

    assert "test-dataset" == dataset.name
    assert exp.sample_name == dataset.sample_name
    assert exp.name == dataset.exp_name


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize('nonexisting_counter', (-1, 0, 1, None))
def test_load_by_counter_for_nonexisting_counter(nonexisting_counter):
    exp_id = 1
    with pytest.raises(RuntimeError, match='Expected one row'):
        _ = load_by_counter(exp_id, nonexisting_counter)


@pytest.mark.usefixtures("empty_temp_db")
@pytest.mark.parametrize('nonexisting_exp_id', (-1, 0, 1, None))
def test_load_by_counter_for_nonexisting_experiment(nonexisting_exp_id):
    with pytest.raises(RuntimeError, match='Expected one row'):
        _ = load_by_counter(nonexisting_exp_id, 1)


@pytest.mark.usefixtures("empty_temp_db")
def test_dataset_with_no_experiment_raises():
    with pytest.raises(ValueError):
        new_data_set("test-dataset")


def test_guid(dataset):
    guid = dataset.guid
    assert len(guid) == 36
    parse_guid(guid)


def test_numpy_ints(dataset):
    """
     Test that we can insert numpy integers in the data set
    """
    xparam = ParamSpecBase('x', 'numeric')
    idps = InterDependencies_(standalones=(xparam,))
    dataset.set_interdependencies(idps)
    dataset.mark_started()

    numpy_ints = [
        np.int, np.int8, np.int16, np.int32, np.int64,
        np.uint, np.uint8, np.uint16, np.uint32, np.uint64
    ]

    results = [{"x": tp(1)} for tp in numpy_ints]
    dataset.add_results(results)
    expected_result = np.ones(len(numpy_ints))
    np.testing.assert_array_equal(dataset.get_parameter_data()["x"]["x"], expected_result)


def test_numpy_floats(dataset):
    """
    Test that we can insert numpy floats in the data set
    """
    float_param = ParamSpecBase('y', 'numeric')
    idps = InterDependencies_(standalones=(float_param,))
    dataset.set_interdependencies(idps)
    dataset.mark_started()

    numpy_floats = [np.float, np.float16, np.float32, np.float64]
    results = [{"y": tp(1.2)} for tp in numpy_floats]
    dataset.add_results(results)
    expected_result = np.array([tp(1.2) for tp in numpy_floats])
    data = dataset.get_parameter_data()["y"]["y"]
    assert np.allclose(data, expected_result, atol=1E-8)


def test_numpy_nan(dataset):
    parameter_m = ParamSpecBase("m", "numeric")
    idps = InterDependencies_(standalones=(parameter_m,))
    dataset.set_interdependencies(idps)
    dataset.mark_started()

    data_dict = [{"m": value} for value in [0.0, np.nan, 1.0]]
    dataset.add_results(data_dict)
    retrieved = dataset.get_parameter_data()["m"]["m"]
    assert np.isnan(retrieved[1])


def test_numpy_inf(dataset):
    """
    Test that we can insert and retrieve numpy inf in the data set
    """
    parameter_m = ParamSpecBase("m", "numeric")
    idps = InterDependencies_(standalones=(parameter_m,))
    dataset.set_interdependencies(idps)
    dataset.mark_started()

    data_dict = [{"m": value} for value in [-np.inf, np.inf]]
    dataset.add_results(data_dict)
    retrieved = dataset.get_parameter_data()["m"]["m"]
    assert np.isinf(retrieved).all()


def test_missing_keys(dataset):
    """
    Test that we can now have partial results with keys missing. This is for
    example handy when having an interleaved 1D and 2D sweep.
    """

    x = ParamSpecBase("x", paramtype='numeric')
    y = ParamSpecBase("y", paramtype='numeric')
    a = ParamSpecBase("a", paramtype='numeric')
    b = ParamSpecBase("b", paramtype='numeric')

    idps = InterDependencies_(dependencies={a: (x,), b: (x, y)})
    dataset.set_interdependencies(idps)
    dataset.mark_started()

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

    loaded_data = dataset.get_parameter_data()

    np.testing.assert_array_equal(loaded_data['a']['x'],
                                  np.array(xvals))
    np.testing.assert_array_equal(loaded_data['a']['a'],
                                  np.array([fa(xv) for xv in xvals]))

    np.testing.assert_array_equal(loaded_data['b']['x'],
                                  np.repeat(np.array(xvals), 3))
    np.testing.assert_array_equal(loaded_data['b']['y'],
                                  np.tile(np.array(yvals), 3))
    np.testing.assert_array_equal(loaded_data['b']['b'],
                                  np.array([fb(xv, yv) for xv in xvals for yv in yvals]))


def test_get_description(experiment, some_interdeps):


    ds = DataSet()

    assert ds.run_id == 1

    desc = ds.description
    assert desc == RunDescriber(InterDependencies_())

    ds.set_interdependencies(some_interdeps[1])

    assert ds.description.interdeps == some_interdeps[1]

    # the run description gets written as the dataset is marked as started,
    # so now no description should be stored in the database
    prematurely_loaded_ds = DataSet(run_id=1)
    assert prematurely_loaded_ds.description == RunDescriber(
                                                    InterDependencies_())

    ds.mark_started()

    loaded_ds = DataSet(run_id=1)

    expected_desc = RunDescriber(some_interdeps[1])

    assert loaded_ds.description == expected_desc


def test_metadata(experiment, request):

    metadata1 = {'number': 1, "string": "Once upon a time..."}
    metadata2 = {'more': 'meta'}

    ds1 = DataSet(metadata=metadata1)
    request.addfinalizer(ds1.conn.close)
    ds2 = DataSet(metadata=metadata2)
    request.addfinalizer(ds2.conn.close)

    assert ds1.run_id == 1
    assert ds1.metadata == metadata1
    assert ds2.run_id == 2
    assert ds2.metadata == metadata2

    loaded_ds1 = DataSet(run_id=1)
    request.addfinalizer(loaded_ds1.conn.close)
    assert loaded_ds1.metadata == metadata1
    loaded_ds2 = DataSet(run_id=2)
    request.addfinalizer(loaded_ds2.conn.close)
    assert loaded_ds2.metadata == metadata2

    badtag = 'lex luthor'
    sorry_metadata = {'superman': 1, badtag: None, 'spiderman': 'two'}

    bad_tag_msg = (f'Tag {badtag} has value None. '
                    ' That is not a valid metadata value!')

    with pytest.raises(RuntimeError,
                       match='Rolling back due to unhandled exception') as e:
        for tag, value in sorry_metadata.items():
            ds1.add_metadata(tag, value)

    assert error_caused_by(e, bad_tag_msg)


def test_the_same_dataset_as(some_interdeps, experiment):

    ds = DataSet()
    ds.set_interdependencies(some_interdeps[1])
    ds.mark_started()
    ds.add_results([{'ps1': 1, 'ps2': 2}])

    same_ds_from_load = DataSet(run_id=ds.run_id)
    assert ds.the_same_dataset_as(same_ds_from_load)

    new_ds = DataSet()
    assert not ds.the_same_dataset_as(new_ds)


@pytest.mark.usefixtures("experiment")
def test_parent_dataset_links_invalid_input():
    """
    Test that invalid input is rejected
    """
    links = generate_some_links(3)

    ds = DataSet()

    for link in links:
        assert link.head != ds.guid

    match = re.escape('Invalid input. Did not receive a list of Links')
    with pytest.raises(ValueError, match=match):
        ds.parent_dataset_links = [ds.guid]

    match = re.escape('Invalid input. All links must point to this dataset. '
                      'Got link(s) with head(s) pointing to another dataset.')
    with pytest.raises(ValueError, match=match):
        ds.parent_dataset_links = links


@pytest.mark.usefixtures("experiment")
def test_parent_dataset_links(some_interdeps):
    """
    Test that we can set links and retrieve them when loading the dataset
    """
    links = generate_some_links(3)

    ds = DataSet()

    for link in links:
        link.head = ds.guid

    ds.set_interdependencies(some_interdeps[1])

    ds.parent_dataset_links = links[:2]
    # setting it again/overwriting it should be okay
    ds.parent_dataset_links = links

    ds.mark_started()

    match = re.escape('Can not set parent dataset links on a dataset '
                      'that has been started.')
    with pytest.raises(RuntimeError, match=match):
        ds.parent_dataset_links = links

    ds.add_results([{'ps1': 1, 'ps2': 2}])
    run_id = ds.run_id

    ds_loaded = DataSet(run_id=run_id)

    assert ds_loaded.parent_dataset_links == links


class TestGetData:
    x = ParamSpecBase("x", paramtype='numeric')
    n_vals = 5
    xvals = list(range(n_vals))
    # this is the format of how data is returned by DataSet.get_data
    # which means "a list of table rows"
    xdata = np.array(xvals)

    @pytest.fixture(autouse=True)
    def ds_with_vals(self, dataset):
        """
        This fixture creates a DataSet with values that is to be used by all
        the tests in this class
        """
        idps = InterDependencies_(standalones=(self.x,))
        dataset.set_interdependencies(idps)
        dataset.mark_started()
        for xv in self.xvals:
            dataset.add_results([{self.x.name: xv}])

        return dataset

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            # test without start and end
            (None, None, xdata),

            # test for start only
            (0, None, xdata),
            (2, None, xdata[(2-1):]),
            (-2, None, xdata),
            (n_vals, None, xdata[(n_vals-1):]),
            (n_vals + 1, None, []),
            (n_vals + 2, None, []),

            # test for end only
            (None, 0, []),
            (None, 2, xdata[:2]),
            (None, -2, []),
            (None, n_vals, xdata),
            (None, n_vals + 1, xdata),
            (None, n_vals + 2, xdata),

            # test for start and end
            (0, 0, []),
            (1, 1, [xdata[1-1]]),
            (2, 1, []),
            (2, 0, []),
            (1, 0, []),
            (n_vals, n_vals, [xdata[n_vals-1]]),
            (n_vals, n_vals - 1, []),
            (2, 4, xdata[(2-1):4]),
        ],
    )
    def test_get_data_with_start_and_end_args(self, ds_with_vals,
                                              start, end, expected):
        data = ds_with_vals.get_parameter_data(self.x, start=start, end=end)['x']
        if len(expected) == 0:
            assert data == {}
        else:
            data = data['x']
            np.testing.assert_array_equal(data, expected)


@settings(deadline=600)
@given(start=hst.one_of(hst.integers(1, 10**3), hst.none()),
       end=hst.one_of(hst.integers(1, 10**3), hst.none()))
def test_get_parameter_data(scalar_dataset, start, end):
    input_names = ['param_3']

    expected_names = {}
    expected_names['param_3'] = ['param_3', 'param_0', 'param_1', 'param_2']
    expected_shapes = {}
    expected_shapes['param_3'] = [(10**3, )]*4
    expected_values = {}
    expected_values['param_3'] = [np.arange(30000, 31000)] + \
                                 [np.arange(10000*a, 10000*a+1000)
                                  for a in range(3)]

    start, end = limit_data_to_start_end(start, end, input_names,
                                         expected_names, expected_shapes,
                                         expected_values)

    parameter_test_helper(scalar_dataset,
                          input_names,
                          expected_names,
                          expected_shapes,
                          expected_values,
                          start,
                          end)


def test_get_scalar_parameter_data_no_nulls(scalar_dataset_with_nulls):

    expected_names = {}
    expected_names['first_value'] = ['first_value', 'setpoint']
    expected_names['second_value'] = ['second_value', 'setpoint']
    expected_shapes = {}
    expected_shapes['first_value'] = [(1, ), (1,)]
    expected_shapes['second_value'] = [(1, ), (1,)]
    expected_values = {}
    expected_values['first_value'] = [np.array([1]), np.array([0])]
    expected_values['second_value'] = [np.array([2]), np.array([0])]

    parameter_test_helper(scalar_dataset_with_nulls,
                          list(expected_names.keys()),
                          expected_names,
                          expected_shapes,
                          expected_values)


def test_get_array_parameter_data_no_nulls(array_dataset_with_nulls):

    types = [p.type for p in array_dataset_with_nulls.paramspecs.values()]

    expected_names = {}
    expected_names['val1'] = ['val1', 'sp1', 'sp2']
    expected_names['val2'] = ['val2', 'sp1']
    expected_shapes = {}
    expected_values = {}

    if 'array' in types:
        shape = (1, 5)
    else:
        shape = (5,)

    expected_shapes['val1'] = [shape] * 3
    expected_shapes['val2'] = [shape] * 2
    expected_values['val1'] = [np.ones(shape),
                               np.arange(0, 5).reshape(shape),
                               np.arange(5, 10).reshape(shape)]
    expected_values['val2'] = [np.zeros(shape),
                               np.arange(0, 5).reshape(shape)]

    parameter_test_helper(array_dataset_with_nulls,
                          list(expected_names.keys()),
                          expected_names,
                          expected_shapes,
                          expected_values)


def test_get_array_parameter_data(array_dataset):
    paramspecs = array_dataset.paramspecs
    types = [param.type for param in paramspecs.values()]
    par_name = "array_setpoint_param"
    setpoint_name = "array_setpoint_param_this_setpoint"

    input_names = [par_name]

    expected_names = {}
    expected_names[par_name] = [par_name, setpoint_name]
    expected_shapes = {}
    expected_len = 5
    expected_shapes[par_name] = [(expected_len,), (expected_len,)]
    expected_values = {}
    expected_values[par_name] = [np.ones(expected_len) + 1,
                                        np.linspace(5, 9, expected_len)]
    if 'array' in types:
        expected_shapes[par_name] = [(1, expected_len),
                                     (1, expected_len)]
        for i in range(len(expected_values[par_name])):
            expected_values[par_name][i] = expected_values[par_name][i].reshape(1, expected_len)
    parameter_test_helper(array_dataset,
                          input_names,
                          expected_names,
                          expected_shapes,
                          expected_values)


def test_get_multi_parameter_data(multi_dataset):
    paramspecs = multi_dataset.paramspecs
    types = [param.type for param in paramspecs.values()]

    input_names = ['this', 'that']
    sp_names = ['multi_2d_setpoint_param_this_setpoint',
                'multi_2d_setpoint_param_that_setpoint']

    expected_names = {}
    expected_names['this'] = ['this'] + sp_names
    expected_names['that'] = ['that'] + sp_names
    expected_shapes = {}
    expected_values = {}
    shape_1 = 5
    shape_2 = 3

    this_data = np.zeros((shape_1, shape_2))
    that_data = np.ones((shape_1, shape_2))
    sp_1_data = np.tile(np.linspace(5, 9, shape_1).reshape(shape_1, 1),
                                           (1, shape_2))
    sp_2_data = np.tile(np.linspace(9, 11, shape_2), (shape_1, 1))
    if 'array' in types:
        expected_shapes['this'] = [(1, shape_1, shape_2), (1, shape_1, shape_2)]
        expected_shapes['that'] = [(1, shape_1, shape_2), (1, shape_1, shape_2)]
        expected_values['this'] = [this_data.reshape(1, shape_1, shape_2),
                                   sp_1_data.reshape(1, shape_1, shape_2),
                                   sp_2_data.reshape(1, shape_1, shape_2)]
        expected_values['that'] = [that_data.reshape(1, shape_1, shape_2),
                                   sp_1_data.reshape(1, shape_1, shape_2),
                                   sp_2_data.reshape(1, shape_1, shape_2)]

    else:
        expected_shapes['this'] = [(15,), (15,)]
        expected_shapes['that'] = [(15,), (15,)]
        expected_values['this'] = [this_data.ravel(),
                                   sp_1_data.ravel(),
                                   sp_2_data.ravel()]
        expected_values['that'] = [that_data.ravel(),
                                   sp_1_data.ravel(),
                                   sp_2_data.ravel()]
    parameter_test_helper(multi_dataset,
                          input_names,
                          expected_names,
                          expected_shapes,
                          expected_values)


@given(start=hst.one_of(hst.integers(1, 9), hst.none()),
       end=hst.one_of(hst.integers(1, 9), hst.none()))
def test_get_array_in_scalar_param_data(array_in_scalar_dataset,
                                        start, end):
    par_name = "array_setpoint_param"
    setpoint_name = "array_setpoint_param_this_setpoint"

    input_names = [par_name]

    expected_names = {}
    expected_names[par_name] = [par_name, 'scalarparam', setpoint_name]
    expected_shapes = {}

    shape_1 = 9
    shape_2 = 5

    test_parameter_values = np.tile((np.ones(shape_2) + 1).reshape(1, shape_2),
                                    (shape_1, 1))
    scalar_param_values = np.tile(np.arange(1, 10).reshape(shape_1, 1),
                                  (1, shape_2))
    setpoint_param_values = np.tile((np.linspace(5, 9, shape_2)).reshape(1, shape_2),
                                    (shape_1, 1))
    expected_shapes[par_name] = {}
    expected_shapes[par_name] = [(shape_1, shape_2), (shape_1, shape_2)]
    expected_values = {}
    expected_values[par_name] = [
        test_parameter_values,
        scalar_param_values,
        setpoint_param_values]

    start, end = limit_data_to_start_end(start, end, input_names,
                                         expected_names, expected_shapes,
                                         expected_values)
    parameter_test_helper(array_in_scalar_dataset,
                          input_names,
                          expected_names,
                          expected_shapes,
                          expected_values,
                          start,
                          end)


def test_get_varlen_array_in_scalar_param_data(varlen_array_in_scalar_dataset):
    par_name = "array_setpoint_param"
    setpoint_name = "array_setpoint_param_this_setpoint"

    input_names = [par_name]

    expected_names = {}
    expected_names[par_name] = [par_name, 'scalarparam', setpoint_name]
    expected_shapes = {}

    n = 9
    n_points = (n*(n+1))//2

    scalar_param_values = []
    setpoint_param_values = []
    for i in range(1, n + 1):
        for j in range(i):
            setpoint_param_values.append(j)
            scalar_param_values.append(i)

    np.random.seed(0)
    test_parameter_values = np.random.rand(n_points)
    scalar_param_values = np.array(scalar_param_values)
    setpoint_param_values = np.array(setpoint_param_values)

    expected_shapes[par_name] = [(n_points,), (n_points,)]
    expected_values = {}
    expected_values[par_name] = [
        test_parameter_values.ravel(),
        scalar_param_values.ravel(),
        setpoint_param_values.ravel()]

    parameter_test_helper(varlen_array_in_scalar_dataset,
                          input_names,
                          expected_names,
                          expected_shapes,
                          expected_values)


@given(start=hst.one_of(hst.integers(1, 45), hst.none()),
       end=hst.one_of(hst.integers(1, 45), hst.none()))
def test_get_array_in_scalar_param_unrolled(array_in_scalar_dataset_unrolled,
                                            start, end):
    par_name = "array_setpoint_param"
    setpoint_name = "array_setpoint_param_this_setpoint"
    input_names = [par_name]

    expected_names = {}
    expected_names[par_name] = [par_name, 'scalarparam', setpoint_name]
    expected_shapes = {}

    shape_1 = 9
    shape_2 = 5

    test_parameter_values = np.tile((np.ones(shape_2) + 1).reshape(1, shape_2),
                                    (shape_1, 1))
    scalar_param_values = np.tile(np.arange(1, 10).reshape(shape_1, 1),
                                  (1, shape_2))
    setpoint_param_values = np.tile((np.linspace(5, 9, shape_2)).reshape(1, shape_2),
                                    (shape_1, 1))
    expected_shapes[par_name] = {}
    expected_shapes[par_name] = [(shape_1*shape_2,), (shape_1*shape_2,)]
    expected_values = {}
    expected_values[par_name] = [
        test_parameter_values.ravel(),
        scalar_param_values.ravel(),
        setpoint_param_values.ravel()]

    start, end = limit_data_to_start_end(start, end, input_names,
                                         expected_names, expected_shapes,
                                         expected_values)
    parameter_test_helper(array_in_scalar_dataset_unrolled,
                          input_names,
                          expected_names,
                          expected_shapes,
                          expected_values,
                          start,
                          end)


def test_get_array_in_str_param_data(array_in_str_dataset):
    paramspecs = array_in_str_dataset.paramspecs
    types = [param.type for param in paramspecs.values()]

    par_name = "array_setpoint_param"
    setpoint_name = "array_setpoint_param_this_setpoint"

    input_names = [par_name]

    expected_names = {}
    expected_names[par_name] = [par_name, 'textparam', setpoint_name]
    expected_shapes = {}

    shape_1 = 3
    shape_2 = 5

    test_parameter_values = np.tile((np.ones(shape_2) + 1).reshape(1, shape_2),
                                    (shape_1, 1))
    scalar_param_values = np.tile(np.array(['A', 'B', 'C']).reshape(shape_1, 1),
                                  (1, shape_2))
    setpoint_param_values = np.tile((np.linspace(5, 9, shape_2)).reshape(1, shape_2),
                                    (shape_1, 1))
    expected_shapes['array_setpoint_param'] = {}
    expected_values = {}

    if 'array' in types:
        expected_shapes[par_name] = [(3, 5), (3, 5)]
        expected_values[par_name] = [
            test_parameter_values,
            scalar_param_values,
            setpoint_param_values]
    else:
        expected_shapes[par_name] = [(15,), (15,)]
        expected_values[par_name] = [
            test_parameter_values.ravel(),
            scalar_param_values.ravel(),
            setpoint_param_values.ravel()]
    parameter_test_helper(array_in_str_dataset,
                          input_names,
                          expected_names,
                          expected_shapes,
                          expected_values)


def test_get_parameter_data_independent_parameters(standalone_parameters_dataset):
    ds = standalone_parameters_dataset

    paramspecs = ds.description.interdeps.non_dependencies
    params = [ps.name for ps in paramspecs]

    expected_toplevel_params = ['param_1', 'param_2', 'param_3']
    assert params == expected_toplevel_params

    expected_names = {}
    expected_names['param_1'] = ['param_1']
    expected_names['param_2'] = ['param_2']
    expected_names['param_3'] = ['param_3', 'param_0']

    expected_shapes = {}
    expected_shapes['param_1'] = [(10 ** 3,)]
    expected_shapes['param_2'] = [(10 ** 3,)]
    expected_shapes['param_3'] = [(10**3, )]*2

    expected_values = {}
    expected_values['param_1'] = [np.arange(10000, 10000 + 1000)]
    expected_values['param_2'] = [np.arange(20000, 20000 + 1000)]
    expected_values['param_3'] = [np.arange(30000, 30000 + 1000),
                                  np.arange(0, 1000)]

    parameter_test_helper(ds,
                          expected_toplevel_params,
                          expected_names,
                          expected_shapes,
                          expected_values)


def parameter_test_helper(ds: DataSet,
                          toplevel_names: Sequence[str],
                          expected_names: Dict[str, Sequence[str]],
                          expected_shapes: Dict[str, Sequence[Tuple[int, ...]]],
                          expected_values: Dict[str, Sequence[np.ndarray]],
                          start: Optional[int] = None,
                          end: Optional[int] = None):
    """
    A helper function to compare the data we actually read out of a given
    dataset with the expected data.

    Args:
        ds: the dataset in question
        toplevel_names: names of the toplevel parameters of the dataset
        expected_names: names of the parameters expected to be loaded for a
            given parameter as a sequence indexed by the parameter.
        expected_shapes: expected shapes of the parameters loaded. The shapes
            should be stored as a tuple per parameter in a sequence containing
            all the loaded parameters for a given requested parameter.
        expected_values: expected content of the data arrays stored in a
            sequenceexpected_names:

    """

    data = ds.get_parameter_data(*toplevel_names, start=start, end=end)
    dataframe = ds.get_data_as_pandas_dataframe(*toplevel_names,
                                                start=start,
                                                end=end)

    all_data = ds.get_parameter_data(start=start, end=end)
    all_dataframe = ds.get_data_as_pandas_dataframe(start=start, end=end)

    all_parameters = list(all_data.keys())
    assert set(data.keys()).issubset(set(all_parameters))
    assert list(data.keys()) == list(dataframe.keys())
    assert len(data.keys()) == len(toplevel_names)
    assert len(dataframe.keys()) == len(toplevel_names)

    verify_data_dict(data, dataframe, toplevel_names, expected_names,
                     expected_shapes, expected_values)
    verify_data_dict(all_data, all_dataframe, toplevel_names, expected_names,
                     expected_shapes, expected_values)

    # Now lets remove a random element from the list
    # We do this one by one until there is only one element in the list
    subset_names = copy(all_parameters)
    while len(subset_names) > 1:
        elem_to_remove = random.randint(0, len(subset_names) - 1)
        name_removed = subset_names.pop(elem_to_remove)
        expected_names.pop(name_removed)
        expected_shapes.pop(name_removed)
        expected_values.pop(name_removed)

        subset_data = ds.get_parameter_data(*subset_names,
                                            start=start, end=end)
        subset_dataframe = ds.get_data_as_pandas_dataframe(*subset_names,
                                                           start=start,
                                                           end=end)
        verify_data_dict(subset_data, subset_dataframe, subset_names,
                         expected_names, expected_shapes, expected_values)


def limit_data_to_start_end(start, end, input_names, expected_names,
                            expected_shapes, expected_values):
    if not (start is None and end is None):
        if start is None:
            start = 1
        elif end is None:
            # all the shapes are the same so pick the first one
            end = expected_shapes[input_names[0]][0][0]
        if end < start:
            for name in input_names:
                expected_names[name] = []
                expected_shapes[name] = ()
                expected_values[name] = {}
        else:
            for name in input_names:
                new_shapes = []
                for shape in expected_shapes[name]:
                    shape_list = list(shape)
                    shape_list[0] = end - start + 1
                    new_shapes.append(tuple(shape_list))
                expected_shapes[name] = new_shapes
                for i in range(len(expected_values[name])):
                    expected_values[name][i] = \
                        expected_values[name][i][start - 1:end]
    return start, end


@pytest.mark.usefixtures('experiment')
def test_write_data_to_text_file_save(tmp_path_factory):
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", 'numeric')
    yparam = ParamSpecBase("y", 'numeric')
    idps = InterDependencies_(dependencies={yparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{'x': 0, 'y': 1}]
    dataset.add_results(results)
    dataset.mark_completed()

    path = str(tmp_path_factory.mktemp("write_data_to_text_file_save"))
    dataset.write_data_to_text_file(path=path)
    assert os.listdir(path) == ['y.dat']
    with open(os.path.join(path, "y.dat")) as f:
        assert f.readlines() == ['0.0\t1.0\n']


@pytest.mark.usefixtures('experiment')
def test_write_data_to_text_file_save_multi_keys(tmp_path_factory):
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", 'numeric')
    yparam = ParamSpecBase("y", 'numeric')
    zparam = ParamSpecBase("z", 'numeric')
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{'x': 0, 'y': 1, 'z': 2}]
    dataset.add_results(results)
    dataset.mark_completed()
    tmp_path = tmp_path_factory.mktemp("data_to_text_file_save_multi_keys")
    path = str(tmp_path)
    dataset.write_data_to_text_file(path=path)
    assert sorted(os.listdir(path)) == ['y.dat', 'z.dat']
    with open(os.path.join(path, "y.dat")) as f:
        assert f.readlines() == ['0.0\t1.0\n']
    with open(os.path.join(path, "z.dat")) as f:
        assert f.readlines() == ['0.0\t2.0\n']


@pytest.mark.usefixtures('experiment')
def test_write_data_to_text_file_save_single_file(tmp_path_factory):
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", 'numeric')
    yparam = ParamSpecBase("y", 'numeric')
    zparam = ParamSpecBase("z", 'numeric')
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{'x': 0, 'y': 1, 'z': 2}]
    dataset.add_results(results)
    dataset.mark_completed()
    tmp_path = tmp_path_factory.mktemp("to_text_file_save_single_file")
    path = str(tmp_path)
    dataset.write_data_to_text_file(path=path, single_file=True,
                                    single_file_name='yz')
    assert os.listdir(path) == ['yz.dat']
    with open(os.path.join(path, "yz.dat")) as f:
        assert f.readlines() == ['0.0\t1.0\t2.0\n']


@pytest.mark.usefixtures('experiment')
def test_write_data_to_text_file_length_exception(tmp_path):
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", 'numeric')
    yparam = ParamSpecBase("y", 'numeric')
    zparam = ParamSpecBase("z", 'numeric')
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results1 = [{'x': 0, 'y': 1}]
    results2 = [{'x': 0, 'z': 2}]
    results3 = [{'x': 1, 'z': 3}]
    dataset.add_results(results1)
    dataset.add_results(results2)
    dataset.add_results(results3)
    dataset.mark_completed()

    temp_dir = str(tmp_path)
    with pytest.raises(Exception, match='different length'):
        dataset.write_data_to_text_file(path=temp_dir, single_file=True,
                                        single_file_name='yz')


@pytest.mark.usefixtures('experiment')
def test_write_data_to_text_file_name_exception(tmp_path):
    dataset = new_data_set("dataset")
    xparam = ParamSpecBase("x", 'numeric')
    yparam = ParamSpecBase("y", 'numeric')
    zparam = ParamSpecBase("z", 'numeric')
    idps = InterDependencies_(dependencies={yparam: (xparam,), zparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()
    results = [{'x': 0, 'y': 1, 'z': 2}]
    dataset.add_results(results)
    dataset.mark_completed()

    temp_dir = str(tmp_path)
    with pytest.raises(Exception, match='desired file name'):
        dataset.write_data_to_text_file(path=temp_dir, single_file=True,
                                        single_file_name=None)
