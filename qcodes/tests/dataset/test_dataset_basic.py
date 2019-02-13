import itertools
from copy import copy
import re
import random

import pytest
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as hst

import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, experiments
from qcodes import load_by_id, load_by_counter
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.dependencies import InterDependencies
from qcodes.tests.common import error_caused_by
from qcodes.tests.dataset.test_descriptions import some_paramspecs
from qcodes.dataset.sqlite_base import _unicode_categories, get_non_dependencies
from qcodes.dataset.database import get_DB_location
from qcodes.dataset.data_set import CompletedError, DataSet
from qcodes.dataset.guids import parse_guid
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)
from qcodes.tests.dataset.dataset_fixtures import scalar_dataset, \
    array_dataset, multi_dataset, array_in_scalar_dataset, array_in_str_dataset, \
    standalone_parameters_dataset, array_in_scalar_dataset_unrolled
# pylint: disable=unused-import
from qcodes.tests.dataset.test_descriptions import some_paramspecs

pytest.register_assert_rewrite('qcodes.tests.dataset.helper_functions')
from qcodes.tests.dataset.helper_functions import verify_data_dict

n_experiments = 0


def make_shadow_dataset(dataset: DataSet):
    """
    Creates a new DataSet object that points to the same run_id in the same
    database file as the given dataset object.

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

    attrs = ['path_to_db', '_path_to_db', 'conn', '_run_id', 'run_id',
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


def test_dataset_read_only_properties(dataset):
    read_only_props = ['run_id', 'path_to_db', 'name', 'table_name', 'guid',
                       'number_of_results', 'counter', 'parameters',
                       'paramspecs', 'exp_id', 'exp_name', 'sample_name',
                       'run_timestamp_raw', 'completed_timestamp_raw',
                       'snapshot', 'snapshot_raw']

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
    with pytest.raises(ValueError, match="Both `path_to_db` and `conn` "
                                         "arguments have been passed together "
                                         "with non-None values. This is not "
                                         "allowed."):
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
    dataset.add_parameter(parameter_a)
    dataset.add_parameter(parameter_b)
    dataset.add_parameter(parameter_c)

    # Now retrieve the paramspecs

    shadow_ds = make_shadow_dataset(dataset)

    paramspecs = shadow_ds.paramspecs

    expected_keys = ['a_param', 'b_param', 'c_param']
    keys = sorted(list(paramspecs.keys()))
    assert keys == expected_keys
    for expected_param_name in expected_keys:
        ps = paramspecs[expected_param_name]
        assert ps.name == expected_param_name

    assert paramspecs['c_param'].inferred_from == 'a_param, b_param'

    assert paramspecs == dataset.paramspecs


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

    shadow_ds = make_shadow_dataset(dataset)

    paramspecs = shadow_ds.paramspecs

    expected_keys = ['a', 'b', 'c']
    keys = sorted(list(paramspecs.keys()))
    assert keys == expected_keys
    for expected_param_name in expected_keys:
        ps = paramspecs[expected_param_name]
        assert ps.name == expected_param_name

    assert paramspecs == dataset.paramspecs

    # Test that is not possible to add the same parameter again to the dataset
    with pytest.raises(ValueError, match=f'Duplicate parameter name: '
                                         f'{parameters[0].name}'):
        dataset.add_parameter(parameters[0])

    assert len(dataset.paramspecs.keys()) == 3
    assert len(shadow_ds.paramspecs.keys()) == 3


@pytest.mark.usefixtures("experiment")
def test_add_data_1d():
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

    shadow_ds = make_shadow_dataset(mydataset)

    assert mydataset.get_data('x') == expected_x
    assert mydataset.get_data('y') == expected_y
    assert shadow_ds.get_data('x') == expected_x
    assert shadow_ds.get_data('y') == expected_y

    with pytest.raises(ValueError):
        mydataset.add_result({'y': 500})

    assert mydataset.completed is False
    mydataset.mark_complete()
    assert mydataset.completed is True

    with pytest.raises(ValueError):
        mydataset.add_result({'y': 500})

    with pytest.raises(CompletedError):
        mydataset.add_result({'x': 5})


@pytest.mark.usefixtures("experiment")
def test_add_data_array():
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

    shadow_ds = make_shadow_dataset(mydataset)

    assert mydataset.get_data('x') == expected_x
    assert shadow_ds.get_data('x') == expected_x

    y_data = mydataset.get_data('y')
    np.testing.assert_allclose(y_data, expected_y)
    y_data = shadow_ds.get_data('y')
    np.testing.assert_allclose(y_data, expected_y)


@pytest.mark.usefixtures("experiment")
def test_adding_too_many_results():
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


def test_modify_results(dataset):
    xparam = ParamSpec("x", "numeric")
    dataset.add_parameter(xparam)
    dataset.add_result({'x': 0})
    dataset.add_result({'x': 1})

    pytest.deprecated_call(dataset.modify_results, 0, [{'x': [10]}])
    assert [[10], [1]] == dataset.get_data(xparam)

    pytest.deprecated_call(dataset.modify_results, 1, [{'x': [14]}])
    assert [[10], [14]] == dataset.get_data(xparam)

    with pytest.raises(RuntimeError,
                       match='Rolling back due to unhandled exception'):
        # not sure calling `modify_results` like this is correct, anyway it
        # is difficult to find out what the call signature for multiple
        # results is supposed to look like...
        pytest.deprecated_call(
            dataset.modify_results, 0, [{'x': [5]}, {'x': [6]}])
        assert [[5], [6]] == dataset.get_data(xparam)

    pytest.xfail('modify_results does not seem to work for cases where '
                 'multiple values of multiple parameters need to be changed. '
                 'Anyway, the signature needs to be revisited, '
                 'and consequently the correct behavior needs to be '
                 'implemented and covered with tests.')


def test_modify_result(dataset):
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

    shadow_ds = make_shadow_dataset(dataset)

    try:
        assert dataset.get_data('x')[0][0] == xdata
        assert dataset.get_data('y')[0][0] == ydata
        assert (dataset.get_data('z')[0][0] == zdata).all()

        assert shadow_ds.get_data('x')[0][0] == xdata
        assert shadow_ds.get_data('y')[0][0] == ydata
        assert (shadow_ds.get_data('z')[0][0] == zdata).all()

        with pytest.raises(ValueError):
            pytest.deprecated_call(
                dataset.modify_result, 0, {' x': 1})

        xdata = 1
        ydata = 12
        zdata = np.linspace(0, 1, 99)

        pytest.deprecated_call(dataset.modify_result, 0, {'x': xdata})
        assert dataset.get_data('x')[0][0] == xdata
        assert shadow_ds.get_data('x')[0][0] == xdata

        pytest.deprecated_call(dataset.modify_result, 0, {'y': ydata})
        assert dataset.get_data('y')[0][0] == ydata
        assert shadow_ds.get_data('y')[0][0] == ydata

        pytest.deprecated_call(dataset.modify_result, 0, {'z': zdata})
        assert (dataset.get_data('z')[0][0] == zdata).all()
        assert (shadow_ds.get_data('z')[0][0] == zdata).all()

        dataset.mark_complete()

        with pytest.raises(CompletedError):
            pytest.deprecated_call(dataset.modify_result, 0, {'x': 2})

    finally:
        shadow_ds.conn.close()


@pytest.mark.xfail(reason='This function does not seem to work the way its '
                          'docstring suggests. See the test body for more '
                          'information.')
def test_add_parameter_values(dataset):
    n = 2
    m = n + 1

    xparam = ParamSpec('x', 'numeric')
    dataset.add_parameter(xparam)

    x_results = [{'x': x} for x in range(n)]
    dataset.add_results(x_results)

    yparam = ParamSpec("y", "numeric")

    match_str = f'Need to have {n} values but got {m}.'
    match_str = re.escape(match_str)
    with pytest.raises(ValueError, match=match_str):
        pytest.deprecated_call(
            dataset.add_parameter_values, yparam, [y for y in range(m)])

    yvals = [y for y in range(n)]

    # Unlike what the docstring of the method suggests,
    # `add_parameter_values` does NOT add a new parameter and values for it
    # "NEXT TO the columns of values of existing parameters".
    #
    # In other words, if the initial state of the table is:
    #
    # |   x  |
    # --------
    # |   1  |
    # |   2  |
    #
    # then the state of the table after calling `add_parameter_values` is
    # going to be:
    #
    # |   x  |   y  |
    # ---------------
    # |   1  | NULL |
    # |   2  | NULL |
    # | NULL |  25  |
    # | NULL |  42  |
    #
    # while the docstring suggests the following state:
    #
    # |   x  |   y  |
    # ---------------
    # |   1  |  25  |
    # |   2  |  42  |
    #

    y_expected = [[None]] * n + [[y] for y in yvals]
    pytest.deprecated_call(
        dataset.add_parameter_values, yparam, yvals)

    shadow_ds = make_shadow_dataset(dataset)

    try:
        assert y_expected == dataset.get_data(yparam)
        assert y_expected == shadow_ds.get_data(yparam)

        dataset.mark_complete()

        # and now let's test that dataset's connection does not commit anymore
        # when `atomic` is used
        dataset.add_results([{yparam.name: -2}])
        y_expected_2 = y_expected + [[-2]]

        assert y_expected_2 == dataset.get_data(yparam)
        assert y_expected_2 == shadow_ds.get_data(yparam)

    finally:
        shadow_ds.conn.close()


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
        new_data_set("test-dataset",
                     specs=[ParamSpec("x", "numeric"),
                            ParamSpec("y", "numeric")])


def test_guid(dataset):
    guid = dataset.guid
    assert len(guid) == 36
    parse_guid(guid)


def test_numpy_ints(dataset):
    """
     Test that we can insert numpy integers in the data set
    """
    xparam = ParamSpec('x', 'numeric')
    dataset.add_parameter(xparam)

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
    dataset.add_parameter(float_param)

    numpy_floats = [np.float, np.float16, np.float32, np.float64]
    results = [{"y": tp(1.2)} for tp in numpy_floats]
    dataset.add_results(results)
    expected_result = [[tp(1.2)] for tp in numpy_floats]
    assert np.allclose(dataset.get_data("y"), expected_result, atol=1E-8)



def test_numpy_nan(dataset):
    parameter_m = ParamSpec("m", "numeric")
    dataset.add_parameter(parameter_m)

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


def test_get_description(experiment, some_paramspecs):

    paramspecs = some_paramspecs[2]

    ds = DataSet()

    assert ds.run_id == 1

    desc = ds.description
    assert desc == RunDescriber(InterDependencies())

    ds.add_parameter(paramspecs['ps1'])
    desc = ds.description
    assert desc == RunDescriber(InterDependencies(paramspecs['ps1']))

    ds.add_parameter(paramspecs['ps2'])
    desc = ds.description
    assert desc == RunDescriber(InterDependencies(paramspecs['ps1'],
                                                  paramspecs['ps2']))

    # the run description gets written as the first data point is added,
    # so now no description should be stored in the database
    prematurely_loaded_ds = DataSet(run_id=1)
    assert prematurely_loaded_ds.description == RunDescriber(InterDependencies())

    ds.add_result({'ps1': 1, 'ps2': 2})

    loaded_ds = DataSet(run_id=1)

    assert loaded_ds.description == desc


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


def test_the_same_dataset_as(some_paramspecs, experiment):
    paramspecs = some_paramspecs[2]
    ds = DataSet()
    ds.add_parameter(paramspecs['ps1'])
    ds.add_parameter(paramspecs['ps2'])
    ds.add_result({'ps1': 1, 'ps2': 2})

    same_ds_from_load = DataSet(run_id=ds.run_id)
    assert ds.the_same_dataset_as(same_ds_from_load)

    new_ds = DataSet()
    assert not ds.the_same_dataset_as(new_ds)


class TestGetData:
    x = ParamSpec("x", paramtype='numeric')
    n_vals = 5
    xvals = list(range(n_vals))
    # this is the format of how data is returned by DataSet.get_data
    # which means "a list of table rows"
    xdata = [[x] for x in xvals]

    @pytest.fixture(autouse=True)
    def ds_with_vals(self, dataset):
        """
        This fixture creates a DataSet with values that is to be used by all
        the tests in this class
        """
        dataset.add_parameter(self.x)
        for xv in self.xvals:
            dataset.add_result({self.x.name: xv})

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
        assert expected == ds_with_vals.get_data(self.x, start=start, end=end)


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


def test_get_array_parameter_data(array_dataset):
    paramspecs = array_dataset.paramspecs
    types = [param.type for param in paramspecs.values()]
    input_names = ['testparameter']

    expected_names = {}
    expected_names['testparameter'] = ['testparameter', 'this_setpoint']
    expected_shapes = {}
    expected_len = 5
    expected_shapes['testparameter'] = [(expected_len,), (expected_len,)]
    expected_values = {}
    expected_values['testparameter'] = [np.ones(expected_len) + 1,
                                        np.linspace(5, 9, expected_len)]
    if 'array' in types:
        expected_shapes['testparameter'] = [(1, expected_len),
                                            (1, expected_len)]
        for i in range(len(expected_values['testparameter'])):
            expected_values['testparameter'][i] = expected_values['testparameter'][i].reshape(1, expected_len)
    parameter_test_helper(array_dataset,
                          input_names,
                          expected_names,
                          expected_shapes,
                          expected_values)


def test_get_multi_parameter_data(multi_dataset):
    paramspecs = multi_dataset.paramspecs
    types = [param.type for param in paramspecs.values()]

    input_names = ['this', 'that']

    expected_names = {}
    expected_names['this'] = ['this', 'this_setpoint', 'that_setpoint']
    expected_names['that'] = ['that', 'this_setpoint', 'that_setpoint']
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
    input_names = ['testparameter']

    expected_names = {}
    expected_names['testparameter'] = ['testparameter', 'scalarparam',
                                       'this_setpoint']
    expected_shapes = {}

    shape_1 = 9
    shape_2 = 5

    test_parameter_values = np.tile((np.ones(shape_2) + 1).reshape(1, shape_2),
                                    (shape_1, 1))
    scalar_param_values = np.tile(np.arange(1, 10).reshape(shape_1, 1),
                                  (1, shape_2))
    setpoint_param_values = np.tile((np.linspace(5, 9, shape_2)).reshape(1, shape_2),
                                    (shape_1, 1))
    expected_shapes['testparameter'] = {}
    expected_shapes['testparameter'] = [(shape_1, shape_2), (shape_1, shape_2)]
    expected_values = {}
    expected_values['testparameter'] = [
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


@given(start=hst.one_of(hst.integers(1, 45), hst.none()),
       end=hst.one_of(hst.integers(1, 45), hst.none()))
def test_get_array_in_scalar_param_unrolled(array_in_scalar_dataset_unrolled,
                                            start, end):
    input_names = ['testparameter']

    expected_names = {}
    expected_names['testparameter'] = ['testparameter', 'scalarparam',
                                       'this_setpoint']
    expected_shapes = {}

    shape_1 = 9
    shape_2 = 5

    test_parameter_values = np.tile((np.ones(shape_2) + 1).reshape(1, shape_2),
                                    (shape_1, 1))
    scalar_param_values = np.tile(np.arange(1, 10).reshape(shape_1, 1),
                                  (1, shape_2))
    setpoint_param_values = np.tile((np.linspace(5, 9, shape_2)).reshape(1, shape_2),
                                    (shape_1, 1))
    expected_shapes['testparameter'] = {}
    expected_shapes['testparameter'] = [(shape_1*shape_2,), (shape_1*shape_2,)]
    expected_values = {}
    expected_values['testparameter'] = [
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

    input_names = ['testparameter']

    expected_names = {}
    expected_names['testparameter'] = ['testparameter', 'textparam',
                                       'this_setpoint']
    expected_shapes = {}

    shape_1 = 3
    shape_2 = 5

    test_parameter_values = np.tile((np.ones(shape_2) + 1).reshape(1, shape_2),
                                    (shape_1, 1))
    scalar_param_values = np.tile(np.array(['A', 'B', 'C']).reshape(shape_1, 1),
                                  (1, shape_2))
    setpoint_param_values = np.tile((np.linspace(5, 9, shape_2)).reshape(1, shape_2),
                                    (shape_1, 1))
    expected_shapes['testparameter'] = {}
    expected_values = {}

    if 'array' in types:
        expected_shapes['testparameter'] = [(3, 5), (3, 5)]
        expected_values['testparameter'] = [
            test_parameter_values,
            scalar_param_values,
            setpoint_param_values]
    else:
        expected_shapes['testparameter'] = [(15,), (15,)]
        expected_values['testparameter'] = [
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
    params = get_non_dependencies(ds.conn, ds.run_id)

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


def parameter_test_helper(ds, toplevel_names,
                          expected_names,
                          expected_shapes,
                          expected_values,
                          start=None,
                          end=None):

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
