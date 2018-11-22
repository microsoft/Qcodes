import itertools

import pytest
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as hst

import qcodes as qc
from qcodes import ParamSpec, new_data_set, new_experiment, experiments
from qcodes import load_by_id, load_by_counter
from qcodes.dataset.descriptions import RunDescriber
from qcodes.dataset.dependencies import InterDependencies
from qcodes.dataset.sqlite_base import _unicode_categories
from qcodes.dataset.database import get_DB_location
from qcodes.dataset.data_set import CompletedError, DataSet
from qcodes.dataset.guids import parse_guid
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)
# pylint: disable=unused-import
from qcodes.tests.dataset.test_descriptions import some_paramspecs

n_experiments = 0

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

    for prop in read_only_props:
        with pytest.raises(AttributeError, match="can't set attribute",
                           message=f"It is not expected to be possible to set "
                                   f"property {prop!r}"):
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


@settings(deadline=None)
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

    # Test that is not possible to add the same parameter again to the dataset
    with pytest.raises(ValueError, match=f'Duplicate parameter name: '
                                         f'{parameters[0].name}'):
        dataset.add_parameter(parameters[0])

    assert len(dataset.paramspecs.keys()) == 3


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
    assert mydataset.get_data('x') == expected_x
    y_data = mydataset.get_data('y')
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


@pytest.mark.usefixtures("experiment")
def test_modify_result():
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
@pytest.mark.usefixtures("experiment")
def test_add_parameter_values(N, M):

    mydataset = new_data_set("test_add_parameter_values")
    xparam = ParamSpec('x', 'numeric')
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


@pytest.mark.usefixtures('experiment')
def test_get_description(some_paramspecs):

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


@pytest.mark.usefixtures('experiment')
def test_metadata():

    metadata1 = {'number': 1, "string": "Once upon a time..."}
    metadata2 = {'more': 'meta'}

    ds1 = DataSet(metadata=metadata1)
    ds2 = DataSet(metadata=metadata2)

    assert ds1.run_id == 1
    assert ds1.metadata == metadata1
    assert ds2.run_id == 2
    assert ds2.metadata == metadata2

    loaded_ds1 = DataSet(run_id=1)
    assert loaded_ds1.metadata == metadata1
    loaded_ds2 = DataSet(run_id=2)
    assert loaded_ds2.metadata == metadata2

    badtag = 'lex luthor'
    sorry_metadata = {'superman': 1, badtag: None, 'spiderman': 'two'}

    match = (f'Tag {badtag} has value None. '
             ' That is not a valid metadata value!')

    with pytest.raises(ValueError, match=match):
        for tag, value in sorry_metadata.items():
            ds1.add_metadata(tag, value)


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
