import jsonpickle

from qcodes.dataset.data_set import new_data_set, DataSet
from qcodes.dataset.param_spec import ParamSpec

from .test_dataset_basic import empty_temp_db, experiment  # noqa: F401


def test_basic_dump_and_load(experiment):  # noqa: F811

    # empty dataset

    ds = new_data_set('jsonpickletest')

    ds_string = jsonpickle.dumps(ds)
    ds_new = jsonpickle.loads(ds_string)
    assert isinstance(ds_new, DataSet)
    assert ds_new.name == ds.name
    assert ds_new.run_id == ds.run_id

    # Fill in some data and correctly read it out

    psx = ParamSpec("x", "numeric")
    psy = ParamSpec("y", "numeric", depends_on=['x'])

    ds = new_data_set("test-dataset", specs=[psx, psy])

    expected_x = []
    expected_y = []
    for x in range(100):
        expected_x.append([x])
        y = 3 * x + 10
        expected_y.append([y])
        ds.add_result({"x": x, "y": y})

    ds_string = jsonpickle.dumps(ds)
    ds_new = jsonpickle.loads(ds_string)

    assert ds_new.get_data('x') == expected_x
    assert ds_new.get_data('y') == expected_y

